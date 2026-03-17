import numpy as np
import psyneulink as pnl
import gen_impgraph
from lambda_labels import lambda_values
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')

class PolicyNetwork():
    def __init__(self, num_vars, num_clauses=None, graph=None, policy_name='Policy', hidden_size=20, learning_rate=0.5, **kwargs):
        # generate structure from implication graph
        # Can either pass in a graph or generate a new one
        if graph is None:
            if num_clauses is None:
                raise ValueError("Must provide either graph or num_clauses")
            self.graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
        else:
            self.graph = graph

        self.num_vars = num_vars
        self.next_steps = gen_impgraph.compute_next_steps(self.graph, num_vars)
        self.memories, self.memory_array, self.literal_to_idx = gen_impgraph.implications_to_memories(self.graph, num_vars)

        # optional matrix specifications
        source_to_hidden_matrix = kwargs.get("source_to_hidden_matrix", None)
        target_to_hidden_matrix = kwargs.get("target_to_hidden_matrix", None)
        hidden_to_output_matrix = kwargs.get("hidden_to_output_matrix", None)

        # Create list of all literals for reference
        self.literals = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]
        self.num_literals = 2 * num_vars
        self.memory_capacity = len(self.memories)

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Create policy network input nodes
        self.policy_source_input = pnl.ProcessingMechanism(
            name=f'{policy_name}_Source_Input',
            input_shapes=self.num_literals
        )

        self.policy_target_input = pnl.ProcessingMechanism(
            name=f'{policy_name}_Target_Input',
            input_shapes=self.num_literals
        )

        # Hidden layer receives projections from both source and target inputs
        self.policy_hidden = pnl.ProcessingMechanism(
            name=f'{policy_name}_Hidden',
            input_shapes=hidden_size,
            function=pnl.Logistic()
        )

        # Output layer produces action logits/probabilities
        self.policy_output = pnl.ProcessingMechanism(
            name=f'{policy_name}_Output',
            input_shapes=self.num_literals,
            function=pnl.SoftMax(gain=10.0)
        )

        # Value output head (scalar value estimate)
        self.policy_value_output = pnl.ProcessingMechanism(
            name=f'{policy_name}_Value_Output',
            input_shapes=1,
            function=pnl.Linear()
        )

        # source input to hidden
        if source_to_hidden_matrix is not None:
            self.source_to_hidden = pnl.MappingProjection(
                matrix=source_to_hidden_matrix
            )
        else:
            self.source_to_hidden = pnl.MappingProjection(
                matrix=(0.2*np.random.rand(self.num_literals, hidden_size) - 0.1)
            )
        # target input to hidden
        if target_to_hidden_matrix is not None:
            self.target_to_hidden = pnl.MappingProjection(
                matrix=target_to_hidden_matrix
            )
        else:
            self.target_to_hidden = pnl.MappingProjection(
                matrix=(0.2*np.random.rand(self.num_literals, hidden_size) - 0.1)
            )
        # hidden to output
        if hidden_to_output_matrix is not None:
            self.hidden_to_output = pnl.MappingProjection(
                matrix=hidden_to_output_matrix
            )
        else:
            self.hidden_to_output = pnl.MappingProjection(
                matrix=(0.2*np.random.rand(hidden_size, self.num_literals) - 0.1)
            )

        # hidden to value output (scalar)
        hidden_to_value_matrix = kwargs.get("hidden_to_value_matrix", None)
        if hidden_to_value_matrix is not None:
            self.hidden_to_value = pnl.MappingProjection(
                matrix=hidden_to_value_matrix
            )
        else:
            self.hidden_to_value = pnl.MappingProjection(
                matrix=(0.02*np.random.rand(hidden_size, 1) - 0.01)
            )

        # Create the policy network composition with symmetric learning pathways
        policy_comp = pnl.Composition(name=policy_name)

        policy_comp.add_linear_processing_pathway(
            [self.policy_source_input, self.source_to_hidden, self.policy_hidden, self.hidden_to_output, self.policy_output]
        )
        policy_comp.add_linear_processing_pathway(
            [self.policy_target_input, self.target_to_hidden, self.policy_hidden, self.hidden_to_output, self.policy_output]
        )

        # Add value head processing pathways (shares hidden layer with policy head)
        policy_comp.add_linear_processing_pathway(
            [self.policy_source_input, self.source_to_hidden, self.policy_hidden, self.hidden_to_value, self.policy_value_output]
        )
        policy_comp.add_linear_processing_pathway(
            [self.policy_target_input, self.target_to_hidden, self.policy_hidden, self.hidden_to_value, self.policy_value_output]
        )

        # Policy head learning pathways
        policy_comp.add_backpropagation_learning_pathway(
            pathway=[self.policy_source_input, self.policy_hidden, self.policy_output],
            learning_rate=learning_rate
        )
        policy_comp.add_backpropagation_learning_pathway(
            pathway=[self.policy_target_input, self.policy_hidden, self.policy_output],
            learning_rate=learning_rate
        )

        # Value head learning pathways
        policy_comp.add_backpropagation_learning_pathway(
            pathway=[self.policy_source_input, self.policy_hidden, self.policy_value_output],
            learning_rate=learning_rate
        )
        policy_comp.add_backpropagation_learning_pathway(
            pathway=[self.policy_target_input, self.policy_hidden, self.policy_value_output],
            learning_rate=learning_rate
        )

        self.policy = policy_comp

        # Get target nodes using PsyNeuLink's learning pathway structure
        # The target nodes are created automatically by add_backpropagation_learning_pathway
        try:
            # Try the dictionary access first (works in some PsyNeuLink versions)
            self.target_node = policy_comp.nodes[f'TARGET for {policy_name}_Output']
            self.value_target_node = policy_comp.nodes[f'TARGET for {policy_name}_Value_Output']
        except (TypeError, KeyError):
            # Fall back to searching through nodes
            self.target_node = None
            self.value_target_node = None
            for node in policy_comp.nodes:
                if hasattr(node, 'name'):
                    if f'TARGET for {policy_name}_Output' in node.name:
                        self.target_node = node
                    elif f'TARGET for {policy_name}_Value_Output' in node.name:
                        self.value_target_node = node

        training_sources = []
        training_targets = []
        expected_answers = []

        for memory in self.memories:
            training_sources.append(memory['Source'])
            training_targets.append(memory['Target'])
            expected_answers.append(memory['Answer'])

        self.training_sources = np.array(training_sources)
        self.training_targets = np.array(training_targets)
        self.expected_answers = np.array(expected_answers)

    def show_graph(self, show_learning=False, show_nested=pnl.INSET):
        self.policy.show_graph(
            show_learning= show_learning,
            show_nested= show_nested
        )

    def test_accuracy(self):
        correct = 0
        for i in range(len(self.memories)):
            retrieved = self.policy.run(
                inputs={
                    self.policy_source_input: self.training_sources[i],
                    self.policy_target_input: self.training_targets[i]
                }
            )
            index = np.argmax(retrieved)
            expected_index = np.argmax(self.expected_answers[i])
            if index == expected_index:
                correct += 1
        accuracy = correct / len(self.memories)
        return accuracy

    def test_loss(self):
        total_loss = 0.0
        for i in range(len(self.memories)):
            retrieved = self.policy.run(
                inputs={
                    self.policy_source_input: self.training_sources[i],
                    self.policy_target_input: self.training_targets[i]
                }
            )
            loss = self._mse(retrieved.flatten(), self.expected_answers[i])
            total_loss += loss
        average_loss = total_loss / len(self.memories)
        return average_loss
    
    def _capture_learning_matrices(self, learning_matrices):
        # capture the weight matrices for visualization later
        source_to_hidden_matrix = self.source_to_hidden.matrix.base
        target_to_hidden_matrix = self.target_to_hidden.matrix.base
        hidden_to_output_matrix = self.hidden_to_output.matrix.base

        learning_matrices['source_to_hidden'].append(source_to_hidden_matrix.copy())
        learning_matrices['target_to_hidden'].append(target_to_hidden_matrix.copy())
        learning_matrices['hidden_to_output'].append(hidden_to_output_matrix.copy())

        return learning_matrices

    def _idx_to_literal(self, idx):
        """
        Convert encoding index to literal value.

        Args:
            idx: int (0 to 2*num_vars-1) from argmax of MLP output

        Returns:
            Literal value (positive or negative integer)
        """
        if idx < self.num_vars:
            return idx + 1
        else:
            return -(idx - self.num_vars + 1)

    def _get_next_literal(self, current_literal, target_literal):
        """
        Get the next literal to traverse in the implication chain.

        Args:
            current_literal: Current literal (integer)
            target_literal: Target literal (integer)

        Returns:
            Next literal on the shortest path, or None if unreachable/at target
        """
        return self.next_steps.get((current_literal, target_literal))

    def _get_trajectory_subset(self, visited_literals, target_literal):
        """
        Create training data subset for visited literals and target.

        Args:
            visited_literals: list of literals visited during trajectory
            target_literal: target literal (integer)

        Returns:
            tuple: (trajectory_sources, trajectory_targets, trajectory_answers)
                   Each is a numpy array with one-hot encodings for the trajectory
        """
        trajectory_sources = []
        trajectory_targets = []
        trajectory_answers = []

        # Get target index
        target_idx = self.literal_to_idx(target_literal)

        for source_lit in visited_literals:
            # Get source index
            source_idx = self.literal_to_idx(source_lit)

            # Create one-hot encodings
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1

            target_encoding = np.zeros(self.num_literals)
            target_encoding[target_idx] = 1

            # Get expected next step
            next_step = self.next_steps.get((source_lit, target_literal))

            if next_step is None:
                # Skip if unreachable or at target
                continue

            answer_encoding = np.zeros(self.num_literals)
            answer_idx = self.literal_to_idx(next_step)
            answer_encoding[answer_idx] = 1

            trajectory_sources.append(source_encoding)
            trajectory_targets.append(target_encoding)
            trajectory_answers.append(answer_encoding)

        return (np.array(trajectory_sources),
                np.array(trajectory_targets),
                np.array(trajectory_answers))

    def _test_trajectory_accuracy(self, trajectory_sources, trajectory_targets, trajectory_answers):
        """
        Test MLP accuracy on a specific trajectory subset.

        Args:
            trajectory_sources: numpy array of source one-hot encodings
            trajectory_targets: numpy array of target one-hot encodings
            trajectory_answers: numpy array of expected answer encodings

        Returns:
            float: accuracy as a fraction (0.0 to 1.0)
        """
        correct = 0
        num_samples = len(trajectory_sources)

        for i in range(num_samples):
            retrieved = self.policy.run(
                inputs={
                    self.policy_source_input: trajectory_sources[i],
                    self.policy_target_input: trajectory_targets[i]
                }
            )
            predicted_index = np.argmax(retrieved)
            expected_index = np.argmax(trajectory_answers[i])
            if predicted_index == expected_index:
                correct += 1

        return correct / num_samples if num_samples > 0 else 0.0
    
    def learn(self, epochs=150, capture_interval=150, on_policy=False, source_literal=None,
              target_literal=None, position_update='actual', max_steps=100):
        '''
        wrapper function for the two learning modes
        '''
        if not on_policy:
            ret = self._learn_off_policy(epochs=epochs, capture_interval=capture_interval)
        else:
            if source_literal is None or target_literal is None:
                raise ValueError("Source and target literals must be provided for on-policy learning")
            ret = self._learn_on_policy(source_literal, target_literal, epochs=epochs, capture_interval=capture_interval,
                                  position_update=position_update, max_steps=max_steps)
        return ret


    def _learn_off_policy(self, epochs=150, capture_interval=10):
        losses = []
        learning_matrices = {
            'source_to_hidden': [],
            'target_to_hidden': [],
            'hidden_to_output': []
        }
        for epoch in range(epochs):
            if epoch % capture_interval == 0:
                loss = self.test_loss()
                accuracy = self.test_accuracy()
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy*100:.2f}%")
                losses.append(loss)
                learning_matrices = self._capture_learning_matrices(learning_matrices)
            self.policy.learn(
                inputs={
                    self.policy_source_input: self.training_sources,
                    self.policy_target_input: self.training_targets,
                    self.target_node: self.expected_answers
                }
            )
        return losses, learning_matrices
    
    def _learn_on_policy(self, source_literal, target_literal, epochs=150, capture_interval=10,
                        position_update='actual', max_steps=100):
        """
        Simulate implication chain learning from source to target with sequential learning.

        The policy network learns by traversing from source_literal to target_literal repeatedly.
        Each epoch is one complete trajectory. The policy network only learns from the
        current source-target pair at each step, not all possible pairs.

        Args:
            source_literal: int - starting literal for each epoch
            target_literal: int - target literal (stays constant)
            epochs: int - number of complete source→target trajectories
            capture_interval: int - how often to capture accuracy/weights (in epochs)
            position_update: str - 'predicted' or 'actual'
                - 'actual': update literal based on correct next step, epoch ends when reaching target
                - 'predicted': update literal based on policy network's predicted next step
            max_steps: int - maximum steps per epoch to prevent infinite loops

        Returns:
            tuple: (accuracies, learning_matrices)
                - accuracies: list of accuracy values at each capture_interval
                - learning_matrices: dict with weight matrix evolution
        """
        # Validate literals exist
        if source_literal not in self.literals:
            raise ValueError(f"source_literal {source_literal} not in valid literals")
        if target_literal not in self.literals:
            raise ValueError(f"target_literal {target_literal} not in valid literals")
        if position_update not in ['predicted', 'actual']:
            raise ValueError("position_update must be 'predicted' or 'actual'")

        # Get literal indices
        source_idx = self.literal_to_idx(source_literal)
        target_idx = self.literal_to_idx(target_literal)

        # Initialize tracking
        accuracies = []
        learning_matrices = {
            'source_to_hidden': [],
            'target_to_hidden': [],
            'hidden_to_output': []
        }

        # Main training loop
        for epoch in range(epochs):
            current_literal = source_literal
            visited_literals = [source_literal]
            step_count = 0

            # Navigate from source to target
            while step_count < max_steps:
                # Get current literal index
                current_idx = self.literal_to_idx(current_literal)

                # Create one-hot encodings for current state
                source_encoding = np.zeros(self.num_literals)
                source_encoding[current_idx] = 1

                target_encoding = np.zeros(self.num_literals)
                target_encoding[target_idx] = 1

                # Get policy network prediction
                prediction = self.policy.run(
                    inputs={
                        self.policy_source_input: source_encoding,
                        self.policy_target_input: target_encoding
                    }
                )
                predicted_literal_idx = np.argmax(prediction)
                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                # Get correct next step
                correct_next_literal = self.next_steps.get((current_literal, target_literal))

                # Check if we've reached the target or it's unreachable
                if correct_next_literal is None:
                    # Already at target or unreachable
                    break

                # Create correct answer encoding for learning
                answer_encoding = np.zeros(self.num_literals)
                correct_next_idx = self.literal_to_idx(correct_next_literal)
                answer_encoding[correct_next_idx] = 1

                # Learn from this single instance
                self.policy.learn(
                    inputs={
                        self.policy_source_input: [source_encoding],
                        self.policy_target_input: [target_encoding],
                        self.target_node: [answer_encoding]
                    }
                )

                # Determine which literal to use for position update
                if position_update == 'predicted':
                    next_literal = predicted_literal
                else:  # 'actual'
                    next_literal = correct_next_literal

                # Update current literal
                current_literal = next_literal

                # Check termination condition (reached target)
                if current_literal == target_literal:
                    break

                # Track visited literals (avoid duplicates)
                if current_literal not in visited_literals:
                    visited_literals.append(current_literal)

                step_count += 1

            # Warn if max steps reached
            if step_count >= max_steps:
                print(f"Warning: Epoch {epoch} reached max_steps ({max_steps}) without reaching target")

            # Capture accuracy and weights at intervals
            if epoch % capture_interval == 0:
                # Get trajectory subset
                traj_sources, traj_targets, traj_answers = self._get_trajectory_subset(
                    visited_literals, target_literal
                )

                # Test accuracy on trajectory
                accuracy = self._test_trajectory_accuracy(traj_sources, traj_targets, traj_answers)
                accuracies.append(accuracy)

                # Capture learning matrices
                learning_matrices = self._capture_learning_matrices(learning_matrices)

                print(f"Epoch {epoch}: Trajectory accuracy = {accuracy*100:.2f}% ({len(visited_literals)} literals)")

        return accuracies, learning_matrices
    
    def _decision_entropy(self, action_vector):
        '''
        determines the entropy of the action vector.
        Args:
            action_vector: np.array - the variable from which entropy will be calculated.
        Returns:
            entropy: float - the entropy of the action_vector'''
        # Handle zero probabilities: 0 * log(0) = 0
        # Create a mask for non-zero values
        non_zero_mask = action_vector > 0

        # Calculate entropy only for non-zero probabilities
        entropy = -np.sum(action_vector[non_zero_mask] * np.log2(action_vector[non_zero_mask]))

        return entropy

    def per_step_entropy(self, path, target_literal):
        """
        Compute the policy entropy at each decision point in a trajectory.

        Args:
            path: list[int] - full trajectory including terminal state
                  (e.g., [15, -10, 3, 14])
            target_literal: int - goal node

        Returns:
            dict with:
              'entropies': list[float] - H(pi) at each non-terminal state
              'states': list[int] - the states where decisions were made
              'total_entropy': float - sum of per-step entropies (Hick's law proxy)
        """
        entropies = []
        states = []

        # Iterate through all non-terminal states (all except the last)
        for i in range(len(path) - 1):
            current_literal = path[i]
            states.append(current_literal)

            # Get policy prediction for this state
            prediction = self.predict(current_literal, target_literal)

            # Compute entropy
            entropy = self._decision_entropy(prediction.flatten())
            entropies.append(entropy)

        # Compute total entropy (sum)
        total_entropy = np.sum(entropies) if len(entropies) > 0 else 0.0

        return {
            'entropies': entropies,
            'states': states,
            'total_entropy': total_entropy
        }

    def chunking_index(self, path, target_literal):
        """
        Compute how much the trajectory behaves as a single chunk vs independent steps.

        chunking_index = 1 - (H_intermediate / H_initiation)

        where H_initiation is entropy at the first decision point and
        H_intermediate is mean entropy at all subsequent decision points.

        Args:
            path: list[int] - full trajectory including terminal state
            target_literal: int - goal node

        Returns:
            float or None:
                - 0.0 = no chunking (uniform entropy)
                - approaches 1.0 = full chunking (intermediate steps deterministic)
                - can be negative if intermediate steps are MORE uncertain than initiation
                - None if trajectory has fewer than 2 decision points or H_initiation is too small
        """
        # Get per-step entropies
        per_step_data = self.per_step_entropy(path, target_literal)
        entropies = per_step_data['entropies']

        # Need at least 2 decision points to compute chunking index
        if len(entropies) < 2:
            return None

        # Extract H_initiation (first decision point)
        H_initiation = entropies[0]

        # Handle edge case: if H_initiation is too small, index is undefined
        if H_initiation < 1e-8:
            return None

        # Extract H_intermediate (mean of all subsequent decision points)
        H_intermediate = np.mean(entropies[1:])

        # Compute chunking index
        chunking_index = 1.0 - (H_intermediate / H_initiation)

        return chunking_index
    
    def traverse_path(self, source, target, tau, force_action=True,
                      max_steps=100):
        '''
        traverses an implication chain from source to target literal. Consults the oracle
        (self.next_steps) for the correct next literal should the entropy of the policy output
        exceed a threshold tau.
        Args:
            source: int - starting literal
            target: int - target literal
            tau: float - entropy threshold for consulting the oracle
            force_action: boolean - determines whether the agent's next literal is dictated by
                the oracle (True) or its own prediction (False) in the event of confident but
                wrong decisions made by the policy network.
            max_steps: integer - maximum number of steps the policy network can take before its solution has diverged
        Returns:
            path: list[int] - list of literals traversed
            oracle_calls: list[boolean] - list of booleans corresponding to literals traversed,
                denoting when the oracle was consulted for next literal.
            accuracy: list[boolean] - list of booleans corresponding to literals traversed, denoting
                steps from which the agent predicted an incorrect next literal. If force_action is True,
                accuracy still denotes when the policy network wanted to mistep, even though its actions are
                forced to be correct.
        '''
        # Validate literals exist
        if source not in self.literals:
            raise ValueError(f"source {source} not in valid literals")
        if target not in self.literals:
            raise ValueError(f"target {target} not in valid literals")

        # Initialize tracking
        current_literal = source
        path = [source]
        oracle_calls = []
        accuracy = []
        step_count = 0

        target_idx = self.literal_to_idx(target)

        # Main navigation loop
        while step_count < max_steps:
            # Get current literal index
            current_idx = self.literal_to_idx(current_literal)

            # Create one-hot encodings for current state
            source_encoding = np.zeros(self.num_literals)
            source_encoding[current_idx] = 1

            target_encoding = np.zeros(self.num_literals)
            target_encoding[target_idx] = 1

            # Get policy network prediction
            prediction = self.policy.run(
                inputs={
                    self.policy_source_input: source_encoding,
                    self.policy_target_input: target_encoding
                }
            )
            predicted_literal_idx = np.argmax(prediction)
            predicted_literal = self._idx_to_literal(predicted_literal_idx)

            # Get correct next literal from oracle
            correct_next_literal = self.next_steps.get((current_literal, target))

            # Check if we've reached the target
            if correct_next_literal is None:
                # Already at target or unreachable
                break

            # Calculate entropy of the prediction vector
            entropy = self._decision_entropy(prediction.flatten())

            # Determine next literal to take based on entropy and force_action
            if entropy > tau:
                # Consult oracle
                oracle_calls.append(True)
                accuracy.append(True)
                next_literal = correct_next_literal
            else:
                # Don't consult oracle, use prediction based on force_action
                oracle_calls.append(False)
                is_correct = (predicted_literal == correct_next_literal)
                accuracy.append(is_correct)
                if force_action:
                    next_literal = correct_next_literal
                else:
                    next_literal = predicted_literal

            # Update current literal
            current_literal = next_literal
            path.append(current_literal)

            # Check if we've reached the target
            if current_literal == target:
                break

            step_count += 1

        # Warn if max steps reached
        if step_count >= max_steps:
            print(f"Warning: traverse_path reached max_steps ({max_steps}) without reaching target")

        return path, oracle_calls, accuracy

    def predict(self, source, target):
        """
        Predict the next literal in the implication chain.

        Args:
            source: int or array - source literal (int) or one-hot encoding (array)
            target: int or array - target literal (int) or one-hot encoding (array)

        Returns:
            numpy array: policy network output prediction (action logits/probabilities over all literals)
        """
        if isinstance(source, int) and isinstance(target, int):
            source_array = [0] * self.num_literals
            source_idx = self.literal_to_idx(source)
            source_array[source_idx] = 1

            target_array = [0] * self.num_literals
            target_idx = self.literal_to_idx(target)
            target_array[target_idx] = 1
        else:
            source_array = source
            target_array = target

        prediction = self.policy.run(
            inputs={
                self.policy_source_input: source_array,
                self.policy_target_input: target_array
            }
        )
        return prediction
    
    def update_single(self, source_encoding, target_encoding, policy_target):
        """
        Perform a single learning update for one (state, goal, target) tuple.

        Args:
            source_encoding: np.array - one-hot encoding of current state
            target_encoding: np.array - one-hot encoding of goal state
            policy_target: np.array - soft target distribution over next states
        """
        self.policy.learn(
            inputs={
                self.policy_source_input: [source_encoding],
                self.policy_target_input: [target_encoding],
                self.target_node: [policy_target]
            }
        )
    
    def update_batch(self, source_encodings, target_encodings, policy_targets):
        """
        Perform a batch learning update for multiple (state, goal, target) tuples.

        Args:
            source_encodings: np.array - shape (batch_size, num_literals) of one-hot encodings for current states
            target_encodings: np.array - shape (batch_size, num_literals) of one-hot encodings for goal states
            policy_targets: np.array - shape (batch_size, num_literals) of soft target distributions over next states
        """
        self.policy.learn(
            inputs={
                self.policy_source_input: source_encodings,
                self.policy_target_input: target_encodings,
                self.target_node: policy_targets
            }
        )

    def _mse(self, predicted, actual):
        # Calculate the squared differences
        squared_errors = np.square(actual - predicted)

        # Calculate the mean of the squared errors
        mse = np.mean(squared_errors)

        return mse

    # ========================================================================
    # Value Head Methods
    # ========================================================================

    def compute_value(self, source_literal, target_literal):
        """
        Compute value estimate for a single (state, goal) pair.

        Args:
            source_literal: int - current state literal
            target_literal: int - goal literal

        Returns:
            float - scalar value estimate
        """
        # Create one-hot encodings
        source_idx = self.literal_to_idx(source_literal)
        source_encoding = np.zeros(self.num_literals)
        source_encoding[source_idx] = 1.0

        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        # Run forward pass through value head only
        result = self.policy.run(
            inputs={
                self.policy_source_input: source_encoding,
                self.policy_target_input: target_encoding
            }
        )

        # Extract value from the value output node
        value_output = self.policy_value_output.parameters.value.get(self.policy)
        return float(value_output[0])

    def compute_trajectory_values(self, path, target_literal):
        """
        Compute value estimates for all states in a trajectory.

        Args:
            path: list[int] - trajectory of literals
            target_literal: int - goal literal

        Returns:
            np.array - value estimates for each state in path, shape (len(path),)
        """
        values = np.zeros(len(path))
        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        for i, literal in enumerate(path):
            source_idx = self.literal_to_idx(literal)
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1.0

            # Run forward pass
            self.policy.run(
                inputs={
                    self.policy_source_input: source_encoding,
                    self.policy_target_input: target_encoding
                }
            )

            # Extract value
            value_output = self.policy_value_output.parameters.value.get(self.policy)
            values[i] = float(value_output[0])

        return values

    def chunkability_from_trajectory(self, path, target_literal):
        """
        Compute chunkability (corridor-likeness) of a trajectory.

        Uses the policy head's entropy to measure how deterministic the policy is.
        Chunkability = mean(1 / exp(H_t)) across trajectory steps.

        Args:
            path: list[int] - trajectory of literals
            target_literal: int - goal literal

        Returns:
            float - chunkability metric (1.0 = corridor-like, 0.0 = diffuse)
        """
        if len(path) <= 1:
            return 0.0

        entropies = []
        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        # Compute entropy at each step (excluding the final state)
        for i in range(len(path) - 1):
            source_idx = self.literal_to_idx(path[i])
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1.0

            # Get policy prediction
            self.policy.run(
                inputs={
                    self.policy_source_input: source_encoding,
                    self.policy_target_input: target_encoding
                }
            )

            # Extract policy output (not value output)
            policy_output = self.policy_output.parameters.value.get(self.policy)

            # Compute entropy
            entropy = self._decision_entropy(policy_output.flatten())
            entropies.append(entropy)

        if len(entropies) == 0:
            return 0.0

        # Chunkability = mean(1 / exp(H_t))
        effective_actions = [np.exp(h) for h in entropies]
        chunkability_values = [1.0 / ea for ea in effective_actions]

        return np.mean(chunkability_values)

    def update_value_single(self, source_encoding, target_encoding, value_target):
        """
        Perform a single value head update for one (state, goal, value_target) tuple.

        Args:
            source_encoding: np.array - one-hot encoding of current state
            target_encoding: np.array - one-hot encoding of goal state
            value_target: float or np.array - scalar value target
        """
        # Ensure value_target is in array form
        if isinstance(value_target, (int, float)):
            value_target = np.array([value_target])

        self.policy.learn(
            inputs={
                self.policy_source_input: [source_encoding],
                self.policy_target_input: [target_encoding],
                self.value_target_node: [value_target]
            }
        )

    def update_value_batch(self, source_encodings, target_encodings, value_targets):
        """
        Perform a batch value head update for multiple (state, goal, value_target) tuples.

        Args:
            source_encodings: np.array - shape (batch_size, num_literals)
            target_encodings: np.array - shape (batch_size, num_literals)
            value_targets: np.array - shape (batch_size,) or (batch_size, 1)
        """
        # Ensure value_targets is 2D
        if len(value_targets.shape) == 1:
            value_targets = value_targets.reshape(-1, 1)

        self.policy.learn(
            inputs={
                self.policy_source_input: source_encodings,
                self.policy_target_input: target_encodings,
                self.value_target_node: value_targets
            }
        )

    def learn_combined_step(self, path, target_literal, rewards=None, gamma=0.99, lambda_exponent=2.5, update_value=True):
        """
        Perform one combined training step that trains both policy and value heads.

        Args:
            path: list[int] - trajectory of literals from source to target
            target_literal: int - goal literal
            rewards: np.array or None - reward at each step (default: sparse terminal [0,0,...,1])
            gamma: float - discount factor for TD(λ)
            lambda_exponent: float - exponent for lambda modulation (λ = chunkability^exponent)
            update_value: bool - if True, update value head; if False, only update policy head

        Returns:
            dict - diagnostics containing:
                - chunkability: float
                - lambda_: float
                - value_targets: np.array
                - value_estimates: np.array
                - policy_loss: float (optional)
                - value_loss: float (optional)
        """
        if len(path) <= 1:
            return {
                'chunkability': 0.0,
                'lambda_': 0.0,
                'value_targets': np.array([]),
                'value_estimates': np.array([])
            }

        # Default sparse terminal rewards
        if rewards is None:
            rewards = np.zeros(len(path))
            rewards[-1] = 1.0

        # 1. Train policy head with teacher-forced one-hot targets
        trajectory_sources, trajectory_targets, trajectory_answers = \
            self._get_trajectory_subset(path, target_literal)

        if len(trajectory_sources) > 0:
            self.update_batch(trajectory_sources, trajectory_targets, trajectory_answers)

        # 2. Compute current value estimates
        value_estimates = self.compute_trajectory_values(path, target_literal)

        # 3. Compute chunkability
        chunkability = self.chunkability_from_trajectory(path, target_literal)

        # 4. Derive lambda from chunkability
        lambda_ = chunkability ** lambda_exponent

        # 5. Compute TD(λ) value targets
        value_targets = lambda_values(rewards, value_estimates, gamma, lambda_)

        # 6. Train value head with TD(λ) targets
        # Prepare encodings for all states in path
        value_source_encodings = []
        value_target_encodings = []

        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        for literal in path:
            source_idx = self.literal_to_idx(literal)
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1.0

            value_source_encodings.append(source_encoding)
            value_target_encodings.append(target_encoding)

        # 6. Train value head with TD(λ) targets (if enabled)
        if update_value:
            self.update_value_batch(
                np.array(value_source_encodings),
                np.array(value_target_encodings),
                value_targets
            )

        # 7. Return diagnostics
        diagnostics = {
            'chunkability': chunkability,
            'lambda_': lambda_,
            'value_targets': value_targets,
            'value_estimates': value_estimates
        }

        return diagnostics