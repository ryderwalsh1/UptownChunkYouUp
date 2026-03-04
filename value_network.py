import numpy as np
import psyneulink as pnl
import gen_impgraph
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')

class ValueNetwork():
    def __init__(self, num_vars, num_clauses=None, graph=None, value_name='Value', hidden_size=20, learning_rate=0.5, **kwargs):
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

        # Create value network input nodes
        self.value_source_input = pnl.ProcessingMechanism(
            name=f'{value_name}_Source_Input',
            input_shapes=self.num_literals
        )

        self.value_target_input = pnl.ProcessingMechanism(
            name=f'{value_name}_Target_Input',
            input_shapes=self.num_literals
        )

        # Hidden layer receives projections from both source and target inputs
        self.value_hidden = pnl.ProcessingMechanism(
            name=f'{value_name}_Hidden',
            input_shapes=hidden_size,
            function=pnl.Logistic()
        )

        # Output layer produces scalar value estimate (logistic activation)
        self.value_output = pnl.ProcessingMechanism(
            name=f'{value_name}_Output',
            input_shapes=1,
            function=pnl.Logistic()
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
        # hidden to output (scalar output)
        if hidden_to_output_matrix is not None:
            self.hidden_to_output = pnl.MappingProjection(
                matrix=hidden_to_output_matrix
            )
        else:
            self.hidden_to_output = pnl.MappingProjection(
                matrix=(0.02*np.random.rand(hidden_size, 1) - 0.01)
            )

        # Create the value network composition with symmetric learning pathways
        value_comp = pnl.Composition(name=value_name)

        value_comp.add_linear_processing_pathway(
            [self.value_source_input, self.source_to_hidden, self.value_hidden, self.hidden_to_output, self.value_output]
        )
        value_comp.add_linear_processing_pathway(
            [self.value_target_input, self.target_to_hidden, self.value_hidden, self.hidden_to_output, self.value_output]
        )

        value_comp.add_backpropagation_learning_pathway(
            pathway=[self.value_source_input, self.value_hidden, self.value_output],
            learning_rate=learning_rate
        )
        value_comp.add_backpropagation_learning_pathway(
            pathway=[self.value_target_input, self.value_hidden, self.value_output],
            learning_rate=learning_rate
        )

        self.target_node = value_comp.nodes[f'TARGET for {value_name}_Output']

        self.value = value_comp

        training_sources = []
        training_targets = []
        expected_values = []

        for memory in self.memories:
            training_sources.append(memory['Source'])
            training_targets.append(memory['Target'])
            # For value network, we need scalar targets (could be based on path length or other metrics)
            # This is a placeholder - you'll need to provide actual value targets
            expected_values.append([0.0])  # Placeholder scalar value

        self.training_sources = np.array(training_sources)
        self.training_targets = np.array(training_targets)
        self.expected_values = np.array(expected_values)

    def show_graph(self, show_learning=False, show_nested=pnl.INSET):
        self.value.show_graph(
            show_learning= show_learning,
            show_nested= show_nested
        )

    def test_loss(self):
        total_loss = 0.0
        for i in range(len(self.memories)):
            retrieved = self.value.run(
                inputs={
                    self.value_source_input: self.training_sources[i],
                    self.value_target_input: self.training_targets[i]
                }
            )
            loss = self._mse(retrieved.flatten(), self.expected_values[i])
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
            idx: int (0 to 2*num_vars-1) from argmax of value network output

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
                print(f"Epoch {epoch}: Loss = {loss:.6f}")
                losses.append(loss)
                learning_matrices = self._capture_learning_matrices(learning_matrices)
            self.value.learn(
                inputs={
                    self.value_source_input: self.training_sources,
                    self.value_target_input: self.training_targets,
                    self.target_node: self.expected_values
                }
            )
        return losses, learning_matrices

    def _learn_on_policy(self, source_literal, target_literal, epochs=150, capture_interval=10,
                        position_update='actual', max_steps=100):
        """
        Simulate implication chain learning from source to target with sequential learning.

        The value network learns by traversing from source_literal to target_literal repeatedly.
        Each epoch is one complete trajectory. The value network only learns from the
        current source-target pair at each step, not all possible pairs.

        Args:
            source_literal: int - starting literal for each epoch
            target_literal: int - target literal (stays constant)
            epochs: int - number of complete source→target trajectories
            capture_interval: int - how often to capture loss/weights (in epochs)
            position_update: str - 'predicted' or 'actual'
                - 'actual': update literal based on correct next step, epoch ends when reaching target
                - 'predicted': update literal based on value network's predicted next step
            max_steps: int - maximum steps per epoch to prevent infinite loops

        Returns:
            tuple: (losses, learning_matrices)
                - losses: list of loss values at each capture_interval
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
        losses = []
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

                # Get value network prediction (scalar value)
                value_prediction = self.value.run(
                    inputs={
                        self.value_source_input: source_encoding,
                        self.value_target_input: target_encoding
                    }
                )

                # Get correct next step
                correct_next_literal = self.next_steps.get((current_literal, target_literal))

                # Check if we've reached the target or it's unreachable
                if correct_next_literal is None:
                    # Already at target or unreachable
                    break

                # Create target value for learning (placeholder - needs actual value estimation logic)
                target_value = np.array([0.0])  # Placeholder

                # Learn from this single instance
                self.value.learn(
                    inputs={
                        self.value_source_input: [source_encoding],
                        self.value_target_input: [target_encoding],
                        self.target_node: [target_value]
                    }
                )

                # Determine which literal to use for position update
                if position_update == 'predicted':
                    # For value network, we can't directly predict next literal from value
                    # Fall back to actual for now
                    next_literal = correct_next_literal
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

            # Capture loss and weights at intervals
            if epoch % capture_interval == 0:
                loss = self.test_loss()
                losses.append(loss)

                # Capture learning matrices
                learning_matrices = self._capture_learning_matrices(learning_matrices)

                print(f"Epoch {epoch}: Loss = {loss:.6f} ({len(visited_literals)} literals)")

        return losses, learning_matrices

    def predict(self, source, target):
        """
        Predict the value of the current state.

        Args:
            source: int or array - source literal (int) or one-hot encoding (array)
            target: int or array - target literal (int) or one-hot encoding (array)

        Returns:
            float: value network output prediction (scalar value estimate)
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

        prediction = self.value.run(
            inputs={
                self.value_source_input: source_array,
                self.value_target_input: target_array
            }
        )
        return prediction.flatten()[0]  # Return scalar value

    def update_single(self, source_encoding, target_encoding, value_target):
        """
        Perform a single learning update for one (state, goal, target) tuple.

        Args:
            source_encoding: np.array - one-hot encoding of current state
            target_encoding: np.array - one-hot encoding of goal state
            value_target: float or np.array - scalar value target
        """
        # Ensure value_target is in array form
        if isinstance(value_target, (int, float)):
            value_target = np.array([value_target])

        self.value.learn(
            inputs={
                self.value_source_input: [source_encoding],
                self.value_target_input: [target_encoding],
                self.target_node: [value_target]
            }
        )

    def _mse(self, predicted, actual):
        # Calculate the squared differences
        squared_errors = np.square(actual - predicted)

        # Calculate the mean of the squared errors
        mse = np.mean(squared_errors)

        return mse
