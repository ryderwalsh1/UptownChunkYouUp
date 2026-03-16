import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gen_impgraph
import sys
import os

# Add the Prospective-Configuration directory to the path to import predictive coding modules
pc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Prospective-Configuration', 'predictive_coding')
if pc_path not in sys.path:
    sys.path.insert(0, pc_path)

from predictive_coding import pc_layer as pc
from predictive_coding import pc_trainer

class PolicyNetworkPC():
    def __init__(self, num_vars, goal_node, num_clauses=None, graph=None, policy_name='Policy',
                 hidden_size=20, learning_rate=0.5, T=256, use_predictive_coding=False, **kwargs):
        """
        Policy network using Prospective Configuration via Predictive Coding.

        Args:
            num_vars: Number of variables in the implication graph
            goal_node: The fixed goal node
            num_clauses: Number of clauses (if generating new graph)
            graph: Pre-existing implication graph
            policy_name: Name for the policy network
            hidden_size: Number of hidden units
            learning_rate: Learning rate for parameter updates
            T: Number of inference iterations for PC (typically 256 for PC, 1 for backprop)
            use_predictive_coding: If True, use PC algorithm; if False, use standard backprop
            **kwargs: Additional arguments (e.g., source_to_hidden_matrix, hidden_to_output_matrix)
        """
        # Generate structure from implication graph
        if graph is None:
            if num_clauses is None:
                raise ValueError("Must provide either graph or num_clauses")
            self.graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
        else:
            self.graph = graph

        self.num_vars = num_vars
        self.goal_node = goal_node
        self.policy_name = policy_name

        # Compute next steps for fixed goal
        all_next_steps = gen_impgraph.compute_next_steps(self.graph, num_vars)
        # Filter to only include steps toward the fixed goal
        self.next_steps = {source: next_lit for (source, target), next_lit in all_next_steps.items() if target == goal_node}

        self.memories, self.memory_array, self.literal_to_idx = gen_impgraph.implications_to_memories(self.graph, num_vars)

        # Create list of all literals for reference
        self.literals = [i for i in range(1, num_vars + 1)] + [-i for i in range(1, num_vars + 1)]
        self.num_literals = 2 * num_vars
        self.memory_capacity = len(self.memories)

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.T = T
        self.use_predictive_coding = use_predictive_coding

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optional matrix specifications
        source_to_hidden_matrix = kwargs.get("source_to_hidden_matrix", None)
        hidden_to_output_matrix = kwargs.get("hidden_to_output_matrix", None)

        # Build the network using PyTorch
        self._build_network(source_to_hidden_matrix, hidden_to_output_matrix)

        # Prepare training data
        self._prepare_training_data()

    def _build_network(self, source_to_hidden_matrix=None, hidden_to_output_matrix=None):
        """Build the policy network with predictive coding layers."""

        # Initialize weights
        if source_to_hidden_matrix is not None:
            w1 = torch.tensor(source_to_hidden_matrix, dtype=torch.float32)
        else:
            w1 = torch.tensor(0.2 * np.random.rand(self.num_literals, self.hidden_size) - 0.1, dtype=torch.float32)

        if hidden_to_output_matrix is not None:
            w2 = torch.tensor(hidden_to_output_matrix, dtype=torch.float32)
        else:
            w2 = torch.tensor(0.2 * np.random.rand(self.hidden_size, self.num_literals) - 0.1, dtype=torch.float32)

        # Build network architecture
        if self.use_predictive_coding:
            # PC network: Linear -> PCLayer -> Activation -> Linear -> PCLayer -> SoftMax
            linear1 = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            linear1.weight.data = w1.T  # PyTorch uses transposed weights

            pc_layer1 = pc.PCLayer()

            linear2 = nn.Linear(self.hidden_size, self.num_literals, bias=False)
            linear2.weight.data = w2.T

            pc_layer2 = pc.PCLayer()

            self.model = nn.Sequential(
                linear1,
                pc_layer1,
                nn.Sigmoid(),  # Using Sigmoid like PsyNeuLink's Logistic
                linear2,
                pc_layer2,
            )

            # Create PC trainer
            self.pc_trainer = pc_trainer.PCTrainer(
                model=self.model,
                optimizer_x_fn=optim.SGD,
                optimizer_x_kwargs={'lr': 0.1},
                optimizer_p_fn=optim.SGD,
                optimizer_p_kwargs={'lr': self.learning_rate},
                T=self.T,
                update_x_at='all',
                update_p_at='last',  # Update parameters only at the end of inference
                plot_progress_at=[],  # Disable plotting to avoid interactive prompts
            )
        else:
            # Standard backprop network (T=1)
            linear1 = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            linear1.weight.data = w1.T

            linear2 = nn.Linear(self.hidden_size, self.num_literals, bias=False)
            linear2.weight.data = w2.T

            self.model = nn.Sequential(
                linear1,
                nn.Sigmoid(),
                linear2,
            )

            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.model.to(self.device)

    def _prepare_training_data(self):
        """Prepare training data filtered for the fixed goal."""
        training_sources = []
        expected_answers = []

        # Filter training data to only include examples for the fixed goal
        goal_idx = self.literal_to_idx(self.goal_node)
        for memory in self.memories:
            # Check if this memory's target matches our goal
            if np.argmax(memory['Target']) == goal_idx:
                training_sources.append(memory['Source'])
                expected_answers.append(memory['Answer'])

        self.training_sources = torch.tensor(np.array(training_sources), dtype=torch.float32).to(self.device)
        self.expected_answers = torch.tensor(np.array(expected_answers), dtype=torch.float32).to(self.device)

    def test_accuracy(self):
        """Test accuracy on the training set."""
        correct = 0
        num_examples = len(self.training_sources)
        if num_examples == 0:
            return 0.0

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i in range(num_examples):
                source = self.training_sources[i:i+1]
                output = self.model(source)
                # Apply softmax with gain like PsyNeuLink (for both PC and backprop)
                output = torch.softmax(output * 10.0, dim=1)

                predicted_index = torch.argmax(output).item()
                expected_index = torch.argmax(self.expected_answers[i]).item()
                if predicted_index == expected_index:
                    correct += 1

        # Restore previous training state
        if was_training:
            self.model.train()

        return correct / num_examples

    def test_loss(self):
        """Test MSE loss on the training set."""
        total_loss = 0.0
        num_examples = len(self.training_sources)
        if num_examples == 0:
            return 0.0

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i in range(num_examples):
                source = self.training_sources[i:i+1]
                output = self.model(source)
                # Apply softmax (for both PC and backprop)
                output = torch.softmax(output * 10.0, dim=1)

                loss = torch.mean((output - self.expected_answers[i:i+1])**2)
                total_loss += loss.item()

        # Restore previous training state
        if was_training:
            self.model.train()

        return total_loss / num_examples

    def _capture_learning_matrices(self, learning_matrices):
        """Capture the weight matrices for visualization."""
        if self.use_predictive_coding:
            # Extract weights from the Linear layers
            source_to_hidden_matrix = self.model[0].weight.data.T.cpu().numpy()
            hidden_to_output_matrix = self.model[3].weight.data.T.cpu().numpy()
        else:
            source_to_hidden_matrix = self.model[0].weight.data.T.cpu().numpy()
            hidden_to_output_matrix = self.model[2].weight.data.T.cpu().numpy()

        learning_matrices['source_to_hidden'].append(source_to_hidden_matrix.copy())
        learning_matrices['hidden_to_output'].append(hidden_to_output_matrix.copy())

        return learning_matrices

    def _idx_to_literal(self, idx):
        """Convert encoding index to literal value."""
        if idx < self.num_vars:
            return idx + 1
        else:
            return -(idx - self.num_vars + 1)

    def _get_next_literal(self, current_literal):
        """Get the next literal to traverse in the implication chain toward the fixed goal."""
        return self.next_steps.get(current_literal)

    def _get_trajectory_subset(self, visited_literals):
        """Create training data subset for visited literals (with fixed goal)."""
        trajectory_sources = []
        trajectory_answers = []

        for source_lit in visited_literals:
            source_idx = self.literal_to_idx(source_lit)
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1

            next_step = self.next_steps.get(source_lit)
            if next_step is None:
                continue

            answer_encoding = np.zeros(self.num_literals)
            answer_idx = self.literal_to_idx(next_step)
            answer_encoding[answer_idx] = 1

            trajectory_sources.append(source_encoding)
            trajectory_answers.append(answer_encoding)

        return (np.array(trajectory_sources), np.array(trajectory_answers))

    def _test_trajectory_accuracy(self, trajectory_sources, trajectory_answers):
        """Test accuracy on a specific trajectory subset."""
        correct = 0
        num_samples = len(trajectory_sources)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                source = torch.tensor(trajectory_sources[i:i+1], dtype=torch.float32).to(self.device)
                output = self.model(source)
                # Apply softmax (for both PC and backprop)
                output = torch.softmax(output * 10.0, dim=1)

                predicted_index = torch.argmax(output).item()
                expected_index = np.argmax(trajectory_answers[i])
                if predicted_index == expected_index:
                    correct += 1

        # Restore previous training state
        if was_training:
            self.model.train()

        return correct / num_samples if num_samples > 0 else 0.0

    def learn(self, epochs=150, capture_interval=150, on_policy=False, source_literal=None,
              position_update='actual', max_steps=100):
        """Wrapper function for the two learning modes."""
        if not on_policy:
            ret = self._learn_off_policy(epochs=epochs, capture_interval=capture_interval)
        else:
            if source_literal is None:
                raise ValueError("Source literal must be provided for on-policy learning")
            ret = self._learn_on_policy(source_literal, epochs=epochs, capture_interval=capture_interval,
                                  position_update=position_update, max_steps=max_steps)
        return ret

    def _learn_off_policy(self, epochs=150, capture_interval=10):
        """Off-policy learning using the full training set."""
        losses = []
        learning_matrices = {
            'source_to_hidden': [],
            'hidden_to_output': []
        }

        self.model.train()

        for epoch in range(epochs):
            if epoch % capture_interval == 0:
                loss = self.test_loss()
                accuracy = self.test_accuracy()
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy*100:.2f}%")
                losses.append(loss)
                learning_matrices = self._capture_learning_matrices(learning_matrices)

            if self.use_predictive_coding:
                # Use PC training
                def loss_fn(output, target):
                    # Apply softmax to output for loss calculation
                    output_softmax = torch.softmax(output * 10.0, dim=1)
                    return torch.mean((output_softmax - target)**2)

                self.pc_trainer.train_on_batch(
                    inputs=self.training_sources,
                    loss_fn=loss_fn,
                    loss_fn_kwargs={'target': self.expected_answers},
                    is_log_progress=False,
                    is_return_results_every_t=False,
                )
            else:
                # Use standard backprop
                self.optimizer.zero_grad()
                output = self.model(self.training_sources)
                output_softmax = torch.softmax(output * 10.0, dim=1)
                loss = torch.mean((output_softmax - self.expected_answers)**2)
                loss.backward()
                self.optimizer.step()

        return losses, learning_matrices

    def _learn_on_policy(self, source_literal, epochs=150, capture_interval=10,
                        position_update='actual', max_steps=100):
        """On-policy learning by traversing from source to goal repeatedly."""
        if source_literal not in self.literals:
            raise ValueError(f"source_literal {source_literal} not in valid literals")
        if self.goal_node not in self.literals:
            raise ValueError(f"goal_node {self.goal_node} not in valid literals")
        if position_update not in ['predicted', 'actual']:
            raise ValueError("position_update must be 'predicted' or 'actual'")

        source_idx = self.literal_to_idx(source_literal)
        accuracies = []
        learning_matrices = {
            'source_to_hidden': [],
            'hidden_to_output': []
        }

        self.model.train()

        for epoch in range(epochs):
            current_literal = source_literal
            visited_literals = [source_literal]
            step_count = 0

            while step_count < max_steps:
                current_idx = self.literal_to_idx(current_literal)
                source_encoding = np.zeros(self.num_literals)
                source_encoding[current_idx] = 1
                source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)

                # Get policy network prediction (temporarily switch to eval mode)
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(source_tensor)
                    # Apply softmax (for both PC and backprop)
                    prediction = torch.softmax(prediction * 10.0, dim=1)
                    predicted_literal_idx = torch.argmax(prediction).item()
                if was_training:
                    self.model.train()

                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                # Get correct next step toward goal
                correct_next_literal = self.next_steps.get(current_literal)

                if correct_next_literal is None:
                    break

                # Create correct answer encoding for learning
                answer_encoding = np.zeros(self.num_literals)
                correct_next_idx = self.literal_to_idx(correct_next_literal)
                answer_encoding[correct_next_idx] = 1
                answer_tensor = torch.tensor([answer_encoding], dtype=torch.float32).to(self.device)

                # Learn from this single instance
                if self.use_predictive_coding:
                    def loss_fn(output, target):
                        output_softmax = torch.softmax(output * 10.0, dim=1)
                        return torch.mean((output_softmax - target)**2)

                    self.pc_trainer.train_on_batch(
                        inputs=source_tensor,
                        loss_fn=loss_fn,
                        loss_fn_kwargs={'target': answer_tensor},
                        is_log_progress=False,
                        is_return_results_every_t=False,
                    )
                else:
                    self.optimizer.zero_grad()
                    output = self.model(source_tensor)
                    output_softmax = torch.softmax(output * 10.0, dim=1)
                    loss = torch.mean((output_softmax - answer_tensor)**2)
                    loss.backward()
                    self.optimizer.step()

                # Determine which literal to use for position update
                if position_update == 'predicted':
                    next_literal = predicted_literal
                else:  # 'actual'
                    next_literal = correct_next_literal

                current_literal = next_literal

                if current_literal == self.goal_node:
                    break

                if current_literal not in visited_literals:
                    visited_literals.append(current_literal)

                step_count += 1

            if step_count >= max_steps:
                print(f"Warning: Epoch {epoch} reached max_steps ({max_steps}) without reaching goal")

            if epoch % capture_interval == 0:
                traj_sources, traj_answers = self._get_trajectory_subset(visited_literals)
                accuracy = self._test_trajectory_accuracy(traj_sources, traj_answers)
                accuracies.append(accuracy)
                learning_matrices = self._capture_learning_matrices(learning_matrices)
                print(f"Epoch {epoch}: Trajectory accuracy = {accuracy*100:.2f}% ({len(visited_literals)} literals)")

        return accuracies, learning_matrices

    def _decision_entropy(self, action_vector):
        """Calculate the entropy of the action vector.

        Args:
            action_vector: np.array or torch.Tensor - the variable from which entropy will be calculated.
        Returns:
            entropy: float - the entropy of the action_vector
        """
        # Convert to numpy if it's a tensor
        if isinstance(action_vector, torch.Tensor):
            action_np = action_vector.cpu().detach().numpy().flatten()
        else:
            action_np = np.array(action_vector).flatten()

        non_zero_mask = action_np > 0
        entropy = -np.sum(action_np[non_zero_mask] * np.log2(action_np[non_zero_mask]))
        return entropy

    def traverse_path(self, source, tau, force_action=True, max_steps=100):
        """Traverse an implication chain from source to goal, consulting oracle when uncertain."""
        if source not in self.literals:
            raise ValueError(f"source {source} not in valid literals")

        current_literal = source
        path = [source]
        oracle_calls = []
        accuracy = []
        step_count = 0

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            while step_count < max_steps:
                current_idx = self.literal_to_idx(current_literal)
                source_encoding = np.zeros(self.num_literals)
                source_encoding[current_idx] = 1
                source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)

                # Get policy network prediction
                prediction = self.model(source_tensor)
                # Apply softmax (for both PC and backprop)
                prediction = torch.softmax(prediction * 10.0, dim=1)

                predicted_literal_idx = torch.argmax(prediction).item()
                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                correct_next_literal = self.next_steps.get(current_literal)

                if correct_next_literal is None:
                    break

                entropy = self._decision_entropy(prediction)

                if entropy > tau:
                    oracle_calls.append(True)
                    accuracy.append(True)
                    next_literal = correct_next_literal
                else:
                    oracle_calls.append(False)
                    is_correct = (predicted_literal == correct_next_literal)
                    accuracy.append(is_correct)
                    if force_action:
                        next_literal = correct_next_literal
                    else:
                        next_literal = predicted_literal

                current_literal = next_literal
                path.append(current_literal)

                if current_literal == self.goal_node:
                    break

                step_count += 1

        # Restore previous training state
        if was_training:
            self.model.train()

        if step_count >= max_steps:
            print(f"Warning: traverse_path reached max_steps ({max_steps}) without reaching goal")

        return path, oracle_calls, accuracy

    def predict(self, source):
        """Predict the next literal in the implication chain toward fixed goal."""
        if isinstance(source, int):
            source_array = np.zeros(self.num_literals)
            source_idx = self.literal_to_idx(source)
            source_array[source_idx] = 1
        else:
            source_array = np.array(source)

        # Convert to tensor efficiently
        source_tensor = torch.from_numpy(np.array([source_array], dtype=np.float32)).to(self.device)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(source_tensor)
            # Apply softmax to convert to probabilities (for both PC and backprop)
            prediction = torch.softmax(prediction * 10.0, dim=1)

        # Restore previous training state
        if was_training:
            self.model.train()

        return prediction.cpu().numpy()[0]

    def update_single(self, source_encoding, policy_target):
        """Perform a single learning update for one (state, target) tuple."""
        source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([policy_target], dtype=torch.float32).to(self.device)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                output_softmax = torch.softmax(output * 10.0, dim=1)
                return torch.mean((output_softmax - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=source_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            output = self.model(source_tensor)
            output_softmax = torch.softmax(output * 10.0, dim=1)
            loss = torch.mean((output_softmax - target_tensor)**2)
            loss.backward()
            self.optimizer.step()

    def update_batch(self, source_encodings, policy_targets):
        """Perform a batch learning update for multiple (state, target) tuples."""
        source_tensor = torch.tensor(source_encodings, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                output_softmax = torch.softmax(output * 10.0, dim=1)
                return torch.mean((output_softmax - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=source_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            output = self.model(source_tensor)
            output_softmax = torch.softmax(output * 10.0, dim=1)
            loss = torch.mean((output_softmax - target_tensor)**2)
            loss.backward()
            self.optimizer.step()
