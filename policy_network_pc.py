import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gen_impgraph
from lambda_labels import lambda_values
import sys
import os

# Add the Prospective-Configuration directory to the path to import predictive coding modules
pc_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Prospective-Configuration', 'predictive_coding')
if pc_path not in sys.path:
    sys.path.insert(0, pc_path)

from predictive_coding import pc_layer as pc
from predictive_coding import pc_trainer

class PolicyNetworkPC():
    def __init__(self, num_vars, num_clauses=None, graph=None, policy_name='Policy',
                 hidden_size=20, learning_rate=0.5, T=256, use_predictive_coding=False, **kwargs):
        """
        Policy network using Prospective Configuration via Predictive Coding.

        Args:
            num_vars: Number of variables in the implication graph
            num_clauses: Number of clauses (if generating new graph)
            graph: Pre-existing implication graph
            policy_name: Name for the policy network
            hidden_size: Number of hidden units
            learning_rate: Learning rate for parameter updates
            T: Number of inference iterations for PC (typically 256 for PC, 1 for backprop)
            use_predictive_coding: If True, use PC algorithm; if False, use standard backprop
            **kwargs: Additional arguments (e.g., source_to_hidden_matrix, target_to_hidden_matrix, hidden_to_output_matrix)
        """
        # Generate structure from implication graph
        if graph is None:
            if num_clauses is None:
                raise ValueError("Must provide either graph or num_clauses")
            self.graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)
        else:
            self.graph = graph

        self.num_vars = num_vars
        self.policy_name = policy_name

        # Compute next steps for all source-target pairs
        self.next_steps = gen_impgraph.compute_next_steps(self.graph, num_vars)

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
        target_to_hidden_matrix = kwargs.get("target_to_hidden_matrix", None)
        hidden_to_output_matrix = kwargs.get("hidden_to_output_matrix", None)
        hidden_to_value_matrix = kwargs.get("hidden_to_value_matrix", None)

        # Build the network using PyTorch
        self._build_network(source_to_hidden_matrix, target_to_hidden_matrix, hidden_to_output_matrix, hidden_to_value_matrix)

        # Prepare training data
        self._prepare_training_data()

    def _build_network(self, source_to_hidden_matrix=None, target_to_hidden_matrix=None, hidden_to_output_matrix=None, hidden_to_value_matrix=None):
        """Build the policy network with predictive coding layers and dual input pathways."""

        # Initialize weights for source pathway
        if source_to_hidden_matrix is not None:
            w_source = torch.tensor(source_to_hidden_matrix, dtype=torch.float32)
        else:
            w_source = torch.tensor(0.2 * np.random.rand(self.num_literals, self.hidden_size) - 0.1, dtype=torch.float32)

        # Initialize weights for target pathway
        if target_to_hidden_matrix is not None:
            w_target = torch.tensor(target_to_hidden_matrix, dtype=torch.float32)
        else:
            w_target = torch.tensor(0.2 * np.random.rand(self.num_literals, self.hidden_size) - 0.1, dtype=torch.float32)

        # Initialize weights for hidden to output
        if hidden_to_output_matrix is not None:
            w_out = torch.tensor(hidden_to_output_matrix, dtype=torch.float32)
        else:
            w_out = torch.tensor(0.2 * np.random.rand(self.hidden_size, self.num_literals) - 0.1, dtype=torch.float32)

        # Build network architecture with dual inputs
        if self.use_predictive_coding:
            # PC network with dual input pathways
            # Source pathway
            self.source_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.source_linear.weight.data = w_source.T

            # Target pathway
            self.target_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.target_linear.weight.data = w_target.T

            # Shared hidden layer components
            self.pc_layer1 = pc.PCLayer()
            self.activation = nn.Sigmoid()

            # Policy output pathway
            self.output_linear = nn.Linear(self.hidden_size, self.num_literals, bias=False)
            self.output_linear.weight.data = w_out.T

            self.pc_layer2 = pc.PCLayer()

            # Value output pathway (shares hidden layer)
            if hidden_to_value_matrix is not None:
                w_value = torch.tensor(hidden_to_value_matrix, dtype=torch.float32)
            else:
                w_value = torch.tensor(0.02 * np.random.rand(self.hidden_size, 1) - 0.01, dtype=torch.float32)

            self.value_linear = nn.Linear(self.hidden_size, 1, bias=False)
            self.value_linear.weight.data = w_value.T

            self.pc_layer_value = pc.PCLayer()

            # Create a custom forward function that handles dual inputs and dual outputs
            class DualInputPCModel(nn.Module):
                def __init__(self, source_linear, target_linear, pc_layer1, activation,
                             output_linear, pc_layer2, value_linear, pc_layer_value):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.pc_layer1 = pc_layer1
                    self.activation = activation
                    self.output_linear = output_linear
                    self.pc_layer2 = pc_layer2
                    self.value_linear = value_linear
                    self.pc_layer_value = pc_layer_value

                def forward(self, x, return_value=False):
                    # x is expected to be concatenated [source, target]
                    # Split the input
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:]

                    # Process both pathways and combine
                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    combined_hidden = source_hidden + target_hidden

                    # Shared hidden representation
                    hidden = self.pc_layer1(combined_hidden)
                    hidden = self.activation(hidden)

                    if return_value:
                        # Return value output
                        value = self.value_linear(hidden)
                        value = self.pc_layer_value(value)
                        return value
                    else:
                        # Return policy output (default)
                        output = self.output_linear(hidden)
                        output = self.pc_layer2(output)
                        return output

            self.model = DualInputPCModel(self.source_linear, self.target_linear, self.pc_layer1,
                                          self.activation, self.output_linear, self.pc_layer2,
                                          self.value_linear, self.pc_layer_value)

            # Create PC trainer
            self.pc_trainer = pc_trainer.PCTrainer(
                model=self.model,
                optimizer_x_fn=optim.SGD,
                optimizer_x_kwargs={'lr': 0.1},
                optimizer_p_fn=optim.SGD,
                optimizer_p_kwargs={'lr': self.learning_rate},
                T=self.T,
                update_x_at='all',
                update_p_at='last',
                plot_progress_at=[],
            )
        else:
            # Standard backprop network with dual inputs
            self.source_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.source_linear.weight.data = w_source.T

            self.target_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.target_linear.weight.data = w_target.T

            self.activation = nn.Sigmoid()

            self.output_linear = nn.Linear(self.hidden_size, self.num_literals, bias=False)
            self.output_linear.weight.data = w_out.T

            # Value output pathway (shares hidden layer)
            if hidden_to_value_matrix is not None:
                w_value = torch.tensor(hidden_to_value_matrix, dtype=torch.float32)
            else:
                w_value = torch.tensor(0.02 * np.random.rand(self.hidden_size, 1) - 0.01, dtype=torch.float32)

            self.value_linear = nn.Linear(self.hidden_size, 1, bias=False)
            self.value_linear.weight.data = w_value.T

            class DualInputModel(nn.Module):
                def __init__(self, source_linear, target_linear, activation, output_linear, value_linear):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.activation = activation
                    self.output_linear = output_linear
                    self.value_linear = value_linear

                def forward(self, x, return_value=False):
                    # x is expected to be concatenated [source, target]
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:]

                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    combined_hidden = source_hidden + target_hidden

                    hidden = self.activation(combined_hidden)

                    if return_value:
                        # Return value output
                        value = self.value_linear(hidden)
                        return value
                    else:
                        # Return policy output (default)
                        output = self.output_linear(hidden)
                        return output

            self.model = DualInputModel(self.source_linear, self.target_linear, self.activation,
                                        self.output_linear, self.value_linear)

            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.model.to(self.device)

    def _prepare_training_data(self):
        """Prepare training data for all source-target pairs."""
        training_sources = []
        training_targets = []
        expected_answers = []

        for memory in self.memories:
            training_sources.append(memory['Source'])
            training_targets.append(memory['Target'])
            expected_answers.append(memory['Answer'])

        self.training_sources = torch.tensor(np.array(training_sources), dtype=torch.float32).to(self.device)
        self.training_targets = torch.tensor(np.array(training_targets), dtype=torch.float32).to(self.device)
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
                # Concatenate source and target inputs
                source_target = torch.cat([self.training_sources[i:i+1], self.training_targets[i:i+1]], dim=1)
                output = self.model(source_target)
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
                # Concatenate source and target inputs
                source_target = torch.cat([self.training_sources[i:i+1], self.training_targets[i:i+1]], dim=1)
                output = self.model(source_target)
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
        # Extract weights from the Linear layers (same for both PC and backprop)
        source_to_hidden_matrix = self.source_linear.weight.data.T.cpu().numpy()
        target_to_hidden_matrix = self.target_linear.weight.data.T.cpu().numpy()
        hidden_to_output_matrix = self.output_linear.weight.data.T.cpu().numpy()

        learning_matrices['source_to_hidden'].append(source_to_hidden_matrix.copy())
        learning_matrices['target_to_hidden'].append(target_to_hidden_matrix.copy())
        learning_matrices['hidden_to_output'].append(hidden_to_output_matrix.copy())

        return learning_matrices

    def _idx_to_literal(self, idx):
        """Convert encoding index to literal value."""
        if idx < self.num_vars:
            return idx + 1
        else:
            return -(idx - self.num_vars + 1)

    def _get_next_literal(self, current_literal, target_literal):
        """Get the next literal to traverse in the implication chain toward target_literal."""
        return self.next_steps.get((current_literal, target_literal))

    def _get_trajectory_subset(self, visited_literals, target_literal):
        """Create training data subset for visited literals and target."""
        trajectory_sources = []
        trajectory_targets = []
        trajectory_answers = []

        # Get target index
        target_idx = self.literal_to_idx(target_literal)

        for source_lit in visited_literals:
            source_idx = self.literal_to_idx(source_lit)
            source_encoding = np.zeros(self.num_literals)
            source_encoding[source_idx] = 1

            target_encoding = np.zeros(self.num_literals)
            target_encoding[target_idx] = 1

            # Get expected next step
            next_step = self.next_steps.get((source_lit, target_literal))
            if next_step is None:
                continue

            answer_encoding = np.zeros(self.num_literals)
            answer_idx = self.literal_to_idx(next_step)
            answer_encoding[answer_idx] = 1

            trajectory_sources.append(source_encoding)
            trajectory_targets.append(target_encoding)
            trajectory_answers.append(answer_encoding)

        return (np.array(trajectory_sources), np.array(trajectory_targets), np.array(trajectory_answers))

    def _test_trajectory_accuracy(self, trajectory_sources, trajectory_targets, trajectory_answers):
        """Test accuracy on a specific trajectory subset."""
        correct = 0
        num_samples = len(trajectory_sources)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                source = torch.tensor(trajectory_sources[i:i+1], dtype=torch.float32).to(self.device)
                target = torch.tensor(trajectory_targets[i:i+1], dtype=torch.float32).to(self.device)
                # Concatenate source and target inputs
                source_target = torch.cat([source, target], dim=1)
                output = self.model(source_target)
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
              target_literal=None, position_update='actual', max_steps=100):
        """Wrapper function for the two learning modes."""
        if not on_policy:
            ret = self._learn_off_policy(epochs=epochs, capture_interval=capture_interval)
        else:
            if source_literal is None or target_literal is None:
                raise ValueError("Source and target literals must be provided for on-policy learning")
            ret = self._learn_on_policy(source_literal, target_literal, epochs=epochs, capture_interval=capture_interval,
                                  position_update=position_update, max_steps=max_steps)
        return ret

    def _learn_off_policy(self, epochs=150, capture_interval=10):
        """Off-policy learning using the full training set."""
        losses = []
        learning_matrices = {
            'source_to_hidden': [],
            'target_to_hidden': [],
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

            # Concatenate source and target for input
            combined_inputs = torch.cat([self.training_sources, self.training_targets], dim=1)

            if self.use_predictive_coding:
                # Use PC training
                def loss_fn(output, target):
                    # Apply softmax to output for loss calculation
                    output_softmax = torch.softmax(output * 10.0, dim=1)
                    return torch.mean((output_softmax - target)**2)

                self.pc_trainer.train_on_batch(
                    inputs=combined_inputs,
                    loss_fn=loss_fn,
                    loss_fn_kwargs={'target': self.expected_answers},
                    is_log_progress=False,
                    is_return_results_every_t=False,
                )
            else:
                # Use standard backprop
                self.optimizer.zero_grad()
                output = self.model(combined_inputs)
                output_softmax = torch.softmax(output * 10.0, dim=1)
                loss = torch.mean((output_softmax - self.expected_answers)**2)
                loss.backward()
                self.optimizer.step()

        return losses, learning_matrices

    def _learn_on_policy(self, source_literal, target_literal, epochs=150, capture_interval=10,
                        position_update='actual', max_steps=100):
        """On-policy learning by traversing from source to target repeatedly."""
        if source_literal not in self.literals:
            raise ValueError(f"source_literal {source_literal} not in valid literals")
        if target_literal not in self.literals:
            raise ValueError(f"target_literal {target_literal} not in valid literals")
        if position_update not in ['predicted', 'actual']:
            raise ValueError("position_update must be 'predicted' or 'actual'")

        source_idx = self.literal_to_idx(source_literal)
        target_idx = self.literal_to_idx(target_literal)
        accuracies = []
        learning_matrices = {
            'source_to_hidden': [],
            'target_to_hidden': [],
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

                target_encoding = np.zeros(self.num_literals)
                target_encoding[target_idx] = 1

                source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
                target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

                # Get policy network prediction (temporarily switch to eval mode)
                was_training = self.model.training
                self.model.eval()
                with torch.no_grad():
                    prediction = self.model(combined_tensor)
                    # Apply softmax (for both PC and backprop)
                    prediction = torch.softmax(prediction * 10.0, dim=1)
                    predicted_literal_idx = torch.argmax(prediction).item()
                if was_training:
                    self.model.train()

                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                # Get correct next step toward target
                correct_next_literal = self.next_steps.get((current_literal, target_literal))

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
                        inputs=combined_tensor,
                        loss_fn=loss_fn,
                        loss_fn_kwargs={'target': answer_tensor},
                        is_log_progress=False,
                        is_return_results_every_t=False,
                    )
                else:
                    self.optimizer.zero_grad()
                    output = self.model(combined_tensor)
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

                if current_literal == target_literal:
                    break

                if current_literal not in visited_literals:
                    visited_literals.append(current_literal)

                step_count += 1

            if step_count >= max_steps:
                print(f"Warning: Epoch {epoch} reached max_steps ({max_steps}) without reaching target")

            if epoch % capture_interval == 0:
                traj_sources, traj_targets, traj_answers = self._get_trajectory_subset(visited_literals, target_literal)
                accuracy = self._test_trajectory_accuracy(traj_sources, traj_targets, traj_answers)
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

    def traverse_path(self, source, target, tau, force_action=True, max_steps=100):
        """Traverse an implication chain from source to target, consulting oracle when uncertain."""
        if source not in self.literals:
            raise ValueError(f"source {source} not in valid literals")
        if target not in self.literals:
            raise ValueError(f"target {target} not in valid literals")

        current_literal = source
        path = [source]
        oracle_calls = []
        accuracy = []
        step_count = 0

        target_idx = self.literal_to_idx(target)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            while step_count < max_steps:
                current_idx = self.literal_to_idx(current_literal)
                source_encoding = np.zeros(self.num_literals)
                source_encoding[current_idx] = 1

                target_encoding = np.zeros(self.num_literals)
                target_encoding[target_idx] = 1

                source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
                target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

                # Get policy network prediction
                prediction = self.model(combined_tensor)
                # Apply softmax (for both PC and backprop)
                prediction = torch.softmax(prediction * 10.0, dim=1)

                predicted_literal_idx = torch.argmax(prediction).item()
                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                correct_next_literal = self.next_steps.get((current_literal, target))

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

                if current_literal == target:
                    break

                step_count += 1

        # Restore previous training state
        if was_training:
            self.model.train()

        if step_count >= max_steps:
            print(f"Warning: traverse_path reached max_steps ({max_steps}) without reaching target")

        return path, oracle_calls, accuracy

    def predict(self, source, target):
        """Predict the next literal in the implication chain from source to target."""
        if isinstance(source, int) and isinstance(target, int):
            source_array = np.zeros(self.num_literals)
            source_idx = self.literal_to_idx(source)
            source_array[source_idx] = 1

            target_array = np.zeros(self.num_literals)
            target_idx = self.literal_to_idx(target)
            target_array[target_idx] = 1
        else:
            source_array = np.array(source)
            target_array = np.array(target)

        # Convert to tensor efficiently and concatenate
        source_tensor = torch.from_numpy(np.array([source_array], dtype=np.float32)).to(self.device)
        target_tensor = torch.from_numpy(np.array([target_array], dtype=np.float32)).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(combined_tensor)
            # Apply softmax to convert to probabilities (for both PC and backprop)
            prediction = torch.softmax(prediction * 10.0, dim=1)

        # Restore previous training state
        if was_training:
            self.model.train()

        return prediction.cpu().numpy()[0]

    def update_single(self, source_encoding, target_encoding, policy_target):
        """Perform a single learning update for one (state, goal, target) tuple."""
        source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)
        policy_target_tensor = torch.tensor([policy_target], dtype=torch.float32).to(self.device)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                output_softmax = torch.softmax(output * 10.0, dim=1)
                return torch.mean((output_softmax - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=combined_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': policy_target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            output = self.model(combined_tensor)
            output_softmax = torch.softmax(output * 10.0, dim=1)
            loss = torch.mean((output_softmax - policy_target_tensor)**2)
            loss.backward()
            self.optimizer.step()

    def update_batch(self, source_encodings, target_encodings, policy_targets):
        """Perform a batch learning update for multiple (state, goal, target) tuples."""
        source_tensor = torch.tensor(source_encodings, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_encodings, dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)
        policy_target_tensor = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                output_softmax = torch.softmax(output * 10.0, dim=1)
                return torch.mean((output_softmax - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=combined_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': policy_target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            output = self.model(combined_tensor)
            output_softmax = torch.softmax(output * 10.0, dim=1)
            loss = torch.mean((output_softmax - policy_target_tensor)**2)
            loss.backward()
            self.optimizer.step()

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

        # Convert to tensors
        source_tensor = torch.from_numpy(np.array([source_encoding], dtype=np.float32)).to(self.device)
        target_tensor = torch.from_numpy(np.array([target_encoding], dtype=np.float32)).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            value = self.model(combined_tensor, return_value=True)

        # Restore previous training state
        if was_training:
            self.model.train()

        return float(value[0, 0].cpu().item())

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

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            for i, literal in enumerate(path):
                source_idx = self.literal_to_idx(literal)
                source_encoding = np.zeros(self.num_literals)
                source_encoding[source_idx] = 1.0

                # Convert to tensors
                source_tensor = torch.from_numpy(np.array([source_encoding], dtype=np.float32)).to(self.device)
                target_tensor = torch.from_numpy(np.array([target_encoding], dtype=np.float32)).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

                # Get value
                value = self.model(combined_tensor, return_value=True)
                values[i] = float(value[0, 0].cpu().item())

        # Restore previous training state
        if was_training:
            self.model.train()

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

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            # Compute entropy at each step (excluding the final state)
            for i in range(len(path) - 1):
                source_idx = self.literal_to_idx(path[i])
                source_encoding = np.zeros(self.num_literals)
                source_encoding[source_idx] = 1.0

                # Convert to tensors
                source_tensor = torch.from_numpy(np.array([source_encoding], dtype=np.float32)).to(self.device)
                target_tensor = torch.from_numpy(np.array([target_encoding], dtype=np.float32)).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

                # Get policy prediction
                prediction = self.model(combined_tensor, return_value=False)
                prediction = torch.softmax(prediction * 10.0, dim=1)

                # Compute entropy
                entropy = self._decision_entropy(prediction)
                entropies.append(entropy)

        # Restore previous training state
        if was_training:
            self.model.train()

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
        source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)

        # Ensure value_target is a tensor
        if isinstance(value_target, (int, float)):
            value_target_tensor = torch.tensor([[value_target]], dtype=torch.float32).to(self.device)
        else:
            value_target_tensor = torch.tensor([value_target], dtype=torch.float32).to(self.device)
            if len(value_target_tensor.shape) == 1:
                value_target_tensor = value_target_tensor.unsqueeze(1)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                return torch.mean((output - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=combined_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': value_target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            value = self.model(combined_tensor, return_value=True)
            loss = torch.mean((value - value_target_tensor)**2)
            loss.backward()
            self.optimizer.step()

    def update_value_batch(self, source_encodings, target_encodings, value_targets):
        """
        Perform a batch value head update for multiple (state, goal, value_target) tuples.

        Args:
            source_encodings: np.array - shape (batch_size, num_literals)
            target_encodings: np.array - shape (batch_size, num_literals)
            value_targets: np.array - shape (batch_size,) or (batch_size, 1)
        """
        source_tensor = torch.tensor(source_encodings, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_encodings, dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor], dim=1)
        value_target_tensor = torch.tensor(value_targets, dtype=torch.float32).to(self.device)

        # Ensure value_target_tensor is 2D (batch_size, 1)
        if len(value_target_tensor.shape) == 1:
            value_target_tensor = value_target_tensor.unsqueeze(1)

        # Ensure model is in training mode
        self.model.train()

        if self.use_predictive_coding:
            def loss_fn(output, target):
                return torch.mean((output - target)**2)

            self.pc_trainer.train_on_batch(
                inputs=combined_tensor,
                loss_fn=loss_fn,
                loss_fn_kwargs={'target': value_target_tensor},
                is_log_progress=False,
                is_return_results_every_t=False,
            )
        else:
            self.optimizer.zero_grad()
            value = self.model(combined_tensor, return_value=True)
            loss = torch.mean((value - value_target_tensor)**2)
            loss.backward()
            self.optimizer.step()

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

        # 6. Train value head with TD(λ) targets (if enabled)
        if update_value:
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
