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
    def __init__(self, num_vars, num_clauses=None, graph=None, policy_name='Policy',
                 hidden_size=20, learning_rate=0.5, T=256, use_predictive_coding=True, **kwargs):
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

        # Build the network using PyTorch
        self._build_network(source_to_hidden_matrix, target_to_hidden_matrix, hidden_to_output_matrix)

        # Prepare training data
        self._prepare_training_data()

    def _build_network(self, source_to_hidden_matrix=None, target_to_hidden_matrix=None, hidden_to_output_matrix=None):
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

            # Output pathway
            self.output_linear = nn.Linear(self.hidden_size, self.num_literals, bias=False)
            self.output_linear.weight.data = w_out.T

            self.pc_layer2 = pc.PCLayer()

            # Create a custom forward function that handles dual inputs
            class DualInputPCModel(nn.Module):
                def __init__(self, source_linear, target_linear, pc_layer1, activation, output_linear, pc_layer2):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.pc_layer1 = pc_layer1
                    self.activation = activation
                    self.output_linear = output_linear
                    self.pc_layer2 = pc_layer2

                def forward(self, x):
                    # x is expected to be concatenated [source, target]
                    # Split the input
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:]

                    # Process both pathways and combine
                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    combined_hidden = source_hidden + target_hidden

                    # Continue through the network
                    hidden = self.pc_layer1(combined_hidden)
                    hidden = self.activation(hidden)
                    output = self.output_linear(hidden)
                    output = self.pc_layer2(output)
                    return output

            self.model = DualInputPCModel(self.source_linear, self.target_linear, self.pc_layer1,
                                          self.activation, self.output_linear, self.pc_layer2)

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

            class DualInputModel(nn.Module):
                def __init__(self, source_linear, target_linear, activation, output_linear):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.activation = activation
                    self.output_linear = output_linear

                def forward(self, x):
                    # x is expected to be concatenated [source, target]
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:]

                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    combined_hidden = source_hidden + target_hidden

                    hidden = self.activation(combined_hidden)
                    output = self.output_linear(hidden)
                    return output

            self.model = DualInputModel(self.source_linear, self.target_linear, self.activation, self.output_linear)

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
