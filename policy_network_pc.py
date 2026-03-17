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

        # Context trace for eligibility-based sequential memory
        self.context_trace = np.zeros(self.num_literals)

        # Optional matrix specifications
        source_to_hidden_matrix = kwargs.get("source_to_hidden_matrix", None)
        target_to_hidden_matrix = kwargs.get("target_to_hidden_matrix", None)
        context_to_hidden_matrix = kwargs.get("context_to_hidden_matrix", None)
        hidden_to_output_matrix = kwargs.get("hidden_to_output_matrix", None)
        hidden_to_value_matrix = kwargs.get("hidden_to_value_matrix", None)

        # Build the network using PyTorch
        self._build_network(source_to_hidden_matrix, target_to_hidden_matrix, context_to_hidden_matrix,
                           hidden_to_output_matrix, hidden_to_value_matrix)

        # Prepare training data
        self._prepare_training_data()

    def _build_network(self, source_to_hidden_matrix=None, target_to_hidden_matrix=None, context_to_hidden_matrix=None,
                      hidden_to_output_matrix=None, hidden_to_value_matrix=None):
        """Build the policy network with predictive coding layers and triple input pathways."""

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

        # Initialize weights for context pathway
        if context_to_hidden_matrix is not None:
            w_context = torch.tensor(context_to_hidden_matrix, dtype=torch.float32)
        else:
            w_context = torch.tensor(0.2 * np.random.rand(self.num_literals, self.hidden_size) - 0.1, dtype=torch.float32)

        # Initialize weights for hidden to output
        if hidden_to_output_matrix is not None:
            w_out = torch.tensor(hidden_to_output_matrix, dtype=torch.float32)
        else:
            w_out = torch.tensor(0.2 * np.random.rand(self.hidden_size, self.num_literals) - 0.1, dtype=torch.float32)

        # Build network architecture with triple inputs
        if self.use_predictive_coding:
            # PC network with triple input pathways
            # Source pathway
            self.source_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.source_linear.weight.data = w_source.T

            # Target pathway
            self.target_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.target_linear.weight.data = w_target.T

            # Context pathway
            self.context_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.context_linear.weight.data = w_context.T

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

            # Create a custom forward function that handles triple inputs and dual outputs
            class TripleInputPCModel(nn.Module):
                def __init__(self, source_linear, target_linear, context_linear, pc_layer1, activation,
                             output_linear, pc_layer2, value_linear, pc_layer_value):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.context_linear = context_linear
                    self.pc_layer1 = pc_layer1
                    self.activation = activation
                    self.output_linear = output_linear
                    self.pc_layer2 = pc_layer2
                    self.value_linear = value_linear
                    self.pc_layer_value = pc_layer_value

                def forward(self, x, return_value=False):
                    # x is expected to be concatenated [source, target, context]
                    # Split the input
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:2*self.source_linear.in_features]
                    context_input = x[:, 2*self.source_linear.in_features:]

                    # Process all three pathways and combine
                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    context_hidden = self.context_linear(context_input)
                    combined_hidden = source_hidden + target_hidden + context_hidden

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

            self.model = TripleInputPCModel(self.source_linear, self.target_linear, self.context_linear,
                                           self.pc_layer1, self.activation, self.output_linear, self.pc_layer2,
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
            # Standard backprop network with triple inputs
            self.source_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.source_linear.weight.data = w_source.T

            self.target_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.target_linear.weight.data = w_target.T

            self.context_linear = nn.Linear(self.num_literals, self.hidden_size, bias=False)
            self.context_linear.weight.data = w_context.T

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

            class TripleInputModel(nn.Module):
                def __init__(self, source_linear, target_linear, context_linear, activation, output_linear, value_linear):
                    super().__init__()
                    self.source_linear = source_linear
                    self.target_linear = target_linear
                    self.context_linear = context_linear
                    self.activation = activation
                    self.output_linear = output_linear
                    self.value_linear = value_linear

                def forward(self, x, return_value=False):
                    # x is expected to be concatenated [source, target, context]
                    source_input = x[:, :self.source_linear.in_features]
                    target_input = x[:, self.source_linear.in_features:2*self.source_linear.in_features]
                    context_input = x[:, 2*self.source_linear.in_features:]

                    source_hidden = self.source_linear(source_input)
                    target_hidden = self.target_linear(target_input)
                    context_hidden = self.context_linear(context_input)
                    combined_hidden = source_hidden + target_hidden + context_hidden

                    hidden = self.activation(combined_hidden)

                    if return_value:
                        # Return value output
                        value = self.value_linear(hidden)
                        return value
                    else:
                        # Return policy output (default)
                        output = self.output_linear(hidden)
                        return output

            self.model = TripleInputModel(self.source_linear, self.target_linear, self.context_linear,
                                         self.activation, self.output_linear, self.value_linear)

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

        # Zero context for testing
        context_encoding = torch.zeros(1, self.num_literals, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(num_examples):
                # Concatenate source, target, and context inputs
                source_target_context = torch.cat([self.training_sources[i:i+1], self.training_targets[i:i+1], context_encoding], dim=1)
                output = self.model(source_target_context)
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

        # Zero context for testing
        context_encoding = torch.zeros(1, self.num_literals, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(num_examples):
                # Concatenate source, target, and context inputs
                source_target_context = torch.cat([self.training_sources[i:i+1], self.training_targets[i:i+1], context_encoding], dim=1)
                output = self.model(source_target_context)
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

        # Zero context for testing
        context_encoding = torch.zeros(1, self.num_literals, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                source = torch.tensor(trajectory_sources[i:i+1], dtype=torch.float32).to(self.device)
                target = torch.tensor(trajectory_targets[i:i+1], dtype=torch.float32).to(self.device)
                # Concatenate source, target, and context inputs
                source_target_context = torch.cat([source, target, context_encoding], dim=1)
                output = self.model(source_target_context)
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

    # ========================================================================
    # Context Trace Methods
    # ========================================================================

    def reset_context(self):
        """Reset the context trace to zeros (start of a new trajectory)."""
        self.context_trace = np.zeros(self.num_literals)

    def update_context(self, current_literal, decay):
        """
        Update the context trace with the current state using eligibility trace dynamics.

        This is mathematically identical to accumulating eligibility traces:
          E(s) ← decay * E(s)  for all s    (decay all)
          E(S) ← E(S) + 1                   (bump current)

        Args:
            current_literal: int - the state just visited
            decay: float - decay rate, should be gamma * lambda where lambda
                   is the chunkability-modulated trace parameter

        Returns:
            np.array - the updated context trace (also stored in self.context_trace)
        """
        self.context_trace = decay * self.context_trace
        current_idx = self.literal_to_idx(current_literal)
        self.context_trace[current_idx] += 1.0
        return self.context_trace.copy()

    def build_trajectory_contexts(self, path, gamma, lambda_):
        """
        Build context trace vectors for each step in a trajectory.

        Args:
            path: list[int] - full trajectory
            gamma: float - discount factor
            lambda_: float - trace decay parameter (chunkability-modulated)

        Returns:
            list[np.array] - context vector for each step in path
        """
        contexts = []
        trace = np.zeros(self.num_literals)
        decay = gamma * lambda_

        for i, literal in enumerate(path):
            # Store current context BEFORE updating with this state
            contexts.append(trace.copy())
            # Update trace with current state
            idx = self.literal_to_idx(literal)
            trace = decay * trace
            trace[idx] += 1.0

        return contexts

    def traverse_path(self, source, target, tau, force_action=True, max_steps=100, gamma=0.99, lambda_=0.0):
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

        # Reset context trace at start of trajectory
        self.reset_context()
        decay = gamma * lambda_

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

                # Get current context (before updating with current state)
                context_encoding = self.context_trace.copy()

                source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
                target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
                context_tensor = torch.tensor([context_encoding], dtype=torch.float32).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

                # Get policy network prediction
                prediction = self.model(combined_tensor)
                # Apply softmax (for both PC and backprop)
                prediction = torch.softmax(prediction * 10.0, dim=1)

                predicted_literal_idx = torch.argmax(prediction).item()
                predicted_literal = self._idx_to_literal(predicted_literal_idx)

                correct_next_literal = self.next_steps.get((current_literal, target))

                if correct_next_literal is None:
                    break

                # Compute entropy of the prediction vector
                prediction_np = prediction.cpu().detach().numpy().flatten()
                non_zero_mask = prediction_np > 0
                entropy = -np.sum(prediction_np[non_zero_mask] * np.log2(prediction_np[non_zero_mask]))

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

                # Update context trace with current state (for next step)
                self.update_context(current_literal, decay)

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

    def predict(self, source, target, context=None):
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

        # Use provided context or default to current context trace
        if context is None:
            context = self.context_trace.copy()

        # Convert to tensor efficiently and concatenate
        source_tensor = torch.from_numpy(np.array([source_array], dtype=np.float32)).to(self.device)
        target_tensor = torch.from_numpy(np.array([target_array], dtype=np.float32)).to(self.device)
        context_tensor = torch.from_numpy(np.array([context], dtype=np.float32)).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

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

    def update_single(self, source_encoding, target_encoding, policy_target, context_encoding=None):
        """Perform a single learning update for one (state, goal, target) tuple."""
        if context_encoding is None:
            context_encoding = np.zeros(self.num_literals)

        source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor([context_encoding], dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)
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

    def update_batch(self, source_encodings, target_encodings, policy_targets, context_encodings=None):
        """Perform a batch learning update for multiple (state, goal, target) tuples."""
        if context_encodings is None:
            context_encodings = np.zeros((len(source_encodings), self.num_literals))

        source_tensor = torch.tensor(source_encodings, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_encodings, dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor(context_encodings, dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)
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

    def compute_value(self, source_literal, target_literal, context=None):
        """
        Compute value estimate for a single (state, goal) pair.

        Args:
            source_literal: int - current state literal
            target_literal: int - goal literal
            context: np.array or None - context trace encoding (defaults to zeros)

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

        # Use provided context or default to zeros
        if context is None:
            context = np.zeros(self.num_literals)

        # Convert to tensors
        source_tensor = torch.from_numpy(np.array([source_encoding], dtype=np.float32)).to(self.device)
        target_tensor = torch.from_numpy(np.array([target_encoding], dtype=np.float32)).to(self.device)
        context_tensor = torch.from_numpy(np.array([context], dtype=np.float32)).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

        # Store current training state
        was_training = self.model.training

        self.model.eval()
        with torch.no_grad():
            value = self.model(combined_tensor, return_value=True)

        # Restore previous training state
        if was_training:
            self.model.train()

        return float(value[0, 0].cpu().item())

    def compute_trajectory_values(self, path, target_literal, gamma=0.99, lambda_=0.0):
        """
        Compute value estimates for all states in a trajectory.

        Args:
            path: list[int] - trajectory of literals
            target_literal: int - goal literal
            gamma: float - discount factor for context trace decay
            lambda_: float - trace decay parameter (default 0.0 for no context)

        Returns:
            np.array - value estimates for each state in path, shape (len(path),)
        """
        values = np.zeros(len(path))
        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        # Build context traces for the trajectory
        contexts = self.build_trajectory_contexts(path, gamma, lambda_)

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
                context_tensor = torch.from_numpy(np.array([contexts[i]], dtype=np.float32)).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

                # Get value
                value = self.model(combined_tensor, return_value=True)
                values[i] = float(value[0, 0].cpu().item())

        # Restore previous training state
        if was_training:
            self.model.train()

        return values

    def chunkability_from_trajectory(self, path, target_literal, gamma=0.99, lambda_=0.0):
        """
        Compute chunkability (corridor-likeness) of a trajectory.

        Uses the policy head's entropy to measure how deterministic the policy is.
        Chunkability = mean(1 / exp(H_t)) across trajectory steps.

        Args:
            path: list[int] - trajectory of literals
            target_literal: int - goal literal
            gamma: float - discount factor for context trace decay
            lambda_: float - trace decay parameter (default 0.0 for no context)

        Returns:
            float - chunkability metric (1.0 = corridor-like, 0.0 = diffuse)
        """
        if len(path) <= 1:
            return 0.0

        entropies = []
        target_idx = self.literal_to_idx(target_literal)
        target_encoding = np.zeros(self.num_literals)
        target_encoding[target_idx] = 1.0

        # Build context traces for the trajectory
        contexts = self.build_trajectory_contexts(path, gamma, lambda_)

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
                context_tensor = torch.from_numpy(np.array([contexts[i]], dtype=np.float32)).to(self.device)
                combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

                # Get policy prediction
                prediction = self.model(combined_tensor, return_value=False)
                prediction = torch.softmax(prediction * 10.0, dim=1)

                # Compute entropy
                prediction_np = prediction.cpu().detach().numpy().flatten()
                non_zero_mask = prediction_np > 0
                entropy = -np.sum(prediction_np[non_zero_mask] * np.log2(prediction_np[non_zero_mask]))
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

    def update_value_single(self, source_encoding, target_encoding, value_target, context_encoding=None):
        """
        Perform a single value head update for one (state, goal, value_target) tuple.

        Args:
            source_encoding: np.array - one-hot encoding of current state
            target_encoding: np.array - one-hot encoding of goal state
            value_target: float or np.array - scalar value target
            context_encoding: np.array or None - context trace encoding (defaults to zeros)
        """
        if context_encoding is None:
            context_encoding = np.zeros(self.num_literals)

        source_tensor = torch.tensor([source_encoding], dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor([target_encoding], dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor([context_encoding], dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)

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

    def update_value_batch(self, source_encodings, target_encodings, value_targets, context_encodings=None):
        """
        Perform a batch value head update for multiple (state, goal, value_target) tuples.

        Args:
            source_encodings: np.array - shape (batch_size, num_literals)
            target_encodings: np.array - shape (batch_size, num_literals)
            value_targets: np.array - shape (batch_size,) or (batch_size, 1)
            context_encodings: np.array or None - shape (batch_size, num_literals) of context traces (defaults to zeros)
        """
        if context_encodings is None:
            context_encodings = np.zeros((len(source_encodings), self.num_literals))

        source_tensor = torch.tensor(source_encodings, dtype=torch.float32).to(self.device)
        target_tensor = torch.tensor(target_encodings, dtype=torch.float32).to(self.device)
        context_tensor = torch.tensor(context_encodings, dtype=torch.float32).to(self.device)
        combined_tensor = torch.cat([source_tensor, target_tensor, context_tensor], dim=1)
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

        # 1. Compute chunkability (using zero context, since we don't know lambda yet)
        chunkability = self.chunkability_from_trajectory(path, target_literal, gamma=gamma, lambda_=0.0)

        # 2. Derive lambda from chunkability
        lambda_ = chunkability ** lambda_exponent

        # 3. Build context traces with the derived lambda
        contexts = self.build_trajectory_contexts(path, gamma, lambda_)

        # 4. Train policy head with teacher-forced one-hot targets and context traces
        trajectory_sources, trajectory_targets, trajectory_answers = \
            self._get_trajectory_subset(path, target_literal)

        if len(trajectory_sources) > 0:
            # Build context traces for the trajectory steps (excluding terminal state)
            trajectory_contexts = contexts[:len(trajectory_sources)]
            self.update_batch(trajectory_sources, trajectory_targets, trajectory_answers,
                            context_encodings=np.array(trajectory_contexts))

        # 5. Compute value estimates with context traces
        value_estimates = self.compute_trajectory_values(path, target_literal, gamma=gamma, lambda_=lambda_)

        # 6. Compute TD(λ) value targets
        value_targets = lambda_values(rewards, value_estimates, gamma, lambda_)

        # 7. Train value head with TD(λ) targets (if enabled)
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
                value_targets,
                context_encodings=np.array(contexts)
            )

        # 8. Return diagnostics
        diagnostics = {
            'chunkability': chunkability,
            'lambda_': lambda_,
            'value_targets': value_targets,
            'value_estimates': value_estimates
        }

        return diagnostics
