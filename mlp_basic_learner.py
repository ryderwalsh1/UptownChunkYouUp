import numpy as np
import warnings
import mlpcomposition as mlp
import gen_impgraph
import csv
import os
import matplotlib.pyplot as plt

# Implication graph parameters
num_vars = 8
num_clauses = 10

# Generate the implication graph
graph = gen_impgraph.generate_implication_graph(num_vars, num_clauses)

# MLP hyperparameters
hidden_size = 25
learning_rate = 0.60

# Create experiment name and directories
experiment_name = f'impgraph_{num_vars}v_{num_clauses}c'
os.makedirs(f'results/graphs/{experiment_name}', exist_ok=True)

# Save the graph structure for reproducibility
import networkx as nx
import pickle

# Save graph as pickle file
with open(f'results/graphs/{experiment_name}/graph.pkl', 'wb') as f:
    pickle.dump(graph, f)

# Save graph as edge list (human-readable)
nx.write_edgelist(graph, f'results/graphs/{experiment_name}/graph_edgelist.txt')

# Save graph metadata
with open(f'results/graphs/{experiment_name}/metadata.txt', 'w') as f:
    f.write(f"Num Variables: {num_vars}\n")
    f.write(f"Num Clauses: {num_clauses}\n")
    f.write(f"Num Nodes: {graph.number_of_nodes()}\n")
    f.write(f"Num Edges: {graph.number_of_edges()}\n")
    f.write(f"Hidden Size: {hidden_size}\n")
    f.write(f"Learning Rate: {learning_rate}\n")

# Visualize and save the graph
gen_impgraph.visualize_graph(graph)
plt.savefig(f'results/graphs/{experiment_name}/graph_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Initialize the MLP composition
composition = mlp.MLPComposition(
    num_vars=num_vars,
    graph=graph,
    hidden_size=hidden_size,
    learning_rate=learning_rate
)

mlp_comp = composition.mlp

composition.show_graph()


print(f"\nTraining data prepared:")
print(f"  Sources: {composition.training_sources.shape} (separate one-hot vectors)")
print(f"  Targets: {composition.training_targets.shape} (separate one-hot vectors)")
print(f"  Expected answers: {composition.expected_answers.shape}")
print(f"  Total examples: {len(composition.training_sources)}")

capture_interval = 5

losses, learning_matrices = composition.learn(epochs=500,capture_interval=capture_interval,on_policy=False)


# Save losses to csv
# Create results directory if it doesn't exist
os.makedirs('results/off_policy', exist_ok=True)

with open(f'results/off_policy/mlp_{experiment_name}_loss.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'loss'])
    for i, loss in enumerate(losses):
        writer.writerow([i * capture_interval, loss])  # epoch = index * capture_interval

# save trained matrices to npy files
for name, matrices in learning_matrices.items():
    os.makedirs(f'results/off_policy/learned_matrices/{experiment_name}/{name}', exist_ok=True)
    for i, matrix in enumerate(matrices):
        np.save(f'results/off_policy/learned_matrices/{experiment_name}/{name}/epoch_{capture_interval*i}.npy', matrix)

print("\nTraining complete.")
