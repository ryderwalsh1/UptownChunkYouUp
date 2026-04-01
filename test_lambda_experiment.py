"""
Quick test script for lambda experiment components.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("Testing lambda experiment components...")

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from lambda_experiment import topology_generators, topology_metrics
    print("  ✓ Imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 2: Generate topologies
print("\n2. Testing topology generation...")
try:
    from lambda_experiment.topology_generators import ALL_TOPOLOGIES, generate_topology

    # Test one from each category
    test_configs = {
        'corridor_short': ALL_TOPOLOGIES['corridor_short'],
        'proc_mixed': ALL_TOPOLOGIES['proc_mixed'],
    }

    for name, config in test_configs.items():
        maze = generate_topology(config, seed=42)
        stats = maze.get_stats()
        print(f"  {name}: {stats['total_nodes']} nodes, {stats['total_edges']} edges")

    print("  ✓ Topology generation successful")
except Exception as e:
    print(f"  ✗ Topology generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Compute metrics
print("\n3. Testing topology metrics...")
try:
    from lambda_experiment.topology_metrics import compute_all_metrics

    maze = generate_topology(ALL_TOPOLOGIES['proc_mixed'], seed=42)
    graph = maze.get_graph()
    nodes = list(graph.nodes())

    metrics = compute_all_metrics(graph, start=nodes[0], goal=nodes[-1])
    print(f"  Computed {len(metrics)} metrics")
    print(f"  Sample: junction_density={metrics['junction_density']:.3f}")

    print("  ✓ Topology metrics successful")
except Exception as e:
    print(f"  ✗ Topology metrics failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Test environment creation
print("\n4. Testing environment creation...")
try:
    from maze_env import MazeEnvironment

    maze = generate_topology(ALL_TOPOLOGIES['corridor_short'], seed=42)
    env = MazeEnvironment(
        length=maze.length,
        width=maze.width,
        corridor=maze.corridor,
        seed=42
    )
    env.maze = maze
    env.graph = maze.get_graph()
    env.nodes_list = list(env.graph.nodes())
    env.node_to_idx = {node: idx for idx, node in enumerate(env.nodes_list)}
    env.num_nodes = len(env.nodes_list)

    state = env.reset()
    print(f"  Environment has {env.num_nodes} nodes")
    print(f"  ✓ Environment creation successful")
except Exception as e:
    print(f"  ✗ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test network creation
print("\n5. Testing network creation...")
try:
    import torch
    from fast import FastNetwork, FastNetworkTrainer

    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=5,
        embedding_dim=64,
        hidden_dim=128,
        prospection_head=False
    )

    trainer = FastNetworkTrainer(
        network=network,
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95
    )

    print(f"  Network has {sum(p.numel() for p in network.parameters())} parameters")
    print(f"  ✓ Network creation successful")
except Exception as e:
    print(f"  ✗ Network creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nYou can now run the full experiment with:")
print("  python lambda_experiment.py")
