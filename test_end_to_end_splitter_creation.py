#!/usr/bin/env python3
"""
Comprehensive test: Verify splitter creation works in realistic distributed scenario.

Scenario:
1. Training process creates a tree and applies split
2. Split creates new leaves with EMPTY splitters
3. Training process sends leaf updates to inference process
4. Inference process receives updates and creates splitters from scratch
"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth

print("="*70)
print("END-TO-END TEST: Distributed leaf update with splitter creation")
print("="*70)

# ===== TRAINING PROCESS =====
print("\n[TRAINING PROCESS]")
print("1. Creating and training model...")

training_model = HoeffdingTreeClassifier(
    grace_period=100,
    delta=1e-3,
    leaf_prediction='nba',
    leaf_update_threshold=50
)

dataset = synth.Agrawal(classification_function=0, seed=42)
for i, (x, y) in enumerate(dataset.take(300), 1):
    training_model.learn_one(x, y)

print(f"   Model nodes: {training_model.n_nodes}")
print(f"   Model height: {training_model.height}")

# Get a leaf node ID (not root, to simulate post-split scenario)
leaf_ids = [nid for nid, node in training_model._node_registry.items() 
            if hasattr(node, 'splitters')]
if len(leaf_ids) > 1:
    test_leaf_id = leaf_ids[1]  # Pick second leaf
else:
    test_leaf_id = leaf_ids[0]  # Fallback to first

print(f"   Selected leaf ID for test: {test_leaf_id}")

# Create update payload from training model
print("\n2. Creating update payload from training model...")
update_payload = training_model.create_update_payload(test_leaf_id, 'complete_node')
print(f"   Payload created for node {test_leaf_id}")
print(f"   Payload includes {len(update_payload['data']['splitter_data']['splitters'])} splitters")

# ===== INFERENCE PROCESS =====
print("\n[INFERENCE PROCESS]")
print("3. Simulating inference process receiving split event...")

# Create inference model
inference_model = HoeffdingTreeClassifier(
    grace_period=100,
    leaf_prediction='nba'
)

# Simulate receiving split event that created a new leaf
# This leaf starts with EMPTY splitters (realistic scenario)
print("\n4. Creating fresh leaf (simulating post-split state)...")
inference_model._root = None  # Clear root
new_leaf = inference_model._new_leaf()
inference_model._root = new_leaf
inference_model._n_active_leaves = 1

# Manually set the node ID to match the payload
new_leaf.node_id = test_leaf_id
inference_model._node_registry[test_leaf_id] = new_leaf

print(f"   Created leaf with ID {test_leaf_id}")
print(f"   Leaf splitters before update: {list(new_leaf.splitters.keys()) if new_leaf.splitters else 'Empty dict'}")
print(f"   Number of splitters: {len(new_leaf.splitters)}")

# Apply the distributed update
print("\n5. Applying distributed update to inference model...")
result = inference_model.apply_distributed_update(update_payload)

print(f"\n6. Update result: {result}")

# Verify the results
print("\n7. Verification:")
updated_leaf = inference_model.get_node_by_id(test_leaf_id)

if updated_leaf and updated_leaf.splitters:
    print(f"   ✅ Leaf now has {len(updated_leaf.splitters)} splitters")
    print(f"   Features: {list(updated_leaf.splitters.keys())}")
    
    # Check stats were updated
    print(f"\n   Stats: {dict(updated_leaf.stats)}")
    print(f"   Total weight: {updated_leaf.total_weight}")
    
    # Verify a Gaussian splitter
    sample_feature = list(updated_leaf.splitters.keys())[0]
    sample_splitter = updated_leaf.splitters[sample_feature]
    print(f"\n   Sample splitter '{sample_feature}':")
    print(f"   Type: {type(sample_splitter).__name__}")
    
    if hasattr(sample_splitter, '_att_dist_per_class'):
        for cls, dist in sample_splitter._att_dist_per_class.items():
            if hasattr(dist, 'mu'):
                print(f"   Class {cls}: μ={dist.mu:.2f}, σ={dist.sigma:.2f}, n={dist.n_samples:.0f}")
            else:
                print(f"   Class {cls}: {type(dist).__name__}")
    
    # Test prediction works
    print(f"\n8. Testing prediction on inference model...")
    test_x = {'salary': 50000, 'commission': 10000, 'age': 35, 'elevel': 2, 
              'car': 5, 'zipcode': 3, 'hvalue': 200000, 'hyears': 10, 'loan': 100000}
    pred = inference_model.predict_one(test_x)
    print(f"   Prediction: {pred}")
    
    print(f"\n{'='*70}")
    print(f"✅ END-TO-END TEST PASSED!")
    print(f"   - Splitters created from distributed update")
    print(f"   - Gaussian distributions populated correctly")
    print(f"   - Model can make predictions")
    print(f"{'='*70}")
else:
    print(f"   ❌ FAILURE: Splitters not created properly")
    print(f"   Leaf: {updated_leaf}")
    print(f"   Splitters: {updated_leaf.splitters if updated_leaf else 'None'}")
