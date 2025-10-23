#!/usr/bin/env python3
"""
END-TO-END TEST: Distributed update with activated leaf

Scenario:
1. Train model enough to create split AND activate child leaves
2. Get payload from activated leaf (has splitters)
3. Send to inference process with empty leaf
4. Verify splitters are created from payload
"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth

print("="*70)
print("END-TO-END TEST: Splitter creation from activated leaf")
print("="*70)

# ===== TRAINING PROCESS =====
print("\n[TRAINING PROCESS]")
print("1. Training model to activate leaves...")

training_model = HoeffdingTreeClassifier(
    grace_period=50,  # Lower to activate leaves faster
    delta=1e-3,
    leaf_prediction='nba',
    leaf_update_threshold=50
)

dataset = synth.Agrawal(classification_function=0, seed=42)
for i, (x, y) in enumerate(dataset.take(500), 1):  # More samples
    training_model.learn_one(x, y)

print(f"   Model nodes: {training_model.n_nodes}")
print(f"   Model height: {training_model.height}")

# Find an activated leaf (has splitters)
activated_leaf_id = None
for nid, node in training_model._node_registry.items():
    if hasattr(node, 'splitters') and len(node.splitters) > 0:
        activated_leaf_id = nid
        print(f"\n   Found activated leaf: ID={nid}")
        print(f"     Total weight: {node.total_weight}")
        print(f"     Stats: {dict(node.stats)}")
        print(f"     Splitters: {len(node.splitters)}")
        print(f"     Features: {list(node.splitters.keys())[:5]}")  # Show first 5
        break

if not activated_leaf_id:
    print("   ❌ No activated leaves found! Training more...")
    for i, (x, y) in enumerate(dataset.take(1000), 501):
        training_model.learn_one(x, y)
    
    # Try again
    for nid, node in training_model._node_registry.items():
        if hasattr(node, 'splitters') and len(node.splitters) > 0:
            activated_leaf_id = nid
            print(f"\n   Found activated leaf after more training: ID={nid}")
            print(f"     Splitters: {len(node.splitters)}")
            break

if not activated_leaf_id:
    print("   ❌ STILL no activated leaves! Checking all nodes...")
    for nid, node in training_model._node_registry.items():
        print(f"   Node {nid}: {type(node).__name__}, ", end="")
        if hasattr(node, 'splitters'):
            print(f"splitters={len(node.splitters)}, weight={node.total_weight}")
        else:
            print("(branch node)")
    exit(1)

# Create update payload
print("\n2. Creating update payload...")
update_payload = training_model.create_update_payload(activated_leaf_id, 'complete_node')
splitter_count = len(update_payload['data']['splitter_data']['splitters'])
print(f"   Payload created for node {activated_leaf_id}")
print(f"   Payload includes {splitter_count} splitters")

if splitter_count == 0:
    print("   ❌ ERROR: Payload has no splitters!")
    # Debug
    source_node = training_model.get_node_by_id(activated_leaf_id)
    print(f"   Source node splitters: {len(source_node.splitters)}")
    print(f"   Source node features: {list(source_node.splitters.keys()) if source_node.splitters else 'None'}")
    exit(1)

print(f"   Splitter features: {list(update_payload['data']['splitter_data']['splitters'].keys())[:5]}")

# ===== INFERENCE PROCESS =====
print("\n[INFERENCE PROCESS]")
print("3. Creating inference model with empty leaf...")

inference_model = HoeffdingTreeClassifier(grace_period=50, leaf_prediction='nba')
inference_model._root = None
new_leaf = inference_model._new_leaf()
inference_model._root = new_leaf
inference_model._n_active_leaves = 1

# Set matching node ID
new_leaf.node_id = activated_leaf_id
inference_model._node_registry[activated_leaf_id] = new_leaf

print(f"   Created leaf with ID {activated_leaf_id}")
print(f"   Splitters before update: {len(new_leaf.splitters)}")

# Apply update
print("\n4. Applying distributed update...")
result = inference_model.apply_distributed_update(update_payload)
print(f"   Update result: {result}")

# Verify
print("\n5. Verification:")
updated_leaf = inference_model.get_node_by_id(activated_leaf_id)

if updated_leaf and len(updated_leaf.splitters) > 0:
    print(f"   ✅ SUCCESS! Leaf now has {len(updated_leaf.splitters)} splitters")
    print(f"   Features: {list(updated_leaf.splitters.keys())[:5]}")
    print(f"   Stats: {dict(updated_leaf.stats)}")
    print(f"   Total weight: {updated_leaf.total_weight}")
    
    # Check Gaussian splitter
    for feature, splitter in list(updated_leaf.splitters.items())[:3]:  # Check first 3
        print(f"\n   Splitter '{feature}': {type(splitter).__name__}")
        if hasattr(splitter, '_att_dist_per_class'):
            for cls, dist in splitter._att_dist_per_class.items():
                if hasattr(dist, 'mu'):
                    print(f"     Class {cls}: μ={dist.mu:.2f}, σ={dist.sigma:.2f}, n={dist.n_samples:.0f}")
    
    # Test prediction
    print(f"\n6. Testing prediction...")
    test_x = {'salary': 50000, 'commission': 10000, 'age': 35, 'elevel': 2,
              'car': 5, 'zipcode': 3, 'hvalue': 200000, 'hyears': 10, 'loan': 100000}
    pred = inference_model.predict_one(test_x)
    print(f"   Prediction: {pred}")
    
    print(f"\n{'='*70}")
    print(f"✅ END-TO-END TEST PASSED!")
    print(f"   - Splitters created from distributed update")
    print(f"   - {len(updated_leaf.splitters)} features synchronized")
    print(f"   - Model can make predictions")
    print(f"{'='*70}")
else:
    print(f"   ❌ FAILURE: Splitters not created")
    print(f"   Splitters: {len(updated_leaf.splitters) if updated_leaf else 'None'}")
