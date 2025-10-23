#!/usr/bin/env python3
"""
Test: Multi-class label type conversion

This test verifies that multiple class labels (0, 1, 2, etc.) 
all maintain their integer type after JSON serialization.
"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth
import json

print("="*70)
print("TEST: Multi-Class Label Type Consistency")
print("="*70)

# Create training model and train until we have both classes
print("\n[TRAINING MODEL]")
training_model = HoeffdingTreeClassifier(grace_period=50, delta=1e-3)
dataset = synth.Agrawal(classification_function=0, seed=42)

for i, (x, y) in enumerate(dataset.take(1000), 1):
    training_model.learn_one(x, y)

print(f"✓ Trained on 1000 samples")

# Find a leaf with multiple classes
leaf_id = None
for nid, node in training_model._node_registry.items():
    if hasattr(node, 'stats') and len(node.stats) > 1:  # Multiple classes
        leaf_id = nid
        print(f"✓ Found multi-class leaf: ID={leaf_id}")
        print(f"  Stats: {dict(node.stats)}")
        print(f"  Stats key types: {[type(k).__name__ for k in node.stats.keys()]}")
        print(f"  Classes present: {list(node.stats.keys())}")
        break

if not leaf_id:
    print("✗ No multi-class leaf found!")
    exit(1)

# Create payload and simulate JSON round-trip
print("\n[JSON SERIALIZATION]")
payload = training_model.create_update_payload(leaf_id, 'complete_node')
json_string = json.dumps(payload)
deserialized_payload = json.loads(json_string)

print(f"  Original stats: {payload['data']['leaf_stats']['stats']}")
print(f"  After JSON: {deserialized_payload['data']['leaf_stats']['stats']}")
print(f"  After JSON key types: {[type(k).__name__ for k in deserialized_payload['data']['leaf_stats']['stats'].keys()]}")

# Apply to inference model
print("\n[APPLYING TO INFERENCE MODEL]")
inference_model = HoeffdingTreeClassifier(grace_period=50)
inference_model._root = None
new_leaf = inference_model._new_leaf()
inference_model._root = new_leaf
new_leaf.node_id = leaf_id
inference_model._node_registry[leaf_id] = new_leaf

result = inference_model.apply_distributed_update(deserialized_payload)

# Verify
print("\n[VERIFICATION]")
updated_leaf = inference_model.get_node_by_id(leaf_id)
print(f"  Updated stats: {dict(updated_leaf.stats)}")
print(f"  Updated stats key types: {[type(k).__name__ for k in updated_leaf.stats.keys()]}")
print(f"  Classes present: {list(updated_leaf.stats.keys())}")

# Check all keys are integers
all_int_keys = all(isinstance(k, int) for k in updated_leaf.stats.keys())
print(f"\n  All keys are integers: {all_int_keys}")

# Compare with training model
training_leaf = training_model.get_node_by_id(leaf_id)
same_classes = set(training_leaf.stats.keys()) == set(updated_leaf.stats.keys())
print(f"  Same classes as training model: {same_classes}")

if all_int_keys and same_classes:
    print(f"\n✅ TEST PASSED: All class labels are integers and match!")
else:
    print(f"\n❌ TEST FAILED!")
    if not all_int_keys:
        print(f"  - Keys are not all integers")
    if not same_classes:
        print(f"  - Classes don't match: {training_leaf.stats.keys()} vs {updated_leaf.stats.keys()}")
    exit(1)

# Test prediction
print("\n[PREDICTION COMPARISON]")
test_sample = {
    'salary': 50000,
    'commission': 10000,
    'age': 35,
    'elevel': 2,
    'car': 5,
    'zipcode': 3,
    'hvalue': 200000,
    'hyears': 10,
    'loan': 100000
}

training_pred = training_model.predict_one(test_sample)
inference_pred = inference_model.predict_one(test_sample)

print(f"  Training model prediction: {training_pred} (type: {type(training_pred).__name__})")
print(f"  Inference model prediction: {inference_pred} (type: {type(inference_pred).__name__})")
print(f"  Predictions match: {training_pred == inference_pred}")

if training_pred == inference_pred and isinstance(inference_pred, int):
    print(f"\n✅ Predictions match and are correct type!")
else:
    print(f"\n⚠️  Predictions differ or wrong type")

print("\n" + "="*70)
print("✅ MULTI-CLASS LABEL TYPE TEST COMPLETED!")
print("="*70)
