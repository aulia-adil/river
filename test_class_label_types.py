#!/usr/bin/env python3
"""
Test: Verify class labels maintain correct type (int) after distributed update

Issue: When stats are serialized (e.g., through JSON), integer keys become strings.
This causes prediction issues because the model expects integer class labels.

This test verifies that the fix properly converts string keys back to integers.
"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth
import json

print("="*70)
print("TEST: Class Label Type Consistency")
print("="*70)

# Create training model
print("\n[TRAINING MODEL]")
training_model = HoeffdingTreeClassifier(grace_period=50, delta=1e-3)
dataset = synth.Agrawal(classification_function=0, seed=42)

for i, (x, y) in enumerate(dataset.take(500), 1):
    training_model.learn_one(x, y)

print(f"✓ Trained on 500 samples")

# Find an activated leaf
leaf_id = None
for nid, node in training_model._node_registry.items():
    if hasattr(node, 'stats') and len(node.stats) > 0:
        leaf_id = nid
        print(f"✓ Found leaf: ID={leaf_id}")
        print(f"  Stats: {dict(node.stats)}")
        print(f"  Stats key types: {[type(k).__name__ for k in node.stats.keys()]}")
        break

if not leaf_id:
    print("✗ No leaf found!")
    exit(1)

# Create payload
print("\n[CREATING PAYLOAD]")
payload = training_model.create_update_payload(leaf_id, 'complete_node')
print(f"✓ Payload created")
print(f"  Payload stats: {payload['data']['leaf_stats']['stats']}")
print(f"  Payload stats key types: {[type(k).__name__ for k in payload['data']['leaf_stats']['stats'].keys()]}")

# Simulate JSON serialization/deserialization (this converts int keys to strings)
print("\n[SIMULATING JSON SERIALIZATION]")
json_string = json.dumps(payload)
deserialized_payload = json.loads(json_string)
print(f"✓ Serialized and deserialized through JSON")
print(f"  Deserialized stats: {deserialized_payload['data']['leaf_stats']['stats']}")
print(f"  Deserialized stats key types: {[type(k).__name__ for k in deserialized_payload['data']['leaf_stats']['stats'].keys()]}")

# Create inference model
print("\n[INFERENCE MODEL - BEFORE UPDATE]")
inference_model = HoeffdingTreeClassifier(grace_period=50)
inference_model._root = None
new_leaf = inference_model._new_leaf()
inference_model._root = new_leaf
new_leaf.node_id = leaf_id
inference_model._node_registry[leaf_id] = new_leaf

print(f"✓ Created empty leaf: ID={leaf_id}")
print(f"  Stats: {dict(new_leaf.stats)}")

# Apply deserialized payload (with string keys)
print("\n[APPLYING DESERIALIZED UPDATE]")
result = inference_model.apply_distributed_update(deserialized_payload)

# Verify the result
print("\n[VERIFICATION]")
updated_leaf = inference_model.get_node_by_id(leaf_id)
print(f"✓ Update result: {result}")
print(f"  Updated stats: {dict(updated_leaf.stats)}")
print(f"  Updated stats key types: {[type(k).__name__ for k in updated_leaf.stats.keys()]}")

# Check if keys are integers (not strings)
all_int_keys = all(isinstance(k, int) for k in updated_leaf.stats.keys())
print(f"\n  All keys are integers: {all_int_keys}")

if all_int_keys:
    print(f"\n  ✅ TEST PASSED: Class labels maintained as integers!")
else:
    print(f"\n  ❌ TEST FAILED: Class labels are still strings!")
    exit(1)

# Test prediction to ensure it works correctly
print("\n[PREDICTION TEST]")
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

pred = inference_model.predict_one(test_sample)
proba = inference_model.predict_proba_one(test_sample)

print(f"  Prediction: {pred}")
print(f"  Prediction type: {type(pred).__name__}")
print(f"  Probabilities: {dict(proba)}")
print(f"  Probability key types: {[type(k).__name__ for k in proba.keys()]}")

# Verify prediction is correct type
if isinstance(pred, int):
    print(f"\n  ✅ Prediction is integer (correct)")
else:
    print(f"\n  ⚠️  Prediction is {type(pred).__name__} (may be unexpected)")

print("\n" + "="*70)
print("✅ CLASS LABEL TYPE TEST COMPLETED SUCCESSFULLY!")
print("="*70)
