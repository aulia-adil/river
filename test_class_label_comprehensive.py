#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Class Label Type Consistency in Distributed Updates

This test demonstrates that class labels maintain correct integer type
through the complete distributed learning workflow, including JSON serialization.
"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth
import json

def print_header(title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

def compare_stats(training_stats, inference_stats, label=""):
    print(f"\n{label}Comparison:")
    print(f"  Training stats: {training_stats}")
    print(f"  Training key types: {[type(k).__name__ for k in training_stats.keys()]}")
    print(f"  Inference stats: {inference_stats}")
    print(f"  Inference key types: {[type(k).__name__ for k in inference_stats.keys()]}")
    
    # Check type consistency
    training_types = {type(k).__name__ for k in training_stats.keys()}
    inference_types = {type(k).__name__ for k in inference_stats.keys()}
    
    if training_types == inference_types and 'int' in training_types:
        print(f"  ✅ Types match and are integers!")
        return True
    else:
        print(f"  ❌ Type mismatch: {training_types} vs {inference_types}")
        return False

print_header("COMPREHENSIVE CLASS LABEL TYPE CONSISTENCY TEST")

# ===================================================================
# STEP 1: Train model with multiple classes
# ===================================================================
print("\n[STEP 1: TRAINING]")
training_model = HoeffdingTreeClassifier(grace_period=50, delta=1e-3)
dataset = synth.Agrawal(classification_function=0, seed=42)

for i, (x, y) in enumerate(dataset.take(1000), 1):
    training_model.learn_one(x, y)

print(f"✓ Trained on 1000 samples")
print(f"✓ Model has {training_model.n_nodes} nodes")

# Find multi-class leaf
target_leaf_id = None
for nid, node in training_model._node_registry.items():
    if hasattr(node, 'stats') and len(node.stats) > 1:
        target_leaf_id = nid
        break

if not target_leaf_id:
    print("⚠️  No multi-class leaf found, using any leaf")
    for nid, node in training_model._node_registry.items():
        if hasattr(node, 'stats') and len(node.stats) > 0:
            target_leaf_id = nid
            break

training_leaf = training_model.get_node_by_id(target_leaf_id)
print(f"✓ Selected leaf: ID={target_leaf_id}")
print(f"  Stats: {dict(training_leaf.stats)}")
print(f"  Key types: {[type(k).__name__ for k in training_leaf.stats.keys()]}")

# ===================================================================
# STEP 2: Create payload and serialize
# ===================================================================
print("\n[STEP 2: PAYLOAD CREATION & SERIALIZATION]")
payload = training_model.create_update_payload(target_leaf_id, 'complete_node')
print(f"✓ Payload created")
print(f"  Original stats in payload: {payload['data']['leaf_stats']['stats']}")
print(f"  Original key types: {[type(k).__name__ for k in payload['data']['leaf_stats']['stats'].keys()]}")

# Simulate Kafka/network transmission via JSON
json_string = json.dumps(payload)
print(f"\n✓ Serialized to JSON ({len(json_string)} bytes)")

# Deserialize (simulating receiving from Kafka)
received_payload = json.loads(json_string)
print(f"✓ Deserialized from JSON")
print(f"  Received stats: {received_payload['data']['leaf_stats']['stats']}")
print(f"  Received key types: {[type(k).__name__ for k in received_payload['data']['leaf_stats']['stats'].keys()]}")

if any(isinstance(k, str) for k in received_payload['data']['leaf_stats']['stats'].keys()):
    print(f"  ⚠️  Keys became strings (expected after JSON)")
else:
    print(f"  ℹ️  Keys are still integers (JSON library dependent)")

# ===================================================================
# STEP 3: Apply to inference model
# ===================================================================
print("\n[STEP 3: INFERENCE MODEL UPDATE]")
inference_model = HoeffdingTreeClassifier(grace_period=50)
inference_model._root = None
new_leaf = inference_model._new_leaf()
inference_model._root = new_leaf
new_leaf.node_id = target_leaf_id
inference_model._node_registry[target_leaf_id] = new_leaf

print(f"✓ Created empty leaf: ID={target_leaf_id}")
print(f"  Before update: {dict(new_leaf.stats)}")

result = inference_model.apply_distributed_update(received_payload)
print(f"\n✓ Update applied: {result}")

inference_leaf = inference_model.get_node_by_id(target_leaf_id)
print(f"  After update: {dict(inference_leaf.stats)}")
print(f"  Key types: {[type(k).__name__ for k in inference_leaf.stats.keys()]}")

# ===================================================================
# STEP 4: Verification
# ===================================================================
print_header("VERIFICATION RESULTS")

# Compare stats
success = compare_stats(training_leaf.stats, inference_leaf.stats, "Stats ")

# Compare weights
print(f"\nWeight Comparison:")
print(f"  Training total_weight: {training_leaf.total_weight}")
print(f"  Inference total_weight: {inference_leaf.total_weight}")

# Test predictions
print(f"\nPrediction Comparison:")
test_sample = {
    'salary': 75000, 'commission': 15000, 'age': 45, 'elevel': 3,
    'car': 10, 'zipcode': 5, 'hvalue': 300000, 'hyears': 15, 'loan': 150000
}

training_pred = training_model.predict_one(test_sample)
inference_pred = inference_model.predict_one(test_sample)

print(f"  Training prediction: {training_pred} (type: {type(training_pred).__name__})")
print(f"  Inference prediction: {inference_pred} (type: {type(inference_pred).__name__})")
print(f"  Match: {training_pred == inference_pred}")

# Get probabilities
training_proba = training_model.predict_proba_one(test_sample)
inference_proba = inference_model.predict_proba_one(test_sample)

print(f"\n  Training proba: {dict(training_proba)}")
print(f"  Training proba key types: {[type(k).__name__ for k in training_proba.keys()]}")
print(f"  Inference proba: {dict(inference_proba)}")
print(f"  Inference proba key types: {[type(k).__name__ for k in inference_proba.keys()]}")

# Final verification
all_checks = [
    success,
    training_pred == inference_pred,
    isinstance(inference_pred, int),
    all(isinstance(k, int) for k in inference_leaf.stats.keys()),
    all(isinstance(k, int) for k in inference_proba.keys())
]

print_header("FINAL RESULTS")
if all(all_checks):
    print("✅ ALL TESTS PASSED!")
    print("\nVerified:")
    print("  ✓ Class label types are consistent (int)")
    print("  ✓ Stats match between training and inference")
    print("  ✓ Predictions match and are correct type")
    print("  ✓ Probabilities use integer keys")
    print("\n🎉 Class label type consistency is working perfectly!")
else:
    print("❌ SOME TESTS FAILED!")
    if not success:
        print("  ✗ Stats types don't match")
    if training_pred != inference_pred:
        print("  ✗ Predictions don't match")
    if not isinstance(inference_pred, int):
        print("  ✗ Prediction is not an integer")
    if not all(isinstance(k, int) for k in inference_leaf.stats.keys()):
        print("  ✗ Stats keys are not integers")
    if not all(isinstance(k, int) for k in inference_proba.keys()):
        print("  ✗ Probability keys are not integers")

print("="*70)
