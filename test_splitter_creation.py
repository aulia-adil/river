#!/usr/bin/env python3
"""Test that apply_distributed_update creates splitters if they don't exist."""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth

print("="*70)
print("TEST: Create splitters from distributed update")
print("="*70)

# Create a tree and train it
model1 = HoeffdingTreeClassifier(
    grace_period=100,
    leaf_prediction='nba',
    leaf_update_threshold=50
)

print("\n1. Training first model...")
dataset = synth.Agrawal(classification_function=0, seed=42)
for i, (x, y) in enumerate(dataset.take(100), 1):
    model1.learn_one(x, y)

print(f"   Model 1 nodes: {model1.n_nodes}")
print(f"   Model 1 root splitters: {list(model1._root.splitters.keys()) if hasattr(model1._root, 'splitters') and model1._root.splitters else 'None'}")

# Create update payload
print("\n2. Creating update payload from model 1...")
update_payload = model1.create_update_payload(0, 'complete_node')
print(f"   Payload created with {len(update_payload['data'])} sections")
print(f"   Splitters in payload: {list(update_payload['data']['splitter_data']['splitters'].keys())}")

# Create a fresh model with just the root
print("\n3. Creating fresh model 2 (receiver)...")
model2 = HoeffdingTreeClassifier(
    grace_period=100,
    leaf_prediction='nba'
)

# Initialize with just 1 sample to create root
x, y = next(iter(synth.Agrawal(classification_function=0, seed=999).take(1)))
model2.learn_one(x, y)

print(f"   Model 2 nodes: {model2.n_nodes}")
print(f"   Model 2 root splitters before update: {list(model2._root.splitters.keys()) if hasattr(model2._root, 'splitters') and model2._root.splitters else 'None'}")

# Apply the update
print("\n4. Applying distributed update to model 2...")
result = model2.apply_distributed_update(update_payload)

print(f"\n5. Update result: {result}")
print(f"   Model 2 root splitters after update: {list(model2._root.splitters.keys()) if hasattr(model2._root, 'splitters') and model2._root.splitters else 'None'}")

# Verify the splitters were created
if model2._root.splitters and len(model2._root.splitters) > 0:
    print(f"\n✅ SUCCESS: {len(model2._root.splitters)} splitters created from distributed update!")
    
    # Check a specific splitter
    if 'salary' in model2._root.splitters:
        salary_splitter = model2._root.splitters['salary']
        print(f"\n   Sample splitter 'salary':")
        print(f"   Type: {type(salary_splitter).__name__}")
        if hasattr(salary_splitter, '_att_dist_per_class'):
            print(f"   Classes tracked: {list(salary_splitter._att_dist_per_class.keys())}")
            for cls, dist in salary_splitter._att_dist_per_class.items():
                if hasattr(dist, 'mu'):
                    print(f"   Class {cls}: μ={dist.mu:.2f}, σ={dist.sigma:.2f}, n={dist.n_samples}")
else:
    print(f"\n❌ FAILURE: No splitters created!")
