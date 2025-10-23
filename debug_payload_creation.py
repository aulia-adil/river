#!/usr/bin/env python3
"""Debug: Check what's in the payload"""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth
import json

print("="*70)
print("DEBUG: Checking payload contents")
print("="*70)

# Train model
model = HoeffdingTreeClassifier(grace_period=100, delta=1e-3)
dataset = synth.Agrawal(classification_function=0, seed=42)

for i, (x, y) in enumerate(dataset.take(300), 1):
    model.learn_one(x, y)

# Get all leaf IDs
leaf_ids = [nid for nid, node in model._node_registry.items() 
            if hasattr(node, 'splitters')]

print(f"\nFound {len(leaf_ids)} leaves: {leaf_ids}")

# Check each leaf's splitters
for leaf_id in leaf_ids:
    node = model.get_node_by_id(leaf_id)
    print(f"\nLeaf {leaf_id}:")
    print(f"  Total weight: {node.total_weight}")
    print(f"  Stats: {dict(node.stats)}")
    print(f"  Number of splitters: {len(node.splitters)}")
    
    if node.splitters:
        print(f"  Splitter features: {list(node.splitters.keys())}")
        # Check one splitter in detail
        first_feature = list(node.splitters.keys())[0]
        splitter = node.splitters[first_feature]
        print(f"\n  Sample splitter '{first_feature}':")
        print(f"    Type: {type(splitter).__name__}")
        if hasattr(splitter, '_att_dist_per_class'):
            for cls, dist in splitter._att_dist_per_class.items():
                if hasattr(dist, 'n_samples'):
                    print(f"    Class {cls}: n={dist.n_samples:.0f}, μ={dist.mu:.2f}")
    
    # Create payload
    payload = model.create_update_payload(leaf_id, 'complete_node')
    splitter_count = len(payload['data']['splitter_data']['splitters'])
    print(f"\n  Payload splitter count: {splitter_count}")
    
    if splitter_count > 0:
        print(f"  Payload splitters: {list(payload['data']['splitter_data']['splitters'].keys())}")
        # Print first splitter in detail
        first_key = list(payload['data']['splitter_data']['splitters'].keys())[0]
        first_splitter_data = payload['data']['splitter_data']['splitters'][first_key]
        print(f"\n  Sample payload splitter '{first_key}':")
        print(json.dumps(first_splitter_data, indent=4))
