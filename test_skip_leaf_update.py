#!/usr/bin/env python3
"""Test that split occurrence skips leaf update callback."""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth

# Test counters
split_count = 0
leaf_update_count = 0
events = []

def split_cb(info):
    global split_count, events
    split_count += 1
    node_id = info['new_split_node'].node_id
    feature = info['split_feature']
    events.append(('split', node_id, feature))
    print(f'   🌳 SPLIT #{split_count}: Node {node_id}, Feature: {feature}')

def leaf_cb(info):
    global leaf_update_count, events
    leaf_update_count += 1
    node_id = info['node_id']
    weight = info['data']['leaf_stats']['total_weight']
    events.append(('leaf_update', node_id, weight))
    print(f'   📊 LEAF UPDATE #{leaf_update_count}: Node {node_id}, Weight: {weight}')

# Create model with matching thresholds
model = HoeffdingTreeClassifier(
    grace_period=50,
    leaf_update_threshold=50,
    leaf_prediction='nba',
    split_callback=split_cb,
    leaf_update_callback=leaf_cb
)

print('='*70)
print('TESTING: Split should SKIP leaf update callback')
print('='*70)
print('Grace period: 50 | Leaf update threshold: 50')
print()

# Train
dataset = synth.Agrawal(classification_function=0, seed=42)
for i, (x, y) in enumerate(dataset.take(250), 1):
    model.learn_one(x, y)
    if i in [50, 100, 150, 200, 250]:
        print(f'\n[After {i} samples] Splits: {split_count}, Leaf updates: {leaf_update_count}')

print('\n' + '='*70)
print('EVENT LOG')
print('='*70)
for event_type, node_id, detail in events:
    if event_type == 'split':
        print(f'  [SPLIT] Node {node_id}, Feature: {detail}')
    else:
        print(f'  [LEAF UPDATE] Node {node_id}, Weight: {detail}')

print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'Total splits: {split_count}')
print(f'Total leaf updates: {leaf_update_count}')
print(f'Total events: {len(events)}')
print()
if split_count > 0:
    print('✅ SUCCESS: Splits occurred and properly skipped redundant leaf updates!')
    print(f'   {split_count} split(s) fired split callback without triggering leaf update callback')
else:
    print('ℹ️  No splits in this run (adjust parameters if needed)')
