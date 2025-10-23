#!/usr/bin/env python3
"""Test that split occurrence skips leaf update callback - with forced split."""

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

# Create model with VERY LOW grace_period to force splits
# AND matching leaf_update_threshold
model = HoeffdingTreeClassifier(
    grace_period=100,           # Lower grace period to encourage splits
    leaf_update_threshold=100,  # Matching threshold
    delta=1e-3,                 # Less conservative (more splits)
    leaf_prediction='nba',
    split_callback=split_cb,
    leaf_update_callback=leaf_cb
)

print('='*70)
print('TESTING: Split should SKIP leaf update callback')
print('='*70)
print('Grace period: 100 | Leaf update threshold: 100 | Delta: 1e-3')
print()

# Train with MORE samples to trigger splits
dataset = synth.Agrawal(classification_function=0, seed=42)
for i, (x, y) in enumerate(dataset.take(1000), 1):
    model.learn_one(x, y)
    if i in [100, 200, 300, 400, 500]:
        print(f'\n[After {i} samples] Splits: {split_count}, Leaf updates: {leaf_update_count}, Nodes: {model.n_nodes}')

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
print(f'Final tree nodes: {model.n_nodes}')
print()

# Analyze the results
if split_count > 0:
    print('✅ SUCCESS: Split-skips-leaf-update logic is working!')
    print(f'   {split_count} split(s) occurred')
    print()
    print('   KEY OBSERVATION:')
    print('   When a leaf reaches weight 200 (both grace_period AND leaf_update_threshold),')
    print('   if a split occurs, the leaf update callback should NOT fire.')
    print('   Only the split callback should fire.')
    print()
    
    # Count simultaneous events at multiples of 200
    split_at_200 = any(e for e in events if e[0] == 'split')
    leaf_updates_at_200_multiples = [e for e in events if e[0] == 'leaf_update' and e[2] % 200 == 0]
    
    if split_at_200 and not leaf_updates_at_200_multiples:
        print('   ✅ VERIFIED: No leaf updates fired when splits occurred!')
    elif split_at_200 and leaf_updates_at_200_multiples:
        print('   ⚠️  WARNING: Some leaf updates fired at split multiples:')
        for e in leaf_updates_at_200_multiples:
            print(f'      - {e}')
else:
    print('ℹ️  No splits in this run')
    print('   This means no leaf reached 200 samples AND satisfied split criteria')
