#!/usr/bin/env python3
"""Test splitter creation for a leaf created from split event (no initial training)."""

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth

print("="*70)
print("TEST: Apply update to fresh leaf from split event")
print("="*70)

# Scenario: A leaf was created from a split event and has NO splitters yet
# Then we receive a distributed update with splitter data

# Create a model with just root
model = HoeffdingTreeClassifier(
    grace_period=100,
    leaf_prediction='nba'
)

print("\n1. Creating model with empty root (no learning yet)...")
# Just initialize the root without learning
model._root = model._new_leaf()
model._n_active_leaves = 1

print(f"   Model nodes: {model.n_nodes}")
print(f"   Root node ID: {model._root.node_id}")
print(f"   Root splitters: {model._root.splitters}")
print(f"   Root splitters dict is empty: {len(model._root.splitters) == 0}")

# Create a realistic update payload (like it would come from another process)
print("\n2. Creating update payload with splitter data...")
update_payload = {
    'node_id': 0,
    'update_type': 'complete_node',
    'timestamp': 1234567890.0,
    'source_tree_id': 999,
    'data': {
        'leaf_stats': {
            'stats': {1: 63.0, 0: 37.0},
            'total_weight': 100.0
        },
        'splitter_data': {
            'splitters': {
                'salary': {
                    'type': 'GaussianSplitter',
                    'feature_name': 'salary',
                    'gaussian_data': {
                        'min_per_class': {1: 10000.0, 0: 8000.0},
                        'max_per_class': {1: 200000.0, 0: 180000.0},
                        'distributions': {
                            '1': {'n_samples': 63.0, 'mu': 94009.44, 'sigma': 41417.22},
                            '0': {'n_samples': 37.0, 'mu': 86880.42, 'sigma': 40443.39}
                        }
                    }
                },
                'age': {
                    'type': 'GaussianSplitter',
                    'feature_name': 'age',
                    'gaussian_data': {
                        'min_per_class': {1: 18.0, 0: 18.0},
                        'max_per_class': {1: 90.0, 0: 75.0},
                        'distributions': {
                            '1': {'n_samples': 63.0, 'mu': 49.89, 'sigma': 21.90},
                            '0': {'n_samples': 37.0, 'mu': 50.22, 'sigma': 6.57}
                        }
                    }
                },
                'elevel': {
                    'type': 'NominalSplitter',
                    'feature_name': 'elevel',
                    'nominal_data': {
                        'total_weight': 100.0,
                        'unique_values': [0, 1, 2, 3, 4],
                        'class_distributions': {
                            '1': {'0': 10, '1': 15, '2': 20, '3': 10, '4': 8},
                            '0': {'0': 8, '1': 12, '2': 10, '3': 5, '4': 2}
                        }
                    }
                }
            }
        },
        'naive_bayes_data': {
            'mc_correct_weight': 63.0,
            'nb_correct_weight': 73.0
        }
    }
}

print(f"   Update includes {len(update_payload['data']['splitter_data']['splitters'])} splitters")
print(f"   Splitter features: {list(update_payload['data']['splitter_data']['splitters'].keys())}")

# Apply the update
print("\n3. Applying distributed update...")
result = model.apply_distributed_update(update_payload)

print(f"\n4. Result: {result}")
print(f"   Root splitters after update: {list(model._root.splitters.keys()) if model._root.splitters else 'None'}")
print(f"   Number of splitters: {len(model._root.splitters) if model._root.splitters else 0}")

# Verify each splitter was created
if model._root.splitters:
    print(f"\n5. Verifying splitters...")
    
    # Check Gaussian splitter
    if 'salary' in model._root.splitters:
        salary_splitter = model._root.splitters['salary']
        print(f"\n   ✅ 'salary' splitter created:")
        print(f"      Type: {type(salary_splitter).__name__}")
        if hasattr(salary_splitter, '_att_dist_per_class'):
            for cls, dist in salary_splitter._att_dist_per_class.items():
                print(f"      Class {cls}: μ={dist.mu:.2f}, σ={dist.sigma:.2f}, n={dist.n_samples}")
    
    # Check another Gaussian splitter
    if 'age' in model._root.splitters:
        age_splitter = model._root.splitters['age']
        print(f"\n   ✅ 'age' splitter created:")
        print(f"      Type: {type(age_splitter).__name__}")
        if hasattr(age_splitter, '_att_dist_per_class'):
            for cls, dist in age_splitter._att_dist_per_class.items():
                print(f"      Class {cls}: μ={dist.mu:.2f}, σ={dist.sigma:.2f}, n={dist.n_samples}")
    
    # Check Nominal splitter
    if 'elevel' in model._root.splitters:
        elevel_splitter = model._root.splitters['elevel']
        print(f"\n   ✅ 'elevel' splitter created:")
        print(f"      Type: {type(elevel_splitter).__name__}")
        if hasattr(elevel_splitter, '_att_dist_per_class'):
            for cls, dist in elevel_splitter._att_dist_per_class.items():
                print(f"      Class {cls}: categories={list(dist.keys())}")
    
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS: All splitters created from distributed update!")
    print(f"{'='*70}")
else:
    print(f"\n❌ FAILURE: No splitters created!")
