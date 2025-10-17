"""
Capture Split Event and Subsequent Leaf Updates

This script trains a Hoeffding Tree until a split occurs, then captures
the split event data and the next 10 leaf update events.

Output:
- json_data/split_event/split_{number}.json - Split event data
- json_data/update_leaf/update_leaf_{number}.json - Leaf update data (10 files)
"""

import json
import os
from pathlib import Path
from river import tree
from river.datasets import synth

# Create output directories
Path("json_data/split_event").mkdir(parents=True, exist_ok=True)
Path("json_data/update_leaf").mkdir(parents=True, exist_ok=True)

# Track events
split_events = []
leaf_updates = []
split_occurred = False
leaf_updates_after_split = 0
max_leaf_updates_after_split = 10

def split_callback(split_info):
    """Callback when a split occurs - GATE 1"""
    global split_occurred, leaf_updates_after_split
    
    print("\n" + "="*80)
    print("🌳 SPLIT EVENT DETECTED!")
    print("="*80)
    
    original_leaf = split_info['original_leaf']
    new_split_node = split_info['new_split_node']
    new_leaves = split_info['new_leaves']
    parent = split_info['parent']
    parent_branch = split_info['parent_branch']
    
    # Extract split node type and parameters
    split_node_type = type(new_split_node).__name__
    
    # Build branch parameters based on node type
    branch_params = {
        'feature': getattr(new_split_node, 'feature', None)
    }
    
    if hasattr(new_split_node, 'threshold'):
        branch_params['threshold'] = float(getattr(new_split_node, 'threshold'))
    
    if hasattr(new_split_node, 'value'):
        branch_params['value'] = getattr(new_split_node, 'value')
    
    if hasattr(new_split_node, 'radius'):
        branch_params['radius'] = float(getattr(new_split_node, 'radius'))
    
    if hasattr(new_split_node, '_mapping'):
        # For multiway splits, extract the mapping
        branch_params['feature_values'] = list(getattr(new_split_node, '_mapping', {}).keys())
    
    # Extract new leaf data
    new_leaves_data = []
    for idx, leaf in enumerate(new_leaves):
        leaf_data = {
            'node_id': getattr(leaf, 'node_id', None),
            'node_type': type(leaf).__name__,
            'depth': getattr(leaf, 'depth', 0),
            'branch_index': idx,
            'stats': {str(k): float(v) for k, v in getattr(leaf, 'stats', {}).items()},
            'total_weight': float(getattr(leaf, 'total_weight', 0)),
            'mc_correct_weight': float(getattr(leaf, '_mc_correct_weight', 0)),
            'nb_correct_weight': float(getattr(leaf, '_nb_correct_weight', 0)),
        }
        
        # Extract splitter data for each leaf
        if hasattr(leaf, 'splitters'):
            splitters_data = {}
            for feature_name, splitter in leaf.splitters.items():
                splitter_info = {
                    'type': type(splitter).__name__,
                    'feature_name': feature_name
                }
                
                # Check if Gaussian splitter
                if hasattr(splitter, '_att_dist_per_class'):
                    att_dist = splitter._att_dist_per_class
                    if att_dist:
                        first_val = next(iter(att_dist.values()))
                        
                        # Gaussian splitter
                        if hasattr(first_val, 'mu'):
                            gaussian_data = {}
                            distributions = {}
                            
                            for class_label, dist_obj in att_dist.items():
                                class_data = {
                                    'n_samples': float(dist_obj.n_samples) if hasattr(dist_obj, 'n_samples') else 0.0,
                                    'mu': float(dist_obj.mu) if hasattr(dist_obj, 'mu') else 0.0,
                                    'sigma': float(dist_obj.sigma) if hasattr(dist_obj, 'sigma') else 1.0
                                }
                                distributions[str(class_label)] = class_data
                            
                            gaussian_data['distributions'] = distributions
                            
                            if hasattr(splitter, '_min_per_class'):
                                gaussian_data['min_per_class'] = {
                                    str(k): float(v) for k, v in splitter._min_per_class.items()
                                }
                            
                            if hasattr(splitter, '_max_per_class'):
                                gaussian_data['max_per_class'] = {
                                    str(k): float(v) for k, v in splitter._max_per_class.items()
                                }
                            
                            splitter_info['gaussian_data'] = gaussian_data
                        
                        # Nominal splitter
                        elif isinstance(first_val, dict):
                            nominal_data = {
                                'class_distributions': {
                                    str(k): dict(v) for k, v in att_dist.items()
                                }
                            }
                            
                            if hasattr(splitter, '_att_values'):
                                nominal_data['unique_values'] = list(splitter._att_values)
                            
                            if hasattr(splitter, '_total_weight_observed'):
                                nominal_data['total_weight'] = float(splitter._total_weight_observed)
                            
                            splitter_info['nominal_data'] = nominal_data
                
                splitters_data[feature_name] = splitter_info
            
            leaf_data['splitters'] = splitters_data
        
        new_leaves_data.append(leaf_data)
    
    # Build complete split event payload
    split_event = {
        'event_type': 'split',
        'timestamp': __import__('time').time(),
        
        # Original leaf that was replaced
        'original_leaf_id': getattr(original_leaf, 'node_id', None),
        
        # New split node created
        'split_node': {
            'node_id': getattr(new_split_node, 'node_id', None),
            'node_type': split_node_type,
            'depth': getattr(new_split_node, 'depth', 0),
            'stats': {str(k): float(v) for k, v in getattr(new_split_node, 'stats', {}).items()},
            'branch_params': branch_params
        },
        
        # New leaf children
        'new_leaves': new_leaves_data,
        
        # Parent context
        'parent_context': {
            'parent_node_id': getattr(parent, 'node_id', None) if parent else None,
            'parent_branch': parent_branch
        }
    }
    
    split_events.append(split_event)
    split_occurred = True
    
    print(f"✅ Split captured: {split_node_type} on feature '{branch_params.get('feature')}'")
    print(f"   Original leaf ID: {split_event['original_leaf_id']}")
    print(f"   New split node ID: {split_event['split_node']['node_id']}")
    print(f"   New leaves: {[leaf['node_id'] for leaf in new_leaves_data]}")
    print("="*80 + "\n")


def leaf_update_callback(update_info):
    """Callback when leaf update occurs - GATE 2"""
    global leaf_updates_after_split
    
    # Only capture leaf updates AFTER a split has occurred
    if not split_occurred:
        return
    
    # Stop after capturing max updates
    if leaf_updates_after_split >= max_leaf_updates_after_split:
        return
    
    print(f"\n📊 LEAF UPDATE #{leaf_updates_after_split + 1} (after split)")
    
    leaf_data = update_info['leaf_data']
    
    # Extract splitter data
    splitters_data = {}
    if 'splitters' in leaf_data:
        for feature_name, splitter_info in leaf_data['splitters'].items():
            splitter_entry = {
                'type': splitter_info.get('type'),
                'feature_name': splitter_info.get('feature_name')
            }
            
            if 'gaussian_data' in splitter_info:
                gaussian_data = splitter_info['gaussian_data']
                # Convert to serializable format
                serializable_gaussian = {
                    'distributions': gaussian_data.get('distributions', {}),
                    'min_per_class': gaussian_data.get('min_per_class', {}),
                    'max_per_class': gaussian_data.get('max_per_class', {})
                }
                splitter_entry['gaussian_data'] = serializable_gaussian
            
            if 'nominal_data' in splitter_info:
                splitter_entry['nominal_data'] = splitter_info['nominal_data']
            
            splitters_data[feature_name] = splitter_entry
    
    # Build leaf update payload
    leaf_update = {
        'event_type': 'leaf_update',
        'iteration': leaf_updates_after_split + 1,
        'timestamp': update_info['timestamp'],
        'node_id': update_info['node_id'],
        'node_type': update_info['node_type'],
        
        # Complete leaf state
        'leaf_data': {
            'depth': leaf_data['depth'],
            'stats': leaf_data['stats'],
            'total_weight': leaf_data['total_weight'],
            'mc_correct_weight': leaf_data['naive_bayes']['mc_correct_weight'],
            'nb_correct_weight': leaf_data['naive_bayes']['nb_correct_weight'],
            'splitters': splitters_data
        }
    }
    
    leaf_updates.append(leaf_update)
    leaf_updates_after_split += 1
    
    print(f"   Node ID: {update_info['node_id']}")
    print(f"   Stats: {leaf_data['stats']}")
    print(f"   Total weight: {leaf_data['total_weight']}")
    print(f"   MC weight: {leaf_data['naive_bayes']['mc_correct_weight']}")
    print(f"   NB weight: {leaf_data['naive_bayes']['nb_correct_weight']}")
    
    if leaf_updates_after_split >= max_leaf_updates_after_split:
        print(f"\n✅ Captured all {max_leaf_updates_after_split} leaf updates after split!")


def main():
    print("🚀 Starting Training: Capturing Split Event + 10 Leaf Updates")
    print("="*80)
    
    # Create tree with callbacks
    model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        split_callback=split_callback,
        leaf_update_callback=leaf_update_callback,
        leaf_update_threshold=1  # Trigger callback after every instance
    )
    
    # Generate dataset
    dataset = synth.Agrawal(classification_function=0, seed=42)
    
    # Train until we have split + 10 leaf updates
    instance_count = 0
    max_instances = 10000  # Safety limit
    
    for x, y in dataset:
        instance_count += 1
        
        # Train on instance
        model.learn_one(x, y)
        
        # Stop when we have captured everything we need
        if split_occurred and leaf_updates_after_split >= max_leaf_updates_after_split:
            print(f"\n✅ Training complete after {instance_count} instances")
            break
        
        # Safety limit
        if instance_count >= max_instances:
            print(f"\n⚠️  Reached safety limit of {max_instances} instances")
            break
    
    print("\n" + "="*80)
    print("💾 SAVING DATA TO JSON FILES")
    print("="*80)
    
    # Save split events
    for idx, split_event in enumerate(split_events):
        filename = f"json_data/split_event/split_{idx + 1}.json"
        with open(filename, 'w') as f:
            json.dump(split_event, f, indent=2)
        print(f"✅ Saved: {filename}")
        print(f"   Split node: {split_event['split_node']['node_type']}")
        print(f"   Feature: {split_event['split_node']['branch_params'].get('feature')}")
        print(f"   Threshold: {split_event['split_node']['branch_params'].get('threshold')}")
    
    # Save leaf updates
    for idx, leaf_update in enumerate(leaf_updates):
        filename = f"json_data/update_leaf/update_leaf_{idx + 1}.json"
        with open(filename, 'w') as f:
            json.dump(leaf_update, f, indent=2)
        print(f"✅ Saved: {filename}")
        print(f"   Node ID: {leaf_update['node_id']}")
        print(f"   Stats: {leaf_update['leaf_data']['stats']}")
    
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"Split events captured: {len(split_events)}")
    print(f"Leaf updates captured: {len(leaf_updates)}")
    print(f"Total instances processed: {instance_count}")
    
    # ========================================================================
    # PREDICTION PHASE - Make predictions on new samples
    # ========================================================================
    print("\n" + "="*80)
    print("🔮 MAKING PREDICTIONS ON TEST SAMPLES")
    print("="*80)
    
    # Generate test samples (different seed for testing)
    test_dataset = synth.Agrawal(classification_function=0, seed=999)
    test_samples = list(test_dataset.take(10))  # Get 10 test samples
    
    predictions = []
    
    for sample_idx, (x, y_true) in enumerate(test_samples, 1):
        # Make prediction
        y_pred = model.predict_one(x)
        y_proba = model.predict_proba_one(x)
        
        # Find which leaf node made the prediction
        # Traverse tree to find the leaf
        current_node = model._root
        path = []
        
        while current_node is not None:
            node_id = getattr(current_node, 'node_id', 'unknown')
            node_type = type(current_node).__name__
            path.append({'node_id': node_id, 'node_type': node_type})
            
            # If it's a leaf, stop
            if not hasattr(current_node, 'children'):
                break
            
            # If it's a branch, traverse to child
            if hasattr(current_node, 'branch_no'):
                branch_idx = current_node.branch_no(x)
                if hasattr(current_node, 'children') and branch_idx < len(current_node.children):
                    current_node = current_node.children[branch_idx]
                else:
                    break
            else:
                break
        
        # The last node in path is the leaf that made prediction
        prediction_node = path[-1] if path else {'node_id': 'unknown', 'node_type': 'unknown'}
        
        # Get branch decision details
        branch_decision = None
        if len(path) > 1:
            split_node = model._root
            if hasattr(split_node, 'feature') and hasattr(split_node, 'threshold'):
                feature_val = x[split_node.feature]
                if feature_val <= split_node.threshold:
                    branch_decision = f"{split_node.feature} <= {split_node.threshold} (value: {feature_val:.2f})"
                else:
                    branch_decision = f"{split_node.feature} > {split_node.threshold} (value: {feature_val:.2f})"
        
        # Build prediction record
        prediction_record = {
            'sample_id': sample_idx,
            'input_features': {k: float(v) if isinstance(v, (int, float)) else v for k, v in x.items()},
            'true_label': int(y_true),
            'predicted_label': int(y_pred) if y_pred is not None else None,
            'prediction_probabilities': {str(k): float(v) for k, v in y_proba.items()} if y_proba else {},
            'correct': bool(y_pred == y_true) if y_pred is not None else False,
            'prediction_node_id': prediction_node['node_id'],
            'prediction_node_type': prediction_node['node_type'],
            'decision_path': path,
            'branch_decision': branch_decision
        }
        
        predictions.append(prediction_record)
        
        # Print prediction details
        print(f"\nSample #{sample_idx}:")
        print(f"  Input: age={x['age']:.1f}, salary={x['salary']:.0f}, loan={x['loan']:.0f}")
        print(f"  Branch: {branch_decision}")
        print(f"  Predicted by: Node {prediction_node['node_id']} ({prediction_node['node_type']})")
        print(f"  True label: {y_true}, Predicted: {y_pred}, Correct: {y_pred == y_true}")
        print(f"  Probabilities: {y_proba}")
    
    # Calculate accuracy
    accuracy = sum(1 for p in predictions if p['correct']) / len(predictions)
    
    # Save predictions to JSON
    predictions_output = {
        'metadata': {
            'model_state': {
                'n_nodes': model.n_nodes,
                'height': model.height,
                'n_active_leaves': model.n_active_leaves,
                'training_instances': instance_count
            },
            'test_dataset': {
                'classification_function': 0,
                'seed': 999,
                'n_samples': len(predictions)
            }
        },
        'predictions': predictions,
        'summary': {
            'total_samples': len(predictions),
            'correct_predictions': sum(1 for p in predictions if p['correct']),
            'accuracy': accuracy,
            'predictions_per_node': {
                'node_1': sum(1 for p in predictions if p['prediction_node_id'] == 1),
                'node_2': sum(1 for p in predictions if p['prediction_node_id'] == 2)
            }
        }
    }
    
    predictions_file = 'json_data/predictions_with_nodes.json'
    with open(predictions_file, 'w') as f:
        json.dump(predictions_output, f, indent=2)
    
    print("\n" + "="*80)
    print("📊 PREDICTION SUMMARY")
    print("="*80)
    print(f"Total predictions: {len(predictions)}")
    print(f"Correct predictions: {sum(1 for p in predictions if p['correct'])}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Predictions by Node 1: {predictions_output['summary']['predictions_per_node']['node_1']}")
    print(f"Predictions by Node 2: {predictions_output['summary']['predictions_per_node']['node_2']}")
    print(f"\n💾 Predictions saved to: {predictions_file}")
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
