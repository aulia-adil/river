"""
Split-Only Verification - Step by Step

This script:
1. Trains until a split occurs
2. Immediately makes predictions (no additional learning after split)
3. Saves the split event to JSON
4. Saves predictions from the training model
5. Reconstructs the tree from the split event JSON
6. Makes predictions with the same inputs
7. Compares the results to verify they match

Goal: Verify that split event contains sufficient data for exact reconstruction
"""

import json
from pathlib import Path
from river import tree
from river.datasets import synth
from river.tree.nodes.htc_nodes import LeafNaiveBayesAdaptive

# Output directories
Path("json_data/verification").mkdir(parents=True, exist_ok=True)

# Global state
split_occurred = False
split_data = None
training_predictions = []
test_samples = []

def split_callback(split_info):
    """Capture split event"""
    global split_occurred, split_data
    
    print("\n" + "="*80)
    print("🌳 SPLIT DETECTED!")
    print("="*80)
    
    original_leaf = split_info['original_leaf']
    new_split_node = split_info['new_split_node']
    new_leaves = split_info['new_leaves']
    
    # Extract split information
    split_node_type = type(new_split_node).__name__
    branch_params = {'feature': getattr(new_split_node, 'feature', None)}
    
    if hasattr(new_split_node, 'threshold'):
        branch_params['threshold'] = float(getattr(new_split_node, 'threshold'))
    
    # CRITICAL: Capture the splitter type so we can reconstruct it
    splitter_type = None
    if new_leaves and hasattr(new_leaves[0], 'splitter') and new_leaves[0].splitter is not None:
        splitter_type = type(new_leaves[0].splitter).__name__
    
    # Extract new leaf data
    new_leaves_data = []
    for idx, leaf in enumerate(new_leaves):
        leaf_data = {
            'node_id': getattr(leaf, 'node_id', None),
            'node_type': type(leaf).__name__,
            'depth': getattr(leaf, 'depth', 0),
            'branch_index': idx,
            'stats': {str(k): float(v) for k, v in getattr(leaf, 'stats', {}).items()},
            'total_weight': float(getattr(leaf, 'total_weight', 0))
        }
        new_leaves_data.append(leaf_data)
    
    # Build split event payload
    split_data = {
        'event_type': 'split',
        'original_leaf_id': getattr(original_leaf, 'node_id', 0),
        'splitter_type': splitter_type,  # Add splitter type to the event
        'split_node': {
            'node_id': getattr(new_split_node, 'node_id', None),
            'node_type': split_node_type,
            'depth': getattr(new_split_node, 'depth', 0),
            'stats': {str(k): float(v) for k, v in getattr(new_split_node, 'stats', {}).items()},
            'branch_params': branch_params
        },
        'new_leaves': new_leaves_data
    }
    
    split_occurred = True
    
    print(f"✅ Split captured:")
    print(f"   Feature: {branch_params.get('feature')}")
    print(f"   Threshold: {branch_params.get('threshold')}")
    print(f"   Split node ID: {split_data['split_node']['node_id']}")
    print(f"   New leaves: {[leaf['node_id'] for leaf in new_leaves_data]}")
    print("="*80 + "\n")


def train_until_split():
    """Train model until split occurs"""
    print("🚀 PHASE 1: TRAINING UNTIL SPLIT")
    print("="*80)
    
    model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        split_callback=split_callback,
        leaf_prediction='nba'  # Use naive Bayes for simplicity
    )
    
    dataset = synth.Agrawal(classification_function=0, seed=42)
    
    instance_count = 0
    for x, y in dataset:
        instance_count += 1
        model.learn_one(x, y)
        
        if split_occurred:
            print(f"✅ Split occurred after {instance_count} instances")
            break
        
        if instance_count >= 10000:
            print("⚠️  Reached safety limit without split")
            break
    
    return model, instance_count


def make_predictions_training(model, n_samples=5):
    """Make predictions with the training model right after split"""
    global training_predictions, test_samples
    
    print("\n" + "="*80)
    print("🔮 PHASE 2: PREDICTIONS WITH TRAINING MODEL (IMMEDIATELY AFTER SPLIT)")
    print("="*80)
    
    # Generate test samples
    test_dataset = synth.Agrawal(classification_function=0, seed=999)
    test_samples = list(test_dataset.take(n_samples))
    
    training_predictions = []
    
    for i, (x, y_true) in enumerate(test_samples, 1):
        y_pred = model.predict_one(x)
        y_proba = model.predict_proba_one(x)
        
        # Find which node made the prediction
        current_node = model._root
        path = []
        
        while current_node is not None:
            node_id = getattr(current_node, 'node_id', 'unknown')
            node_type = type(current_node).__name__
            path.append({'node_id': node_id, 'node_type': node_type})
            
            if not hasattr(current_node, 'children'):
                break
            
            if hasattr(current_node, 'branch_no'):
                branch_idx = current_node.branch_no(x)
                if hasattr(current_node, 'children') and branch_idx < len(current_node.children):
                    current_node = current_node.children[branch_idx]
                else:
                    break
            else:
                break
        
        prediction_node = path[-1] if path else {'node_id': 'unknown', 'node_type': 'unknown'}
        
        # Get branch decision
        branch_decision = None
        if len(path) > 1:
            split_node = model._root
            if hasattr(split_node, 'feature') and hasattr(split_node, 'threshold'):
                feature_val = x[split_node.feature]
                if feature_val <= split_node.threshold:
                    branch_decision = f"{split_node.feature} <= {split_node.threshold} (value: {feature_val:.2f})"
                else:
                    branch_decision = f"{split_node.feature} > {split_node.threshold} (value: {feature_val:.2f})"
        
        result = {
            'sample_id': i,
            'input_features': {k: float(v) if isinstance(v, (int, float)) else v for k, v in x.items()},
            'true_label': int(y_true),
            'predicted_label': int(y_pred) if y_pred is not None else None,
            'prediction_probabilities': {str(k): float(v) for k, v in y_proba.items()} if y_proba else {},
            'prediction_node_id': prediction_node['node_id'],
            'decision_path': path,
            'branch_decision': branch_decision
        }
        
        training_predictions.append(result)
        
        print(f"Sample #{i}: age={x['age']:.1f}")
        print(f"  Branch: {branch_decision}")
        print(f"  Node: {prediction_node['node_id']}, Predicted: {y_pred}, True: {y_true}")
        print(f"  Probabilities: {y_proba}")
    
    print(f"\n✅ Made {len(training_predictions)} predictions with training model")


def reconstruct_and_predict():
    """Reconstruct tree from split JSON and make predictions"""
    print("\n" + "="*80)
    print("🔧 PHASE 3: RECONSTRUCT TREE FROM SPLIT EVENT")
    print("="*80)
    
    # Create new model
    inference_model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'  # Use naive Bayes for simplicity
    )
    
    print(f"Initial state: {inference_model.n_nodes} nodes")
    
    # Apply split event
    from river.tree.nodes.branch import NumericBinaryBranch
    from river.tree.nodes.htc_nodes import LeafMajorityClass
    
    split_node_info = split_data['split_node']
    new_leaves_info = split_data['new_leaves']
    splitter_type = split_data.get('splitter_type')
    
    print(f"\nApplying split event:")
    print(f"  Split node ID: {split_node_info['node_id']}")
    print(f"  Split type: {split_node_info['node_type']}")
    print(f"  Feature: {split_node_info['branch_params']['feature']}")
    print(f"  Threshold: {split_node_info['branch_params']['threshold']}")
    print(f"  Splitter type: {splitter_type}")
    
    # Create leaves with proper splitter - this is critical!
    leaves = []
    for leaf_info in new_leaves_info:
        stats = {int(k): v for k, v in leaf_info['stats'].items()}
        # Use the model's splitter (same as _new_leaf does)
        leaf = LeafNaiveBayesAdaptive(stats=stats, depth=leaf_info['depth'], splitter=inference_model.splitter)
        leaves.append(leaf)
        print(f"  Created leaf: node_id will be {leaf_info['node_id']}, stats={stats}, splitter={type(leaf.splitter).__name__}")
    
    # Create split node with leaves
    stats_dict = {int(k): v for k, v in split_node_info['stats'].items()}
    split_node = NumericBinaryBranch(
        stats=stats_dict,
        feature=split_node_info['branch_params']['feature'],
        threshold=split_node_info['branch_params']['threshold'],
        depth=split_node_info['depth'],
        left=leaves[0],
        right=leaves[1]
    )
    
    # Set as root
    inference_model._root = split_node
    
    # Register nodes
    if hasattr(inference_model, '_register_node'):
        inference_model._register_node(split_node, split_node_info['node_id'])
        for leaf, leaf_info in zip(leaves, new_leaves_info):
            inference_model._register_node(leaf, leaf_info['node_id'])
    
    print(f"\n✅ Tree reconstructed: {inference_model.n_nodes} nodes, height {inference_model.height}")
    
    # Make predictions with same inputs
    print("\n" + "="*80)
    print("🔮 PHASE 4: PREDICTIONS WITH RECONSTRUCTED MODEL")
    print("="*80)
    
    inference_predictions = []
    
    for i, (x, y_true) in enumerate(test_samples, 1):
        y_pred = inference_model.predict_one(x)
        y_proba = inference_model.predict_proba_one(x)
        
        # Find which node made the prediction
        current_node = inference_model._root
        path = []
        
        while current_node is not None:
            node_id = getattr(current_node, 'node_id', 'unknown')
            node_type = type(current_node).__name__
            path.append({'node_id': node_id, 'node_type': node_type})
            
            if not hasattr(current_node, 'children'):
                break
            
            if hasattr(current_node, 'branch_no'):
                branch_idx = current_node.branch_no(x)
                if hasattr(current_node, 'children') and branch_idx < len(current_node.children):
                    current_node = current_node.children[branch_idx]
                else:
                    break
            else:
                break
        
        prediction_node = path[-1] if path else {'node_id': 'unknown', 'node_type': 'unknown'}
        
        # Get branch decision
        branch_decision = None
        if len(path) > 1:
            split_node_recon = inference_model._root
            if hasattr(split_node_recon, 'feature') and hasattr(split_node_recon, 'threshold'):
                feature_val = x[split_node_recon.feature]
                if feature_val <= split_node_recon.threshold:
                    branch_decision = f"{split_node_recon.feature} <= {split_node_recon.threshold} (value: {feature_val:.2f})"
                else:
                    branch_decision = f"{split_node_recon.feature} > {split_node_recon.threshold} (value: {feature_val:.2f})"
        
        result = {
            'sample_id': i,
            'input_features': {k: float(v) if isinstance(v, (int, float)) else v for k, v in x.items()},
            'true_label': int(y_true),
            'predicted_label': int(y_pred) if y_pred is not None else None,
            'prediction_probabilities': {str(k): float(v) for k, v in y_proba.items()} if y_proba else {},
            'prediction_node_id': prediction_node['node_id'],
            'decision_path': path,
            'branch_decision': branch_decision
        }
        
        inference_predictions.append(result)
        
        print(f"Sample #{i}: age={x['age']:.1f}")
        print(f"  Branch: {branch_decision}")
        print(f"  Node: {prediction_node['node_id']}, Predicted: {y_pred}, True: {y_true}")
        print(f"  Probabilities: {y_proba}")
    
    return inference_predictions


def compare_predictions(training_preds, inference_preds):
    """Compare predictions from both models"""
    print("\n" + "="*80)
    print("🔍 PHASE 5: COMPARISON")
    print("="*80)
    
    all_match = True
    differences = []
    
    for i in range(len(training_preds)):
        train = training_preds[i]
        infer = inference_preds[i]
        
        sample_match = True
        diff = {'sample_id': i + 1, 'issues': []}
        
        # Compare predictions
        if train['predicted_label'] != infer['predicted_label']:
            sample_match = False
            diff['issues'].append(f"Prediction: {train['predicted_label']} vs {infer['predicted_label']}")
        
        # Compare probabilities (need to normalize for comparison)
        # Training model might include all classes with 0.0, inference might omit them
        train_probs = train['prediction_probabilities']
        infer_probs = infer['prediction_probabilities']
        
        # Check if probabilities are functionally equivalent
        probs_match = True
        all_classes = set(train_probs.keys()) | set(infer_probs.keys())
        for cls in all_classes:
            train_val = train_probs.get(cls, 0.0)
            infer_val = infer_probs.get(cls, 0.0)
            if abs(train_val - infer_val) > 1e-10:
                probs_match = False
                break
        
        if not probs_match:
            sample_match = False
            diff['issues'].append(f"Probabilities: {train['prediction_probabilities']} vs {infer['prediction_probabilities']}")
        
        # Compare which node made prediction
        if train['prediction_node_id'] != infer['prediction_node_id']:
            sample_match = False
            diff['issues'].append(f"Node: {train['prediction_node_id']} vs {infer['prediction_node_id']}")
        
        if sample_match:
            print(f"✅ Sample #{i+1}: MATCH")
        else:
            print(f"❌ Sample #{i+1}: MISMATCH")
            for issue in diff['issues']:
                print(f"   {issue}")
            all_match = False
            differences.append(diff)
    
    print("\n" + "="*80)
    if all_match:
        print("🎉 SUCCESS! All predictions match perfectly!")
    else:
        print(f"⚠️  ISSUES FOUND: {len(differences)} samples have mismatches")
    print("="*80)
    
    return all_match, differences


def save_results(training_preds, inference_preds, match_result, differences, instance_count):
    """Save all results to JSON files"""
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    
    # Save split event
    split_file = 'json_data/verification/split_event.json'
    with open(split_file, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"✅ Saved: {split_file}")
    
    # Save training predictions
    training_file = 'json_data/verification/training_predictions.json'
    with open(training_file, 'w') as f:
        json.dump({'predictions': training_preds, 'source': 'training_model'}, f, indent=2)
    print(f"✅ Saved: {training_file}")
    
    # Save inference predictions
    inference_file = 'json_data/verification/inference_predictions.json'
    with open(inference_file, 'w') as f:
        json.dump({'predictions': inference_preds, 'source': 'reconstructed_model'}, f, indent=2)
    print(f"✅ Saved: {inference_file}")
    
    # Save comparison results
    comparison_file = 'json_data/verification/comparison_results.json'
    comparison_data = {
        'all_match': match_result,
        'total_samples': len(training_preds),
        'matching_samples': len(training_preds) - len(differences),
        'mismatching_samples': len(differences),
        'differences': differences,
        'training_instances': instance_count
    }
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✅ Saved: {comparison_file}")


def main():
    """Main verification process"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*20 + "SPLIT-ONLY VERIFICATION" + " "*35 + "║")
    print("╚" + "="*78 + "╝\n")
    
    # Phase 1: Train until split
    model, instance_count = train_until_split()
    
    if not split_occurred:
        print("❌ No split occurred - cannot continue verification")
        return
    
    # Phase 2: Make predictions with training model
    make_predictions_training(model, n_samples=10)
    
    # Phase 3 & 4: Reconstruct and predict
    inference_preds = reconstruct_and_predict()
    
    # Phase 5: Compare
    match_result, differences = compare_predictions(training_predictions, inference_preds)
    
    # Save everything
    save_results(training_predictions, inference_preds, match_result, differences, instance_count)
    
    print("\n" + "="*80)
    print("✅ VERIFICATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
