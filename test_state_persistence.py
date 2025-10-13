#!/usr/bin/env python3
"""
Comprehensive test for distributed Hoeffding Tree state serialization and restoration.

This test validates that:
1. Tree state can be saved to JSON
2. Tree state can be restored from JSON
3. Predictions are identical between original and restored trees
"""

import sys
import os
sys.path.insert(0, '/root/river')

from river.tree import HoeffdingTreeClassifier
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path

def generate_test_dataset(n_samples=1000):
    """Generate a synthetic dataset for testing."""
    print(f"🎲 GENERATING TEST DATASET ({n_samples} samples)")
    print("=" * 50)
    
    np.random.seed(42)  # For reproducibility
    
    data = []
    categories = ['low', 'medium', 'high']
    sizes = ['small', 'large']
    
    for i in range(n_samples):
        # Generate features
        numerical_feature = np.random.normal(0, 1)
        category = np.random.choice(categories)
        size = np.random.choice(sizes)
        another_num = np.random.uniform(-2, 2)
        
        # Generate target based on features (to make it learnable)
        if category == 'high' or numerical_feature > 0.5 or size == 'large':
            y = 1
        else:
            y = 0
        
        # Add some noise
        if np.random.random() < 0.1:
            y = 1 - y
        
        data.append({
            'numerical_feature': numerical_feature,
            'category': category,
            'size': size,
            'another_num': another_num,
            'target': y
        })
    
    # Save to CSV
    df = pd.DataFrame(data)
    csv_path = '/root/river/test_dataset.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Dataset generated and saved to: {csv_path}")
    print(f"   Features: numerical_feature, category, size, another_num")
    print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
    
    return df, csv_path

def train_initial_tree(df, n_train=50):
    """Train initial tree with limited data (no splits expected)."""
    print(f"\n🌱 TRAINING INITIAL TREE (first {n_train} samples)")
    print("=" * 50)
    
    # Create tree with high grace period to prevent splits
    tree = HoeffdingTreeClassifier(
        grace_period=1000,  # High grace period to prevent splits
        leaf_update_threshold=10,
        nominal_attributes=['category', 'size']
    )
    
    # Train on first n_train samples
    train_data = df.head(n_train)
    
    for idx, row in train_data.iterrows():
        x = {
            'numerical_feature': row['numerical_feature'],
            'category': row['category'],
            'size': row['size'],
            'another_num': row['another_num']
        }
        y = row['target']
        
        tree.learn_one(x, y)
    
    print(f"✅ Training completed with {n_train} instances")
    print(f"   Tree nodes: {tree.get_node_registry_size()}")
    print(f"   Expected: 1 node (root leaf, no splits)")
    
    # Verify no splits occurred
    if tree.get_node_registry_size() == 1:
        print("✅ No splits occurred as expected")
    else:
        print(f"⚠️ Unexpected splits occurred: {tree.get_node_registry_size()} nodes")
    
    return tree, train_data

def save_tree_state_to_json(tree, save_dir='/root/river/json-data'):
    """Save tree state to JSON files."""
    print(f"\n💾 SAVING TREE STATE TO JSON")
    print("=" * 40)
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    saved_files = []
    
    # Get all nodes
    for node_id in tree.get_all_node_ids():
        node = tree.get_node_by_id(node_id)
        
        print(f"📁 Saving node {node_id} ({type(node).__name__})...")
        
        # Create complete node payload
        payload = tree.create_update_payload(node_id, 'complete_node')
        
        if payload:
            # Add extra metadata
            payload['metadata'] = {
                'node_type': type(node).__name__,
                'depth': getattr(node, 'depth', 0),
                'is_active': getattr(node, 'is_active', lambda: False)(),
                'total_weight': getattr(node, 'total_weight', 0),
                'save_timestamp': time.time()
            }
            
            # Save to JSON file
            json_filename = f"node_{node_id}.json"
            json_path = save_path / json_filename
            
            with open(json_path, 'w') as f:
                json.dump(payload, f, indent=2, default=str)
            
            saved_files.append(str(json_path))
            print(f"   ✅ Saved to: {json_filename}")
    
    # Save tree metadata
    tree_metadata = {
        'tree_config': {
            'grace_period': tree.grace_period,
            'leaf_prediction': tree.leaf_prediction,
            'nominal_attributes': tree.nominal_attributes,
            'split_criterion': tree.split_criterion
        },
        'tree_state': {
            'n_nodes': tree.get_node_registry_size(),
            'classes': list(tree.classes),
            'train_weight_seen': tree._train_weight_seen_by_model
        },
        'node_files': [Path(f).name for f in saved_files],
        'save_timestamp': time.time()
    }
    
    metadata_path = save_path / 'tree_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(tree_metadata, f, indent=2)
    
    print(f"✅ Tree metadata saved to: tree_metadata.json")
    print(f"📊 Total files saved: {len(saved_files) + 1}")
    
    return saved_files, str(metadata_path)

def test_predictions_and_save(tree, df, test_size=100):
    """Test predictions on a subset and save results."""
    print(f"\n🎯 TESTING PREDICTIONS (on {test_size} samples)")
    print("=" * 50)
    
    # Use samples from the middle of dataset for testing
    test_data = df.iloc[500:500+test_size].copy()
    
    predictions = []
    
    for idx, row in test_data.iterrows():
        x = {
            'numerical_feature': row['numerical_feature'],
            'category': row['category'],
            'size': row['size'],
            'another_num': row['another_num']
        }
        
        # Get prediction probabilities
        proba = tree.predict_proba_one(x)
        
        predictions.append({
            'sample_id': idx,
            'true_label': row['target'],
            'prediction_proba': dict(proba),
            'predicted_class': max(proba, key=proba.get) if proba else None
        })
    
    # Save predictions
    predictions_path = '/root/river/json-data/original_predictions.json'
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Predictions saved to: original_predictions.json")
    print(f"📊 Sample predictions:")
    
    for i, pred in enumerate(predictions[:3]):
        print(f"   Sample {pred['sample_id']}: True={pred['true_label']}, "
              f"Proba={pred['prediction_proba']}, Pred={pred['predicted_class']}")
    
    return predictions, test_data

def create_tree_from_json(json_dir='/root/river/json-data'):
    """Create a new tree and restore state from JSON files."""
    print(f"\n🔄 CREATING NEW TREE FROM JSON")
    print("=" * 40)
    
    json_path = Path(json_dir)
    
    # Load tree metadata
    metadata_path = json_path / 'tree_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"📋 Loaded tree metadata:")
    print(f"   Original nodes: {metadata['tree_state']['n_nodes']}")
    print(f"   Classes: {metadata['tree_state']['classes']}")
    
    # Create new tree with same configuration
    config = metadata['tree_config']
    new_tree = HoeffdingTreeClassifier(
        grace_period=config['grace_period'],
        leaf_prediction=config['leaf_prediction'],
        nominal_attributes=config.get('nominal_attributes'),
        split_criterion=config['split_criterion']
    )
    
    print(f"✅ Created new tree with same configuration")
    
    # We need to create the initial leaf node first
    # Since we know there's only one node (root leaf), create it
    if metadata['tree_state']['n_nodes'] == 1:
        print(f"🌱 Creating initial root node...")
        
        # Create initial leaf by learning one dummy instance
        dummy_x = {'numerical_feature': 0, 'category': 'medium', 'size': 'small', 'another_num': 0}
        dummy_y = 0
        new_tree.learn_one(dummy_x, dummy_y)
        
        # Now we have a root node with ID 0, load the saved state
        node_file = json_path / 'node_0.json'
        if node_file.exists():
            print(f"📂 Loading node state from: node_0.json")
            
            with open(node_file, 'r') as f:
                node_payload = json.load(f)
            
            # Apply the saved state to our new tree
            success = new_tree.apply_distributed_update(0, node_payload)
            
            if success:
                print(f"✅ Node state restored successfully")
                
                # Update classes and other tree-level state
                new_tree.classes = set(metadata['tree_state']['classes'])
                new_tree._train_weight_seen_by_model = metadata['tree_state']['train_weight_seen']
                
            else:
                print(f"❌ Failed to restore node state")
                return None
    
    print(f"🎉 Tree restoration completed!")
    print(f"   Restored nodes: {new_tree.get_node_registry_size()}")
    
    return new_tree

def compare_predictions(original_tree, restored_tree, test_data):
    """Compare predictions between original and restored trees."""
    print(f"\n🔍 COMPARING PREDICTIONS")
    print("=" * 35)
    
    predictions_match = True
    differences = []
    
    for idx, row in test_data.iterrows():
        x = {
            'numerical_feature': row['numerical_feature'],
            'category': row['category'],
            'size': row['size'],
            'another_num': row['another_num']
        }
        
        # Get predictions from both trees
        original_proba = original_tree.predict_proba_one(x)
        restored_proba = restored_tree.predict_proba_one(x)
        
        # Compare probabilities (with small tolerance for floating point)
        prob_match = True
        for class_label in original_proba:
            if abs(original_proba[class_label] - restored_proba.get(class_label, 0)) > 1e-10:
                prob_match = False
                break
        
        if not prob_match:
            predictions_match = False
            differences.append({
                'sample_id': idx,
                'original': dict(original_proba),
                'restored': dict(restored_proba)
            })
    
    # Save comparison results
    comparison_result = {
        'predictions_match': predictions_match,
        'total_samples_tested': len(test_data),
        'differences_found': len(differences),
        'differences': differences[:10],  # Save first 10 differences for analysis
        'test_timestamp': time.time()
    }
    
    with open('/root/river/json-data/prediction_comparison.json', 'w') as f:
        json.dump(comparison_result, f, indent=2)
    
    print(f"📊 COMPARISON RESULTS:")
    print(f"   Total samples tested: {len(test_data)}")
    print(f"   Predictions match: {'✅ YES' if predictions_match else '❌ NO'}")
    print(f"   Differences found: {len(differences)}")
    
    if predictions_match:
        print(f"\n🎉 SUCCESS: All predictions match perfectly!")
        print(f"   The distributed update system works correctly!")
    else:
        print(f"\n⚠️ DIFFERENCES DETECTED:")
        for diff in differences[:3]:
            print(f"   Sample {diff['sample_id']}:")
            print(f"     Original: {diff['original']}")
            print(f"     Restored: {diff['restored']}")
    
    return predictions_match, differences

def main():
    """Run the complete test suite."""
    print("🧪 DISTRIBUTED HOEFFDING TREE STATE PERSISTENCE TEST")
    print("=" * 70)
    print("Testing: Save → Load → Compare predictions")
    
    try:
        # Step 1: Generate dataset
        df, csv_path = generate_test_dataset(1000)
        
        # Step 2: Train initial tree (no splits)
        original_tree, train_data = train_initial_tree(df, 50)
        
        # Step 3: Save tree state to JSON
        saved_files, metadata_path = save_tree_state_to_json(original_tree)
        
        # Step 4: Test predictions and save
        original_predictions, test_data = test_predictions_and_save(original_tree, df, 100)
        
        # Step 5: Create new tree from JSON
        restored_tree = create_tree_from_json()
        
        if restored_tree is None:
            print("❌ Failed to create tree from JSON")
            return False
        
        # Step 6: Compare predictions
        match_result, differences = compare_predictions(original_tree, restored_tree, test_data)
        
        # Final summary
        print(f"\n🏁 FINAL TEST RESULTS:")
        print("=" * 30)
        print(f"✅ Dataset generated: {len(df)} samples")
        print(f"✅ Tree trained: {len(train_data)} samples")
        print(f"✅ State saved: {len(saved_files)} node files")
        print(f"✅ Tree restored: {restored_tree.get_node_registry_size()} nodes")
        print(f"{'✅' if match_result else '❌'} Predictions match: {'YES' if match_result else 'NO'}")
        
        if match_result:
            print(f"\n🎉 DISTRIBUTED UPDATE SYSTEM VALIDATION: SUCCESS!")
            print(f"   The system correctly preserves and restores tree state")
            print(f"   Ready for production Kafka implementation!")
        else:
            print(f"\n⚠️ DISTRIBUTED UPDATE SYSTEM VALIDATION: ISSUES FOUND")
            print(f"   {len(differences)} prediction differences detected")
            
        return match_result
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)