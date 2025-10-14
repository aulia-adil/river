#!/usr/bin/env python3
"""
🔍 KAFKA DISTRIBUTED TREE DIAGNOSTIC
===================================
Diagnoses why inference predictions differ from original predictions.
Focuses on understanding the state synchronization issues.
"""

import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


def diagnostic_test():
    """Run diagnostic test to understand prediction differences."""
    print("🔍 DIAGNOSTIC: Why do inference predictions differ?")
    print("=" * 60)
    
    # Generate same dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    instances = []
    for i in range(len(df)):
        row = df.iloc[i]
        x = {col: row[col] for col in df.columns if col != 'target'}
        instances.append({'x': x, 'y': row['target']})
    
    print(f"📊 Dataset: {len(instances)} instances, features: {list(df.columns[:-1])}")
    
    # Create original tree (with training)
    print(f"\n🌱 CREATING ORIGINAL TREE")
    print("=" * 30)
    
    original_tree = HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'
    )
    
    # Train original tree with 50 instances
    for i, instance in enumerate(instances[:50]):
        x, y = instance['x'], instance['y']
        original_tree.learn_one(x, y)
        if (i + 1) % 10 == 0:
            print(f"   📈 Trained on {i + 1}/50 instances")
    
    print(f"✅ Original tree trained")
    original_tree.print_node_registry()
    
    # Create inference tree (with minimal training)
    print(f"\n🔍 CREATING INFERENCE TREE")
    print("=" * 30)
    
    inference_tree = HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'
    )
    
    # Initialize with just first instance
    inference_tree.learn_one(instances[0]['x'], instances[0]['y'])
    print(f"   🌱 Initialized with first instance only")
    
    # Now apply the final state from original tree
    original_node = list(original_tree._node_registry.values())[0]
    original_stats = dict(getattr(original_node, 'stats', {}))
    original_weight = getattr(original_node, 'total_weight', 0)
    
    print(f"   📊 Original tree final stats: {original_stats}")
    print(f"   ⚖️ Original tree final weight: {original_weight}")
    
    # Apply leaf stats synchronization
    update_payload = {
        'update_type': 'leaf_stats',
        'data': {
            'stats': original_stats,
            'total_weight': original_weight
        }
    }
    
    node_id = list(inference_tree._node_registry.keys())[0]
    success = inference_tree.apply_distributed_update(node_id, update_payload)
    print(f"   🔄 Stats synchronization: {'✅ Success' if success else '❌ Failed'}")
    
    inference_tree.print_node_registry()
    
    # Compare splitter states
    print(f"\n🔍 COMPARING SPLITTER STATES")
    print("=" * 35)
    
    orig_node = list(original_tree._node_registry.values())[0]
    infer_node = list(inference_tree._node_registry.values())[0]
    
    print(f"Original tree splitters:")
    if hasattr(orig_node, 'splitters') and orig_node.splitters:
        for feature, splitter in orig_node.splitters.items():
            if hasattr(splitter, '_att_dist_per_class'):
                print(f"   {feature}: {len(splitter._att_dist_per_class)} class distributions")
                for class_label, dist in splitter._att_dist_per_class.items():
                    print(f"      Class {class_label}: {dist}")
            else:
                print(f"   {feature}: No _att_dist_per_class")
    else:
        print(f"   No splitters found")
    
    print(f"\nInference tree splitters:")
    if hasattr(infer_node, 'splitters') and infer_node.splitters:
        for feature, splitter in infer_node.splitters.items():
            if hasattr(splitter, '_att_dist_per_class'):
                print(f"   {feature}: {len(splitter._att_dist_per_class)} class distributions")
                for class_label, dist in splitter._att_dist_per_class.items():
                    print(f"      Class {class_label}: {dist}")
            else:
                print(f"   {feature}: No _att_dist_per_class")
    else:
        print(f"   No splitters found")
    
    # Test prediction differences
    print(f"\n🎯 DETAILED PREDICTION ANALYSIS")
    print("=" * 40)
    
    test_instances = instances[50:55]  # Test first 5
    
    for i, instance in enumerate(test_instances):
        x = instance['x']
        
        print(f"\n📋 Sample {i + 1}:")
        print(f"   Features: {x}")
        
        # Get detailed probabilities
        orig_proba = original_tree.predict_proba_one(x)
        infer_proba = inference_tree.predict_proba_one(x)
        
        orig_pred = max(orig_proba, key=orig_proba.get)
        infer_pred = max(infer_proba, key=infer_proba.get)
        
        print(f"   Original probabilities: {orig_proba}")
        print(f"   Inference probabilities: {infer_proba}")
        print(f"   Original prediction: {orig_pred}")
        print(f"   Inference prediction: {infer_pred}")
        print(f"   Match: {'✅' if orig_pred == infer_pred else '❌'}")
    
    # Analyze why inference tree predicts all 1.0
    print(f"\n🤔 WHY DOES INFERENCE TREE PREDICT ALL 1.0?")
    print("=" * 50)
    
    # Check leaf stats
    infer_stats = dict(getattr(infer_node, 'stats', {}))
    print(f"Inference tree stats: {infer_stats}")
    
    if infer_stats:
        total_samples = sum(infer_stats.values())
        for class_label, count in infer_stats.items():
            percentage = (count / total_samples) * 100
            print(f"   Class {class_label}: {count} samples ({percentage:.1f}%)")
    
    # Check if it's using majority class prediction
    if hasattr(infer_node, '_mc_correct_weight') and hasattr(infer_node, '_nb_correct_weight'):
        mc_weight = getattr(infer_node, '_mc_correct_weight', 0)
        nb_weight = getattr(infer_node, '_nb_correct_weight', 0)
        print(f"Majority Class weight: {mc_weight}")
        print(f"Naive Bayes weight: {nb_weight}")
        
        if mc_weight >= nb_weight:
            print(f"🎯 DIAGNOSIS: Using Majority Class prediction!")
            majority_class = max(infer_stats, key=infer_stats.get) if infer_stats else 'unknown'
            print(f"   Majority class: {majority_class}")
        else:
            print(f"🧠 Using Naive Bayes prediction")
    
    # Check splitter learning
    if hasattr(infer_node, 'splitters') and infer_node.splitters:
        print(f"\nSplitter training status:")
        for feature, splitter in infer_node.splitters.items():
            if hasattr(splitter, '_att_dist_per_class'):
                n_classes = len(splitter._att_dist_per_class)
                print(f"   {feature}: {n_classes} classes learned")
                
                # Check if distributions are meaningful
                for class_label, dist in splitter._att_dist_per_class.items():
                    if hasattr(dist, 'n_samples'):
                        print(f"      Class {class_label}: {dist.n_samples} samples")
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 20)
    
    if len(set([orig_pred, infer_pred])) == 1:
        print("✅ Trees are predicting the same (might be majority class)")
    else:
        print("❌ Trees have different prediction logic")
        print("   🔧 Need to synchronize splitter distributions")
        print("   🔧 Or ensure both use same prediction mechanism")
    
    return {
        'original_stats': original_stats,
        'inference_stats': infer_stats,
        'original_predictions': [max(original_tree.predict_proba_one(inst['x']), key=original_tree.predict_proba_one(inst['x']).get) for inst in test_instances],
        'inference_predictions': [max(inference_tree.predict_proba_one(inst['x']), key=inference_tree.predict_proba_one(inst['x']).get) for inst in test_instances]
    }


if __name__ == "__main__":
    results = diagnostic_test()
    
    print(f"\n🏁 DIAGNOSTIC COMPLETE")
    print("=" * 25)
    print(f"Check the analysis above to understand prediction differences!")