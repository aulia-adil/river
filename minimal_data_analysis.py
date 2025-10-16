#!/usr/bin/env python3
"""
🔍 MINIMAL DATA ANALYSIS FOR DISTRIBUTED UPDATES
===============================================
Let's analyze exactly what data is needed to synchronize inference trees.
"""

import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import json


class MinimalDataAnalysis:
    """Analyze the minimal data requirements for distributed sync."""
    
    def analyze_required_data(self):
        """Analyze what data is actually needed for synchronization."""
        print("🔍 ANALYZING MINIMAL DATA REQUIREMENTS")
        print("=" * 45)
        
        # Create and train a tree
        tree = HoeffdingTreeClassifier(grace_period=30, leaf_prediction='nba')
        
        # Generate some training data
        X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)
        
        for i in range(50):
            x = {f'feature_{j}': X[i][j] for j in range(X.shape[1])}
            tree.learn_one(x, y[i])
        
        # Get the leaf node
        node_id = list(tree._node_registry.keys())[0]
        node = tree._node_registry[node_id]
        
        print(f"📊 LEAF NODE ANALYSIS:")
        print(f"   Node type: {type(node).__name__}")
        print(f"   Stats: {dict(getattr(node, 'stats', {}))}")
        print(f"   Total weight: {getattr(node, 'total_weight', 0)}")
        print(f"   MC weight: {getattr(node, '_mc_correct_weight', 0)}")
        print(f"   NB weight: {getattr(node, '_nb_correct_weight', 0)}")
        
        print(f"\n🔍 SPLITTER ANALYSIS:")
        if hasattr(node, '_splitters') and node._splitters:
            for feature_id, splitter in node._splitters.items():
                print(f"   Feature {feature_id}:")
                print(f"      Type: {type(splitter).__name__}")
                
                if hasattr(splitter, '_att_dist_per_class'):
                    print(f"      _att_dist_per_class: {len(splitter._att_dist_per_class)} classes")
                    for class_label, dist_obj in splitter._att_dist_per_class.items():
                        if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                            print(f"         Class {class_label}: Gaussian(μ={dist_obj.mu:.3f}, σ={dist_obj.sigma:.3f}, n={dist_obj.n_samples})")
                        else:
                            print(f"         Class {class_label}: {type(dist_obj).__name__}")
        else:
            print("   No splitters found")
        
        return tree, node
    
    def test_minimal_sync_approaches(self):
        """Test different minimal synchronization approaches."""
        print(f"\n🧪 TESTING MINIMAL SYNC APPROACHES")
        print("=" * 40)
        
        # Train original tree
        original_tree = HoeffdingTreeClassifier(grace_period=30, leaf_prediction='nba')
        X, y = make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)
        
        for i in range(50):
            x = {f'feature_{j}': X[i][j] for j in range(X.shape[1])}
            original_tree.learn_one(x, y[i])
        
        print("✅ Original tree trained")
        
        # Test different sync approaches
        approaches = [
            "1. Only leaf stats",
            "2. Only splitter distributions", 
            "3. Only Naive Bayes weights",
            "4. Splitter + NB weights",
            "5. All data (complete)"
        ]
        
        results = {}
        
        for i, approach in enumerate(approaches, 1):
            print(f"\n📊 Testing approach {i}: {approach.split('. ')[1]}")
            
            # Create inference tree
            inference_tree = HoeffdingTreeClassifier(grace_period=30, leaf_prediction='nba')
            
            # Initialize with both classes
            inference_tree.learn_one({f'feature_{j}': X[0][j] for j in range(X.shape[1])}, y[0])
            inference_tree.learn_one({f'feature_{j}': X[1][j] for j in range(X.shape[1])}, y[1])
            
            # Apply specific sync approach
            node_id = list(inference_tree._node_registry.keys())[0]
            
            if i == 1:  # Only leaf stats
                payload = original_tree.create_update_payload(list(original_tree._node_registry.keys())[0], 'leaf_stats')
            elif i == 2:  # Only splitter distributions
                payload = original_tree.create_update_payload(list(original_tree._node_registry.keys())[0], 'splitter_data')
            elif i == 3:  # Only NB weights
                payload = original_tree.create_update_payload(list(original_tree._node_registry.keys())[0], 'naive_bayes_data')
            elif i == 4:  # Splitter + NB
                # Create custom payload with both
                orig_node_id = list(original_tree._node_registry.keys())[0]
                splitter_payload = original_tree.create_update_payload(orig_node_id, 'splitter_data')
                nb_payload = original_tree.create_update_payload(orig_node_id, 'naive_bayes_data')
                
                payload = {
                    'update_type': 'complete_node',
                    'data': {
                        'splitter_data': splitter_payload['data'],
                        'naive_bayes_data': nb_payload['data']
                    }
                }
            else:  # Complete
                payload = original_tree.create_update_payload(list(original_tree._node_registry.keys())[0], 'complete_node')
            
            # Apply update
            success = inference_tree.apply_distributed_update(node_id, payload)
            
            # Test prediction accuracy
            test_instances = X[50:55]
            test_labels = y[50:55]
            matches = 0
            
            for j, (test_x, test_y) in enumerate(zip(test_instances, test_labels)):
                x_dict = {f'feature_{k}': test_x[k] for k in range(len(test_x))}
                orig_pred = original_tree.predict_one(x_dict)
                infer_pred = inference_tree.predict_one(x_dict)
                if orig_pred == infer_pred:
                    matches += 1
            
            accuracy = (matches / len(test_instances)) * 100
            results[approach] = {
                'success': success,
                'accuracy': accuracy,
                'matches': f"{matches}/{len(test_instances)}"
            }
            
            print(f"      Update success: {'✅' if success else '❌'}")
            print(f"      Accuracy: {accuracy:.1f}% ({matches}/{len(test_instances)})")
        
        # Summary
        print(f"\n📊 MINIMAL DATA REQUIREMENTS SUMMARY")
        print("=" * 45)
        
        for approach, result in results.items():
            status = "🏆" if result['accuracy'] > 60 else "✅" if result['accuracy'] > 20 else "❌"
            print(f"{status} {approach}: {result['accuracy']:.1f}% ({result['matches']})")
        
        # Find minimal sufficient approach
        best_minimal = None
        for approach, result in results.items():
            if result['accuracy'] > 40:  # Good enough threshold
                if best_minimal is None or len(approach) < len(best_minimal):
                    best_minimal = approach
        
        print(f"\n🎯 RECOMMENDATION:")
        if best_minimal:
            print(f"   Minimal sufficient data: {best_minimal.split('. ')[1]}")
        else:
            print("   Complete data package recommended")
        
        return results
    
    def analyze_splitter_only_approach(self):
        """Specifically analyze if ONLY splitter._att_dist_per_class is sufficient."""
        print(f"\n🎯 TESTING: ONLY splitter._att_dist_per_class")
        print("=" * 45)
        
        # Your specific question: Is only splitter._att_dist_per_class sufficient?
        print("Your question: 'Is only splitter._att_dist_per_class sufficient?'")
        print("\nAnalysis:")
        print("   🔍 splitter._att_dist_per_class contains:")
        print("      • Gaussian distributions for each class")
        print("      • μ (mean), σ (standard deviation), n (sample count)")
        print("      • This drives the Naive Bayes probability calculations")
        
        print("\n   🤔 What splitter._att_dist_per_class does NOT contain:")
        print("      • Leaf class statistics (node.stats)")
        print("      • MC/NB correctness weights")
        print("      • Total sample count")
        
        print("\n   💡 Impact analysis:")
        print("      ✅ Pros: Contains core prediction logic (Gaussian params)")
        print("      ❌ Cons: Missing class priors and weight decisions")
        
        # Test it empirically
        print(f"\n🧪 EMPIRICAL TEST:")
        
        # Create trees
        original = HoeffdingTreeClassifier(grace_period=30, leaf_prediction='nba')
        inference = HoeffdingTreeClassifier(grace_period=30, leaf_prediction='nba')
        
        # Generate data
        X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
        
        # Train original
        for i in range(40):
            x = {f'feature_{j}': X[i][j] for j in range(X.shape[1])}
            original.learn_one(x, y[i])
        
        # Initialize inference with minimal data
        inference.learn_one({f'feature_{j}': X[0][j] for j in range(X.shape[1])}, y[0])
        inference.learn_one({f'feature_{j}': X[1][j] for j in range(X.shape[1])}, y[1])
        
        # Extract ONLY splitter distributions
        orig_node = list(original._node_registry.values())[0]
        infer_node = list(inference._node_registry.values())[0]
        
        print("   📊 Copying ONLY _att_dist_per_class...")
        
        if hasattr(orig_node, '_splitters') and hasattr(infer_node, '_splitters'):
            for feature_id in orig_node._splitters:
                if feature_id in infer_node._splitters:
                    orig_splitter = orig_node._splitters[feature_id]
                    infer_splitter = infer_node._splitters[feature_id]
                    
                    if hasattr(orig_splitter, '_att_dist_per_class'):
                        # Copy the distributions
                        infer_splitter._att_dist_per_class = orig_splitter._att_dist_per_class.copy()
                        print(f"      ✅ Copied distributions for feature {feature_id}")
        
        # Test predictions
        test_x = {f'feature_{j}': X[50][j] for j in range(X.shape[1])}
        orig_pred = original.predict_one(test_x)
        infer_pred = inference.predict_one(test_x)
        
        orig_proba = original.predict_proba_one(test_x)
        infer_proba = inference.predict_proba_one(test_x)
        
        print(f"\n   🎯 Results:")
        print(f"      Original: {orig_pred} (prob: {orig_proba})")
        print(f"      Inference: {infer_pred} (prob: {infer_proba})")
        print(f"      Match: {'✅' if orig_pred == infer_pred else '❌'}")
        
        # Conclusion
        print(f"\n📋 CONCLUSION for splitter._att_dist_per_class ONLY:")
        if orig_pred == infer_pred:
            print("   ✅ SUFFICIENT: Only splitter distributions needed!")
        else:
            print("   ❌ INSUFFICIENT: Additional data required")
            print("   💡 Likely need: class statistics + NB weights")
        
        return orig_pred == infer_pred


def main():
    """Run the minimal data analysis."""
    analyzer = MinimalDataAnalysis()
    
    # Step 1: Analyze what data exists
    tree, node = analyzer.analyze_required_data()
    
    # Step 2: Test different approaches
    results = analyzer.test_minimal_sync_approaches()
    
    # Step 3: Answer the specific question
    sufficient = analyzer.analyze_splitter_only_approach()
    
    print(f"\n" + "="*50)
    print("🎯 FINAL ANSWER TO YOUR QUESTION:")
    print("="*50)
    print(f"Question: 'Should I give only splitter._att_dist_per_class?'")
    print(f"Answer: {'✅ YES - Sufficient' if sufficient else '❌ NO - Need more data'}")
    
    if not sufficient:
        print(f"\n💡 RECOMMENDATION:")
        print(f"   For reliable synchronization, you need:")
        print(f"   1. ✅ splitter._att_dist_per_class (Gaussian distributions)")
        print(f"   2. ✅ node.stats (class counts)")
        print(f"   3. ✅ MC/NB correctness weights")
        print(f"   → Use 'complete_node' or 'splitter_data + naive_bayes_data'")


if __name__ == "__main__":
    main()