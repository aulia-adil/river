#!/usr/bin/env python3
"""
🔍 UNDERSTANDING GaussianSplitter ATTRIBUTES
============================================
Let's understand what each attribute does and what's needed for distributed updates.
"""
import sys
sys.path.append('/root/river')

from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import numpy as np


def demonstrate_gaussian_splitter_usage():
    """Show how GaussianSplitter attributes are used."""
    
    print("="*80)
    print("🔍 UNDERSTANDING GaussianSplitter ATTRIBUTES")
    print("="*80)
    
    # Create and train a simple tree
    tree = HoeffdingTreeClassifier(grace_period=50, leaf_prediction='nba')
    
    # Generate simple data
    X, y = make_classification(n_samples=20, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # Train
    for i in range(20):
        x = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
        tree.learn_one(x, y[i])
    
    # Get the leaf node
    node_id = list(tree._node_registry.keys())[0]
    node = tree._node_registry[node_id]
    
    print("\n📊 EXAMINING A TRAINED LEAF NODE:")
    print("-" * 80)
    
    if hasattr(node, '_splitters'):
        for feature_name, splitter in node._splitters.items():
            print(f"\n🔍 Feature: {feature_name}")
            print(f"   Type: {type(splitter).__name__}")
            
            # Show all 4 attributes
            print(f"\n   1️⃣  _att_dist_per_class (Gaussian distributions):")
            if hasattr(splitter, '_att_dist_per_class'):
                for class_label, gaussian in splitter._att_dist_per_class.items():
                    print(f"      Class {class_label}: μ={gaussian.mu:.3f}, σ={gaussian.sigma:.3f}, n={gaussian.n_samples}")
            
            print(f"\n   2️⃣  _min_per_class (minimum values):")
            if hasattr(splitter, '_min_per_class'):
                for class_label, min_val in splitter._min_per_class.items():
                    print(f"      Class {class_label}: {min_val:.3f}")
            
            print(f"\n   3️⃣  _max_per_class (maximum values):")
            if hasattr(splitter, '_max_per_class'):
                for class_label, max_val in splitter._max_per_class.items():
                    print(f"      Class {class_label}: {max_val:.3f}")
            
            print(f"\n   4️⃣  n_splits (number of split candidates):")
            if hasattr(splitter, 'n_splits'):
                print(f"      {splitter.n_splits}")
    
    # Now test what's needed for PREDICTION
    print(f"\n" + "="*80)
    print("🎯 TESTING: What's needed for PREDICTION?")
    print("="*80)
    
    test_x = {f'feature_{j}': X[15, j] for j in range(X.shape[1])}
    
    print(f"\nTest instance: {test_x}")
    print(f"\nMaking prediction...")
    
    pred_proba = tree.predict_proba_one(test_x)
    pred_class = max(pred_proba, key=pred_proba.get)
    
    print(f"\n✅ Prediction: {pred_class}")
    print(f"   Probabilities: {pred_proba}")
    
    # Show which splitter method is called for prediction
    print(f"\n📋 How prediction works:")
    print(f"   1. For each feature, splitter.cond_proba() is called")
    print(f"   2. cond_proba() uses: _att_dist_per_class[target_val](att_val)")
    print(f"   3. This calls the Gaussian distribution's __call__ method")
    print(f"   4. Returns probability density at that point")
    
    # Now test what's needed for SPLITTING
    print(f"\n" + "="*80)
    print("🔀 TESTING: What's needed for SPLITTING?")
    print("="*80)
    
    print(f"\nWhen deciding to split:")
    print(f"   1. best_evaluated_split_suggestion() is called")
    print(f"   2. _split_point_suggestions() uses:")
    print(f"      ✅ _min_per_class (to find global min)")
    print(f"      ✅ _max_per_class (to find global max)")
    print(f"      ✅ n_splits (number of candidates)")
    print(f"   3. _class_dists_from_binary_split() uses:")
    print(f"      ✅ _att_dist_per_class (for Gaussian CDF)")
    print(f"      ✅ _min_per_class (boundary check)")
    print(f"      ✅ _max_per_class (boundary check)")
    
    # ANSWER THE USER'S QUESTION
    print(f"\n" + "="*80)
    print("💡 ANSWER TO YOUR QUESTION:")
    print("="*80)
    
    print(f"""
Your question: "Therefore, what is needed, is just the _att_dist_per_class 
               isn't it? At least for making an update in leaf?"

ANSWER: It depends on what you mean by "update in leaf"!

📊 For PREDICTION (inference) in a leaf:
   ✅ YES! You only need: _att_dist_per_class
   
   Why? Because cond_proba() only uses:
      return self._att_dist_per_class[target_val](att_val)

🔀 For SPLITTING decision in a leaf:
   ❌ NO! You need ALL 4 attributes:
      • _att_dist_per_class  (for Gaussian CDF calculations)
      • _min_per_class       (for split point boundaries)
      • _max_per_class       (for split point boundaries)  
      • n_splits             (for number of candidates)

🎯 For your DISTRIBUTED SYSTEM:

   If inference trees NEVER split (only predict):
      ✅ Send only: _att_dist_per_class
      
   If inference trees MIGHT split later:
      ✅ Send all 4: _att_dist_per_class, _min_per_class, 
                     _max_per_class, n_splits

🏆 RECOMMENDATION for your thesis:
   Since your inference trees are receiving updates and might
   eventually need to split themselves, it's SAFER to send:
   
   ✅ _att_dist_per_class (MUST have - for prediction)
   ✅ _min_per_class      (nice to have - for future splits)
   ✅ _max_per_class      (nice to have - for future splits)
   
   The min/max values are just single floats per class,
   so the overhead is minimal!
    """)


if __name__ == "__main__":
    demonstrate_gaussian_splitter_usage()
