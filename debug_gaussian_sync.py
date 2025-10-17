#!/usr/bin/env python3
"""
🔍 DEBUG: Why are predictions non-deterministic?
===============================================
This script shows EXACTLY what's happening with Gaussian synchronization.
"""
import sys
sys.path.append('/root/river')

from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


def show_the_problem():
    """Demonstrate the root cause of non-deterministic predictions."""
    
    print("="*80)
    print("🔍 DEBUGGING NON-DETERMINISTIC PREDICTIONS")
    print("="*80)
    
    # Generate simple dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    instances = []
    for i in range(len(X)):
        instance = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
        instances.append({'x': instance, 'y': int(y[i])})
    
    # STEP 1: Train original tree
    print("\n📊 STEP 1: Train Original Tree")
    print("-" * 80)
    
    original_tree = HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'
    )
    
    for i in range(10):
        original_tree.learn_one(instances[i]['x'], instances[i]['y'])
    
    print(f"✅ Trained with 10 instances")
    
    # Get the node and show ORIGINAL Gaussian parameters
    node_id = list(original_tree._node_registry.keys())[0]
    orig_node = original_tree._node_registry[node_id]
    
    print(f"\n📈 ORIGINAL Gaussian Parameters (TARGET VALUES):")
    print("-" * 80)
    
    original_gaussians = {}
    
    if hasattr(orig_node, '_splitters'):
        for feature_name, splitter in orig_node._splitters.items():
            print(f"\nFeature: {feature_name}")
            if hasattr(splitter, '_att_dist_per_class'):
                for class_label, dist_obj in splitter._att_dist_per_class.items():
                    if hasattr(dist_obj, 'mu'):
                        key = f"{feature_name}_{class_label}"
                        original_gaussians[key] = {
                            'mu': dist_obj.mu,
                            'sigma': dist_obj.sigma,
                            'n': dist_obj.n_samples
                        }
                        print(f"  Class {class_label}:")
                        print(f"    μ (mean)      = {dist_obj.mu:.8f}")
                        print(f"    σ (std dev)   = {dist_obj.sigma:.8f}")
                        print(f"    n (samples)   = {dist_obj.n_samples}")
    
    # STEP 2: Create payload
    print(f"\n📦 STEP 2: Create Update Payload")
    print("-" * 80)
    
    payload = original_tree.create_update_payload(node_id, 'splitter_data')
    print(f"✅ Payload created")
    
    # STEP 3: Apply update TWICE to show randomness
    print(f"\n🔄 STEP 3: Apply Update TWICE (Same Payload, Different Results!)")
    print("-" * 80)
    
    for run in [1, 2]:
        print(f"\n{'='*40}")
        print(f"RUN #{run}")
        print(f"{'='*40}")
        
        # Create fresh inference tree
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with 2 instances (one per class)
        inference_tree.learn_one(instances[0]['x'], instances[0]['y'])
        inference_tree.learn_one(instances[1]['x'], instances[1]['y'])
        
        infer_node_id = list(inference_tree._node_registry.keys())[0]
        
        print(f"\n📡 Applying the SAME payload...")
        success = inference_tree.apply_distributed_update(infer_node_id, payload)
        print(f"   Update success: {success}")
        
        # Show RECONSTRUCTED Gaussian parameters
        infer_node = inference_tree._node_registry[infer_node_id]
        
        print(f"\n📈 RECONSTRUCTED Gaussian Parameters (Run #{run}):")
        
        if hasattr(infer_node, '_splitters'):
            for feature_name, splitter in infer_node._splitters.items():
                print(f"\nFeature: {feature_name}")
                if hasattr(splitter, '_att_dist_per_class'):
                    for class_label, dist_obj in splitter._att_dist_per_class.items():
                        if hasattr(dist_obj, 'mu'):
                            key = f"{feature_name}_{class_label}"
                            
                            # Compare with original
                            orig = original_gaussians.get(key, {})
                            mu_diff = abs(dist_obj.mu - orig.get('mu', 0))
                            sigma_diff = abs(dist_obj.sigma - orig.get('sigma', 0))
                            
                            match = "✅" if mu_diff < 0.001 and sigma_diff < 0.001 else "❌"
                            
                            print(f"  Class {class_label}: {match}")
                            print(f"    μ (mean)      = {dist_obj.mu:.8f}  (target: {orig.get('mu', 0):.8f}, diff: {mu_diff:.8f})")
                            print(f"    σ (std dev)   = {dist_obj.sigma:.8f}  (target: {orig.get('sigma', 0):.8f}, diff: {sigma_diff:.8f})")
                            print(f"    n (samples)   = {dist_obj.n_samples}")
        
        # Make prediction
        test_x = instances[20]['x']
        pred = inference_tree.predict_proba_one(test_x)
        predicted_class = max(pred, key=pred.get)
        
        print(f"\n🎯 Test Prediction (Run #{run}):")
        print(f"   Predicted: {predicted_class}")
        print(f"   Probabilities: {pred}")
    
    # Show original prediction
    print(f"\n{'='*80}")
    print("📊 ORIGINAL TREE PREDICTION (What we should match):")
    print("="*80)
    
    test_x = instances[20]['x']
    orig_pred = original_tree.predict_proba_one(test_x)
    orig_class = max(orig_pred, key=orig_pred.get)
    
    print(f"   Predicted: {orig_class}")
    print(f"   Probabilities: {orig_pred}")
    
    # Show the problematic code
    print(f"\n{'='*80}")
    print("❌ ROOT CAUSE: THE PROBLEM IS HERE!")
    print("="*80)
    
    print("""
In file: /root/river/river/tree/hoeffding_tree_classifier.py
Around line 876-891:

    # Generate synthetic values to achieve target parameters
    if target_n > 0:
        # Learn synthetic data to reach target mu/sigma approximately
        import numpy as np
        synthetic_data = np.random.normal(target_mu, max(target_sigma, 0.01), ...)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        THIS USES RANDOM DATA!
        
        for value in synthetic_data:
            new_gaussian.update(value)

🔴 PROBLEM:
   • np.random.normal() generates DIFFERENT random samples each run
   • Different samples → Different final μ and σ values
   • Different μ/σ → Different predictions
   • Result: NON-DETERMINISTIC behavior!

✅ SOLUTION NEEDED:
   We need to DIRECTLY set the Gaussian's internal parameters
   WITHOUT using random data generation.
    """)


if __name__ == "__main__":
    show_the_problem()


# Original content below (keeping for reference)
"""
🔍 DEBUG GAUSSIAN SYNCHRONIZATION ISSUE
=======================================
Show exactly why predictions are non-deterministic.
"""
import sys
sys.path.append('/root/river')

from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


def debug_gaussian_synchronization():
    """Debug the Gaussian synchronization process."""
    
    print("\n" + "="*80)
    print("🔍 DEBUGGING GAUSSIAN SYNCHRONIZATION")
    print("="*80)
    
    # Generate simple dataset
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create instances
    instances = []
    for i in range(len(X)):
        instance = {f'feature_{j}': X[i, j] for j in range(X.shape[1])}
        instances.append({'x': instance, 'y': int(y[i])})
    
    # Train original tree
    print("\n📊 STEP 1: Training Original Tree")
    print("-" * 80)
    original_tree = HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba',
        split_criterion='info_gain'
    )
    
    for i in range(10):
        original_tree.learn_one(instances[i]['x'], instances[i]['y'])
    
    # Extract root node and splitter data
    root = original_tree._root
    print(f"\n✅ Original tree trained with {10} instances")
    print(f"   Root stats: {dict(root.stats)}")
    
    # Show ORIGINAL Gaussian parameters
    print(f"\n📈 ORIGINAL Gaussian Parameters:")
    for feature_name, splitter in root.splitters.items():
        print(f"\n   Feature: {feature_name}")
        if hasattr(splitter, '_att_dist_per_class'):
            for class_label, dist_obj in splitter._att_dist_per_class.items():
                if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                    print(f"      Class {class_label}:")
                    print(f"         μ (mean)     = {dist_obj.mu:.6f}")
                    print(f"         σ (std dev)  = {dist_obj.sigma:.6f}")
                    print(f"         n (samples)  = {dist_obj.n_samples}")
    
    # Create payload (current approach)
    print(f"\n📦 STEP 2: Creating Update Payload")
    print("-" * 80)
    
    payload_data = {}
    for feature_name, splitter in root.splitters.items():
        if hasattr(splitter, '_att_dist_per_class'):
            gaussian_data = {'distributions': {}}
            
            for class_label, dist_obj in splitter._att_dist_per_class.items():
                if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                    gaussian_data['distributions'][str(class_label)] = {
                        'mu': dist_obj.mu,
                        'sigma': dist_obj.sigma,
                        'n_samples': dist_obj.n_samples
                    }
                    print(f"   Extracted {feature_name}, class {class_label}: μ={dist_obj.mu:.6f}, σ={dist_obj.sigma:.6f}")
            
            payload_data[feature_name] = {'gaussian_data': gaussian_data}
    
    # Simulate applying update TWICE to show non-determinism
    print(f"\n🔄 STEP 3: Applying Update TWICE (to show randomness)")
    print("-" * 80)
    
    for run in [1, 2]:
        print(f"\n   === RUN {run} ===")
        
        # Create fresh inference tree
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba',
            split_criterion='info_gain'
        )
        
        # Initialize with 1 dummy instance
        inference_tree.learn_one(instances[0]['x'], instances[0]['y'])
        
        # Get root node
        inference_root = inference_tree._root
        
        # Apply the distributed update
        update_payload = {
            'update_type': 'splitter_data',
            'data': {'splitters': payload_data}
        }
        
        print(f"\n   📡 Applying distributed update...")
        success = inference_tree.apply_distributed_update(0, update_payload)
        
        # Show RECONSTRUCTED Gaussian parameters
        print(f"\n   📈 RECONSTRUCTED Gaussian Parameters (Run {run}):")
        for feature_name, splitter in inference_root.splitters.items():
            print(f"\n      Feature: {feature_name}")
            if hasattr(splitter, '_att_dist_per_class'):
                for class_label, dist_obj in splitter._att_dist_per_class.items():
                    if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                        print(f"         Class {class_label}:")
                        print(f"            μ (mean)     = {dist_obj.mu:.6f}")
                        print(f"            σ (std dev)  = {dist_obj.sigma:.6f}")
                        print(f"            n (samples)  = {dist_obj.n_samples}")
        
        # Make a test prediction
        test_x = instances[20]['x']
        pred = inference_tree.predict_proba_one(test_x)
        predicted_class = max(pred, key=pred.get)
        confidence = pred[predicted_class]
        
        print(f"\n   🎯 Test Prediction (Run {run}):")
        print(f"      Predicted class: {predicted_class} (confidence: {confidence:.3f})")
        print(f"      Full probabilities: {pred}")
    
    # Compare original prediction
    print(f"\n" + "="*80)
    print("📊 ORIGINAL TREE PREDICTION:")
    print("="*80)
    test_x = instances[20]['x']
    orig_pred = original_tree.predict_proba_one(test_x)
    orig_class = max(orig_pred, key=orig_pred.get)
    orig_confidence = orig_pred[orig_class]
    print(f"   Predicted class: {orig_class} (confidence: {orig_confidence:.3f})")
    print(f"   Full probabilities: {orig_pred}")
    
    print(f"\n" + "="*80)
    print("🔍 ROOT CAUSE IDENTIFIED:")
    print("="*80)
    print("""
    ❌ PROBLEM: The _apply_splitter_data_update() method uses:
       
       np.random.normal(target_mu, max(target_sigma, 0.01), ...)
       
       This generates RANDOM synthetic data each time, which means:
       - Different random samples → Different final μ and σ
       - Non-deterministic predictions
       - Inference tree ≠ Original tree
    
    ✅ SOLUTION: We need to directly set the Gaussian parameters
       without using random data generation!
    """)


if __name__ == "__main__":
    debug_gaussian_synchronization()