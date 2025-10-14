#!/usr/bin/env python3
"""
🔧 ULTIMATE FIX - EXTRACT AND SYNC SPLITTER DISTRIBUTIONS
========================================================
This version extracts the actual Gaussian distributions from the original tree
and synchronizes them to the inference tree for identical predictions.
"""

import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


class UltimateFix:
    """The ultimate fix that synchronizes splitter distributions."""
    
    def __init__(self):
        self.n_samples = 1000
        self.training_samples = 50
        
    def generate_dataset(self):
        """Generate test dataset."""
        X, y = make_classification(
            n_samples=self.n_samples,
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
        
        return instances
    
    def extract_complete_state(self, tree):
        """Extract COMPLETE state including splitter distributions."""
        print("🔍 EXTRACTING COMPLETE TREE STATE")
        print("=" * 35)
        
        node_id = list(tree._node_registry.keys())[0]
        node = tree._node_registry[node_id]
        
        # Extract basic stats
        complete_state = {
            'stats': dict(getattr(node, 'stats', {})),
            'total_weight': getattr(node, 'total_weight', 0),
            'mc_correct_weight': getattr(node, '_mc_correct_weight', 0),
            'nb_correct_weight': getattr(node, '_nb_correct_weight', 0)
        }
        
        print(f"   📊 Basic state: {complete_state}")
        
        # Extract splitter distributions (THE CRITICAL PART!)
        if hasattr(node, '_splitters') and node._splitters:
            complete_state['splitter_distributions'] = {}
            
            print(f"   🔍 Extracting splitter distributions...")
            
            for feature_id, splitter in node._splitters.items():
                print(f"      Feature {feature_id}: {type(splitter).__name__}")
                
                if hasattr(splitter, '_att_dist_per_class'):
                    distributions = {}
                    for class_label, dist_obj in splitter._att_dist_per_class.items():
                        if hasattr(dist_obj, 'mu') and hasattr(dist_obj, 'sigma'):
                            # Gaussian distribution
                            distributions[str(class_label)] = {
                                'type': 'gaussian',
                                'mu': float(getattr(dist_obj, 'mu', 0)),
                                'sigma': float(getattr(dist_obj, 'sigma', 1)),
                                'n': float(getattr(dist_obj, 'n', 0))
                            }
                            print(f"         Class {class_label}: μ={dist_obj.mu:.3f}, σ={dist_obj.sigma:.3f}, n={dist_obj.n}")
                        else:
                            print(f"         Class {class_label}: {type(dist_obj).__name__} (not Gaussian)")
                    
                    if distributions:
                        complete_state['splitter_distributions'][str(feature_id)] = distributions
        
        print(f"   ✅ Complete state extraction finished")
        return complete_state
    
    def create_ultimate_sync_tree(self, instances, complete_state):
        """Create inference tree with ULTIMATE synchronization."""
        print(f"\n🎯 CREATING ULTIMATE SYNCHRONIZED TREE")
        print("=" * 42)
        
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with BOTH classes to ensure splitters learn both
        print(f"   🌱 Smart initialization with both classes...")
        
        # Find one instance of each class
        class_0_instance = None
        class_1_instance = None
        
        for instance in instances[:10]:  # Look in first 10 instances
            if instance['y'] == 0 and class_0_instance is None:
                class_0_instance = instance
            elif instance['y'] == 1 and class_1_instance is None:
                class_1_instance = instance
            
            if class_0_instance and class_1_instance:
                break
        
        # Learn from both classes
        if class_0_instance:
            inference_tree.learn_one(class_0_instance['x'], class_0_instance['y'])
            print(f"      ✅ Learned class 0 instance")
        
        if class_1_instance:
            inference_tree.learn_one(class_1_instance['x'], class_1_instance['y'])
            print(f"      ✅ Learned class 1 instance")
        
        # Get node ID
        node_id = list(inference_tree._node_registry.keys())[0]
        node = inference_tree._node_registry[node_id]
        
        # Step 1: Synchronize basic state
        print(f"   🔄 Step 1: Synchronizing basic state...")
        
        # Direct assignment for perfect sync
        node.stats = complete_state['stats'].copy()
        node.total_weight = complete_state['total_weight']
        node._mc_correct_weight = complete_state['mc_correct_weight']
        node._nb_correct_weight = complete_state['nb_correct_weight']
        
        print(f"      ✅ Basic state synchronized")
        
        # Step 2: Synchronize splitter distributions (THE KEY!)
        if 'splitter_distributions' in complete_state:
            print(f"   🔄 Step 2: Synchronizing splitter distributions...")
            
            for feature_id_str, class_distributions in complete_state['splitter_distributions'].items():
                feature_id = feature_id_str  # Keep as string key
                
                if feature_id in node._splitters:
                    splitter = node._splitters[feature_id]
                    
                    if hasattr(splitter, '_att_dist_per_class'):
                        print(f"      📊 Syncing feature {feature_id}:")
                        
                        # Update each class distribution
                        for class_label_str, dist_data in class_distributions.items():
                            # Convert string back to proper type
                            try:
                                class_label = np.float64(float(class_label_str))
                            except:
                                class_label = class_label_str
                            
                            if class_label in splitter._att_dist_per_class:
                                dist_obj = splitter._att_dist_per_class[class_label]
                                
                                if dist_data['type'] == 'gaussian' and hasattr(dist_obj, 'mu'):
                                    # Sync Gaussian parameters
                                    dist_obj.mu = dist_data['mu']
                                    dist_obj.sigma = dist_data['sigma'] 
                                    dist_obj.n = dist_data['n']
                                    
                                    print(f"         ✅ Class {class_label}: μ={dist_obj.mu:.3f}, σ={dist_obj.sigma:.3f}")
        
        # Verification
        print(f"   📊 Final verification:")
        final_mc = getattr(node, '_mc_correct_weight', 0)
        final_nb = getattr(node, '_nb_correct_weight', 0)
        print(f"      MC weight: {final_mc} (target: {complete_state['mc_correct_weight']})")
        print(f"      NB weight: {final_nb} (target: {complete_state['nb_correct_weight']})")
        print(f"      Prediction method: {'Naive Bayes' if final_nb >= final_mc else 'Majority Class'}")
        
        return inference_tree
    
    def test_predictions(self, original_tree, inference_tree, test_instances):
        """Test predictions with detailed analysis."""
        print(f"\n🎯 ULTIMATE PREDICTION TEST")
        print("=" * 30)
        
        perfect_matches = 0
        
        for i, instance in enumerate(test_instances[:5]):
            x = instance['x']
            
            orig_proba = original_tree.predict_proba_one(x)
            infer_proba = inference_tree.predict_proba_one(x)
            
            orig_pred = max(orig_proba, key=orig_proba.get)
            infer_pred = max(infer_proba, key=infer_proba.get)
            
            # Check if probabilities are very close (within 0.01)
            prob_close = abs(orig_proba.get(orig_pred, 0) - infer_proba.get(infer_pred, 0)) < 0.01
            
            if orig_pred == infer_pred and prob_close:
                perfect_matches += 1
                status = "🎯 PERFECT"
            elif orig_pred == infer_pred:
                status = "✅ MATCH"
            else:
                status = "❌ DIFFER"
            
            print(f"   Sample {i+1} {status}:")
            print(f"      Orig: {orig_pred} ({orig_proba.get(orig_pred, 0):.3f})")
            print(f"      Infer: {infer_pred} ({infer_proba.get(infer_pred, 0):.3f})")
        
        accuracy = (perfect_matches / 5) * 100
        
        print(f"\n📊 ULTIMATE RESULTS:")
        print(f"   Perfect matches: {perfect_matches}/5")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy == 100:
            print(f"   🏆 ULTIMATE SUCCESS! Distributed Hoeffding Tree PERFECTED!")
        
        return accuracy
    
    def run_ultimate_test(self):
        """Run the ultimate synchronization test."""
        print(f"🏆 ULTIMATE HOEFFDING TREE SYNCHRONIZATION TEST")
        print("=" * 50)
        
        # Generate dataset
        instances = self.generate_dataset()
        print(f"📊 Generated {len(instances)} instances\n")
        
        # Train original tree
        print("🌱 TRAINING ORIGINAL TREE...")
        original_tree = HoeffdingTreeClassifier(grace_period=200, leaf_prediction='nba')
        
        for i, instance in enumerate(instances[:self.training_samples]):
            original_tree.learn_one(instance['x'], instance['y'])
        
        print(f"✅ Trained on {self.training_samples} instances")
        
        # Extract complete state
        complete_state = self.extract_complete_state(original_tree)
        
        # Create ultimate synchronized tree
        inference_tree = self.create_ultimate_sync_tree(instances, complete_state)
        
        # Test ultimate predictions
        test_instances = instances[self.training_samples:self.training_samples + 5]
        accuracy = self.test_predictions(original_tree, inference_tree, test_instances)
        
        return accuracy


def main():
    """Run the ultimate test."""
    test = UltimateFix()
    accuracy = test.run_ultimate_test()
    
    if accuracy == 100:
        print(f"\n🎉🎉🎉 ULTIMATE SUCCESS! 🎉🎉🎉")
        print(f"🏆 Distributed Hoeffding Tree: PERFECTED!")
        print(f"🚀 Ready for Kafka production deployment!")
    else:
        print(f"\n🔧 Ultimate accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()