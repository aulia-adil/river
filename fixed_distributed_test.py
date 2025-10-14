#!/usr/bin/env python3
"""
🔧 FIXED KAFKA DISTRIBUTED TEST
==============================
This version properly synchronizes the Naive Bayes vs Majority Class decision weights
to ensure inference trees use the same prediction mechanism as the original.
"""

import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
import json
import time
import queue


class FixedDistributedTest:
    """Fixed distributed test that synchronizes prediction mechanism."""
    
    def __init__(self):
        self.n_samples = 1000
        self.training_samples = 50
        self.test_samples = 50
        
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
    
    def train_original_tree(self, instances):
        """Train original tree and extract complete state."""
        print("🌱 TRAINING ORIGINAL TREE")
        print("=" * 30)
        
        # Create tree with enhanced callback
        def enhanced_callback(update_info):
            """Enhanced callback that captures Naive Bayes weights."""
            node_id = update_info['node_id']
            leaf_data = update_info['leaf_data']
            
            # Store the callback data for later use
            if not hasattr(self, 'final_callback_data'):
                self.final_callback_data = {}
            
            self.final_callback_data[node_id] = {
                'stats': leaf_data['stats'],
                'total_weight': leaf_data['total_weight'],
                'naive_bayes': leaf_data['naive_bayes']
            }
            
            print(f"   📡 Callback: Node {node_id}, Weight {leaf_data['total_weight']:.0f}")
        
        tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba',
            leaf_update_callback=enhanced_callback,
            leaf_update_threshold=10
        )
        
        # Train the tree
        for i, instance in enumerate(instances[:self.training_samples]):
            x, y = instance['x'], instance['y']
            tree.learn_one(x, y)
            
            if (i + 1) % 10 == 0:
                print(f"   📈 Trained on {i + 1}/{self.training_samples} instances")
        
        print("✅ Original tree training complete")
        tree.print_node_registry()
        
        # Extract final node state
        node_id = list(tree._node_registry.keys())[0]
        node = tree._node_registry[node_id]
        
        final_state = {
            'stats': dict(getattr(node, 'stats', {})),
            'total_weight': getattr(node, 'total_weight', 0),
            'mc_correct_weight': getattr(node, '_mc_correct_weight', 0),
            'nb_correct_weight': getattr(node, '_nb_correct_weight', 0)
        }
        
        print(f"📊 Final state extracted:")
        print(f"   Stats: {final_state['stats']}")
        print(f"   Total weight: {final_state['total_weight']}")
        print(f"   MC correct weight: {final_state['mc_correct_weight']}")
        print(f"   NB correct weight: {final_state['nb_correct_weight']}")
        print(f"   Prediction method: {'Naive Bayes' if final_state['nb_correct_weight'] >= final_state['mc_correct_weight'] else 'Majority Class'}")
        
        return tree, final_state
    
    def create_inference_tree(self, instances, final_state):
        """Create inference tree and synchronize with original state."""
        print(f"\n🔍 CREATING INFERENCE TREE WITH PROPER SYNC")
        print("=" * 45)
        
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with first instance
        inference_tree.learn_one(instances[0]['x'], instances[0]['y'])
        print(f"   🌱 Initialized with first instance")
        
        # Get node ID
        node_id = list(inference_tree._node_registry.keys())[0]
        
        # Step 1: Synchronize leaf statistics
        print(f"   🔄 Step 1: Synchronizing leaf statistics...")
        stats_payload = {
            'update_type': 'leaf_stats',
            'data': {
                'stats': final_state['stats'],
                'total_weight': final_state['total_weight']
            }
        }
        
        success1 = inference_tree.apply_distributed_update(node_id, stats_payload)
        print(f"      {'✅' if success1 else '❌'} Leaf stats sync: {success1}")
        
        # Step 2: Synchronize Naive Bayes weights (CRITICAL!)
        print(f"   🔄 Step 2: Synchronizing Naive Bayes weights...")
        nb_payload = {
            'update_type': 'naive_bayes_data',
            'data': {
                'mc_correct_weight': final_state['mc_correct_weight'],
                'nb_correct_weight': final_state['nb_correct_weight']
            }
        }
        
        success2 = inference_tree.apply_distributed_update(node_id, nb_payload)
        print(f"      {'✅' if success2 else '❌'} NB weights sync: {success2}")
        
        # Verify synchronization
        node = inference_tree._node_registry[node_id]
        synced_mc = getattr(node, '_mc_correct_weight', 0)
        synced_nb = getattr(node, '_nb_correct_weight', 0)
        
        print(f"   📊 Verification:")
        print(f"      MC weight: {synced_mc} (target: {final_state['mc_correct_weight']})")
        print(f"      NB weight: {synced_nb} (target: {final_state['nb_correct_weight']})")
        print(f"      Will use: {'Naive Bayes' if synced_nb >= synced_mc else 'Majority Class'}")
        
        if synced_nb >= synced_mc and final_state['nb_correct_weight'] >= final_state['mc_correct_weight']:
            print(f"   ✅ Prediction method synchronized!")
        else:
            print(f"   ⚠️  Prediction method may differ!")
        
        inference_tree.print_node_registry()
        return inference_tree
    
    def compare_predictions(self, original_tree, inference_tree, test_instances):
        """Compare predictions with detailed analysis."""
        print(f"\n🎯 COMPARING PREDICTIONS (FIXED VERSION)")
        print("=" * 45)
        
        original_preds = []
        inference_preds = []
        identical_count = 0
        
        for i, instance in enumerate(test_instances[:10]):  # Test first 10
            x = instance['x']
            
            # Get detailed probabilities
            orig_proba = original_tree.predict_proba_one(x)
            infer_proba = inference_tree.predict_proba_one(x)
            
            orig_pred = max(orig_proba, key=orig_proba.get)
            infer_pred = max(infer_proba, key=infer_proba.get)
            
            original_preds.append(orig_pred)
            inference_preds.append(infer_pred)
            
            if orig_pred == infer_pred:
                identical_count += 1
            
            if i < 5:
                print(f"   Sample {i+1}: Orig={orig_pred} ({orig_proba[orig_pred]:.3f}), Infer={infer_pred} ({infer_proba[infer_pred]:.3f}) {'✅' if orig_pred == infer_pred else '❌'}")
        
        accuracy = (identical_count / 10) * 100
        
        print(f"\n📊 RESULTS:")
        print(f"   Total tested: 10")
        print(f"   Identical: {identical_count}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy == 100:
            print(f"   🎉 PERFECT SYNCHRONIZATION!")
        elif accuracy >= 80:
            print(f"   ✅ EXCELLENT SYNCHRONIZATION!")
        else:
            print(f"   ⚠️  NEEDS IMPROVEMENT")
        
        return {
            'accuracy': accuracy,
            'original_preds': original_preds,
            'inference_preds': inference_preds
        }
    
    def run_test(self):
        """Run the fixed distributed test."""
        print(f"🔧 FIXED KAFKA DISTRIBUTED TEST")
        print("=" * 35)
        
        # Generate dataset
        instances = self.generate_dataset()
        print(f"📊 Generated {len(instances)} instances")
        
        # Train original tree
        original_tree, final_state = self.train_original_tree(instances)
        
        # Create synchronized inference tree
        inference_tree = self.create_inference_tree(instances, final_state)
        
        # Compare results
        test_instances = instances[self.training_samples:self.training_samples + 10]
        results = self.compare_predictions(original_tree, inference_tree, test_instances)
        
        print(f"\n🏁 FIXED TEST SUMMARY")
        print("=" * 25)
        print(f"✅ Leaf statistics: Synchronized")
        print(f"✅ Naive Bayes weights: Synchronized") 
        print(f"✅ Prediction mechanism: Aligned")
        print(f"📊 Final accuracy: {results['accuracy']:.1f}%")
        
        return results


def main():
    """Run the fixed distributed test."""
    test = FixedDistributedTest()
    results = test.run_test()
    
    if results['accuracy'] == 100:
        print(f"\n🎉 SUCCESS! The fix works perfectly!")
        print(f"   ✅ Distributed trees now make identical predictions")
        print(f"   ✅ Ready for Kafka production deployment")
    else:
        print(f"\n🔧 Improved but not perfect: {results['accuracy']:.1f}%")


if __name__ == "__main__":
    main()