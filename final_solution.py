#!/usr/bin/env python3
"""
🎯 FINAL SOLUTION - PROPER API USAGE
===================================
Use the existing distributed update API properly to achieve perfect synchronization.
"""

import sys
import os
import time
import json
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


class FinalSolution:
    """Final solution using proper distributed update API."""
    
    def __init__(self):
        self.n_samples = 1000
        self.training_samples = 10
        self.callback_iteration = 0
        self.json_output_dir = "test_json_file"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.json_output_dir, exist_ok=True)
        
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
    
    def train_and_extract_payload(self, instances):
        """Train original tree and extract complete payload for distribution."""
        print("🌱 TRAINING ORIGINAL TREE")
        print("=" * 30)

        def _leaf_update_callback(update_info):
            print("YOLO THE KID")
            print(update_info)
            """Callback to save only _att_dist_per_class to JSON."""
            self.callback_iteration += 1
            
            # Extract node information
            node_id = update_info.get('node_id')
            leaf_data = update_info.get('leaf_data', {})
            
            print(f"\n📡 Callback #{self.callback_iteration}: Node {node_id}")
            
            # Extract only _att_dist_per_class from splitters
            att_dist_data = {}
            
            if 'splitters' in leaf_data:
                for feature_name, splitter_info in leaf_data['splitters'].items():
                    if 'gaussian_data' in splitter_info:
                        gaussian_data = splitter_info['gaussian_data']
                        if 'distributions' in gaussian_data:
                            att_dist_data[feature_name] = gaussian_data['distributions']
                            
                            # Show what we're saving
                            print(f"   Feature '{feature_name}':")
                            for class_label, dist_params in gaussian_data['distributions'].items():
                                mu = dist_params.get('mu', 0)
                                sigma = dist_params.get('sigma', 0)
                                n = dist_params.get('n_samples', 0)
                                print(f"      Class {class_label}: μ={mu:.3f}, σ={sigma:.3f}, n={n}")
            
            # Create JSON payload with only _att_dist_per_class
            json_payload = {
                'callback_iteration': self.callback_iteration,
                'node_id': node_id,
                'timestamp': time.time(),
                '_att_dist_per_class': att_dist_data
            }
            
            # Save to JSON file
            json_filename = os.path.join(
                self.json_output_dir, 
                f"iteration_{self.callback_iteration}.json"
            )
            
            with open(json_filename, 'w') as f:
                json.dump(json_payload, f, indent=2)
            
            print(f"   ✅ Saved to: {json_filename}")

        tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba',
            leaf_update_threshold=1,
            leaf_update_callback=_leaf_update_callback
        )
        
        # Train the tree
        for i, instance in enumerate(instances[:self.training_samples]):
            # print(f"instance[x]: {instance['x']}")
            # print(f"instance[y]: {instance['y']}")
            tree.learn_one(instance['x'], instance['y'])
            
            if (i + 1) % 10 == 0:
                print(f"   📈 Trained on {i + 1}/{self.training_samples} instances")
        
        print("✅ Original tree training complete")
        
        # Extract complete update payload using the tree's own method
        node_id = list(tree._node_registry.keys())[0]
        
        print(f"📦 Creating complete update payload...")
        complete_payload = tree.create_update_payload(node_id, 'complete_node')
        
        print(f"   📊 Payload contents:")
        print(complete_payload)
        # print(f"      Update type: {complete_payload.get('update_type')}")
        # print(f"      Data keys: {list(complete_payload.get('data', {}).keys())}")
        
        # if 'data' in complete_payload and 'stats' in complete_payload['data']:
        #     stats = complete_payload['data']['stats']
        #     print(f"      Stats: {stats}")
        
        return tree, complete_payload
    
    def create_inference_tree_from_payload(self, instances, complete_payload):
        """Create inference tree using the complete payload."""
        print(f"\n🎯 CREATING INFERENCE TREE FROM PAYLOAD")
        print("=" * 45)
        print(complete_payload)
        
        # Create inference tree
        inference_tree = HoeffdingTreeClassifier(grace_period=200, leaf_prediction='nba')
        
        # Initialize with one instance from each class to ensure splitters exist
        print(f"   🌱 Smart initialization...")
        
        # Find instances of both classes
        class_instances = {0: None, 1: None}
        for instance in instances[:20]:
            if instance['y'] in class_instances and class_instances[instance['y']] is None:
                class_instances[instance['y']] = instance
        
        # Learn from both classes
        for class_label, instance in class_instances.items():
            if instance:
                inference_tree.learn_one(instance['x'], instance['y'])
                print(f"      ✅ Initialized with class {class_label}")
        
        # Apply the complete payload
        node_id = list(inference_tree._node_registry.keys())[0]
        
        print(f"   📦 Applying complete payload...")
        success = inference_tree.apply_distributed_update(node_id, complete_payload)
        print(f"      {'✅' if success else '❌'} Payload application: {success}")
        
        # Verify synchronization
        node = inference_tree._node_registry[node_id]
        final_stats = dict(getattr(node, 'stats', {}))
        final_weight = getattr(node, 'total_weight', 0)
        final_mc = getattr(node, '_mc_correct_weight', 0)
        final_nb = getattr(node, '_nb_correct_weight', 0)
        
        print(f"   📊 Final state verification:")
        print(f"      Stats: {final_stats}")
        print(f"      Total weight: {final_weight}")
        print(f"      MC weight: {final_mc}")
        print(f"      NB weight: {final_nb}")
        print(f"      Prediction method: {'Naive Bayes' if final_nb >= final_mc else 'Majority Class'}")
        
        return inference_tree
    
    def compare_trees_comprehensively(self, original_tree, inference_tree, test_instances):
        """Comprehensive comparison of trees."""
        print(f"\n🔍 COMPREHENSIVE TREE COMPARISON")
        print("=" * 40)
        
        # Compare node structures
        orig_nodes = list(original_tree._node_registry.keys())
        infer_nodes = list(inference_tree._node_registry.keys())
        
        print(f"   📋 Structure comparison:")
        print(f"      Original nodes: {len(orig_nodes)}")
        print(f"      Inference nodes: {len(infer_nodes)}")
        print(f"      Structure match: {orig_nodes == infer_nodes}")
        
        if orig_nodes == infer_nodes:
            node_id = orig_nodes[0]
            orig_node = original_tree._node_registry[node_id]
            infer_node = inference_tree._node_registry[node_id]
            
            # Compare stats
            orig_stats = dict(getattr(orig_node, 'stats', {}))
            infer_stats = dict(getattr(infer_node, 'stats', {}))
            
            print(f"   📊 Statistics comparison:")
            print(f"      Original stats: {orig_stats}")
            print(f"      Inference stats: {infer_stats}")
            print(f"      Stats match: {orig_stats == infer_stats}")
            
            # Compare weights
            orig_mc = getattr(orig_node, '_mc_correct_weight', 0)
            orig_nb = getattr(orig_node, '_nb_correct_weight', 0)
            infer_mc = getattr(infer_node, '_mc_correct_weight', 0)
            infer_nb = getattr(infer_node, '_nb_correct_weight', 0)
            
            print(f"   ⚖️  Weight comparison:")
            print(f"      Original MC/NB: {orig_mc}/{orig_nb}")
            print(f"      Inference MC/NB: {infer_mc}/{infer_nb}")
            print(f"      Weights match: {orig_mc == infer_mc and orig_nb == infer_nb}")
        
        # Compare predictions
        print(f"\n🎯 Prediction comparison:")
        perfect_matches = 0
        
        for i, instance in enumerate(test_instances[:5]):
            x = instance['x']
            
            orig_proba = original_tree.predict_proba_one(x)
            infer_proba = inference_tree.predict_proba_one(x)
            
            orig_pred = max(orig_proba, key=orig_proba.get)
            infer_pred = max(infer_proba, key=infer_proba.get)
            
            # Calculate probability difference
            orig_conf = orig_proba.get(orig_pred, 0)
            infer_conf = infer_proba.get(infer_pred, 0)
            prob_diff = abs(orig_conf - infer_conf)
            
            match = orig_pred == infer_pred
            if match and prob_diff < 0.01:
                perfect_matches += 1
                status = "🎯"
            elif match:
                status = "✅"
            else:
                status = "❌"
            
            print(f"      Sample {i+1} {status}: {orig_pred}({orig_conf:.3f}) vs {infer_pred}({infer_conf:.3f})")
        
        accuracy = (perfect_matches / 5) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"   Perfect matches: {perfect_matches}/5")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy == 100:
            print(f"   🏆 PERFECT SYNCHRONIZATION ACHIEVED!")
            print(f"   🎉 Distributed Hoeffding Tree is READY!")
        
        return accuracy
    
    def run_final_test(self):
        """Run the final comprehensive test."""
        print(f"🏆 FINAL DISTRIBUTED HOEFFDING TREE TEST")
        print("=" * 45)
        
        # Generate dataset
        instances = self.generate_dataset()
        print(f"📊 Generated {len(instances)} instances\n")
        
        # Train original and extract payload
        original_tree, complete_payload = self.train_and_extract_payload(instances)
        
        # Create inference tree from payload
        inference_tree = self.create_inference_tree_from_payload(instances, complete_payload)
        
        # Comprehensive comparison
        test_instances = instances[self.training_samples:self.training_samples + 5]
        accuracy = self.compare_trees_comprehensively(original_tree, inference_tree, test_instances)
        
        print(f"\n🏁 FINAL TEST SUMMARY")
        print("=" * 25)
        
        if accuracy == 100:
            print(f"🎉🎉🎉 MISSION ACCOMPLISHED! 🎉🎉🎉")
            print(f"✅ Perfect distributed synchronization achieved")
            print(f"✅ Both trees make identical predictions")
            print(f"✅ Ready for Kafka production deployment")
            print(f"🚀 Distributed Hoeffding Tree: SUCCESS!")
        else:
            print(f"🔧 Final accuracy: {accuracy:.1f}%")
            print(f"📋 Additional investigation needed")
        
        return accuracy


def main():
    """Run the final solution test."""
    test = FinalSolution()
    accuracy = test.run_final_test()


if __name__ == "__main__":
    main()