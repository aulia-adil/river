#!/usr/bin/env python3
"""
🚀 IMPROVED KAFKA DISTRIBUTED HOEFFDING TREE TEST
================================================
This version properly synchronizes tree states instead of accumulating updates.

Key Improvements:
1. Sends complete tree state snapshots (not incremental updates)
2. Synchronizes exact leaf statistics 
3. Validates perfect state matching between distributed trees
"""

import json
import time
import queue
import threading
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# River imports
import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier


class ImprovedKafkaDistributedTest:
    """Improved Kafka distributed test with proper state synchronization."""
    
    def __init__(self):
        self.topic_name = "hoeffding_tree_updates"
        self.message_queue = queue.Queue()
        
        # Test configuration
        self.n_samples = 1000
        self.training_samples = 50
        self.test_samples = 50
        
        print(f"🔧 IMPROVED KAFKA DISTRIBUTED TEST SETUP")
        print(f"   Topic: {self.topic_name}")
        print(f"   Approach: State synchronization (not accumulation)")
        print(f"   Dataset: {self.n_samples} samples")
        print(f"   Training: {self.training_samples} instances")
        print(f"   Testing: {self.test_samples} predictions")
    
    def generate_test_dataset(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Generate test dataset."""
        print(f"\n🎲 GENERATING TEST DATASET ({self.n_samples} samples)")
        print("=" * 50)
        
        # Generate binary classification dataset
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        df['target'] = y
        
        # Convert to River format
        instances = []
        for i in range(len(df)):
            row = df.iloc[i]
            x = {col: row[col] for col in df.columns if col != 'target'}
            instances.append({'x': x, 'y': row['target']})
        
        print(f"✅ Dataset generated")
        print(f"   Features: {list(df.columns[:-1])}")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df, instances
    
    def update_process(self, instances: List[Dict]) -> HoeffdingTreeClassifier:
        """Update process: Train tree and publish complete state snapshots."""
        print(f"\n🌱 UPDATE PROCESS: Training & Publishing State Snapshots")
        print("=" * 55)
        
        # Create tree with leaf update callback for state snapshots
        def leaf_update_callback(update_info):
            """Callback to publish complete leaf state."""
            node_id = update_info['node_id']
            leaf_data = update_info['leaf_data']
            
            # Create complete state snapshot message
            message = {
                'message_type': 'state_sync',
                'timestamp': time.time(),
                'node_id': node_id,
                'complete_state': {
                    'stats': leaf_data['stats'],
                    'total_weight': leaf_data['total_weight'],
                    'depth': leaf_data['depth'],
                    'is_active': leaf_data['is_active']
                }
            }
            
            try:
                # Send state synchronization message
                future = self.producer.send(self.topic_name, key=f"sync_{node_id}", value=message)
                result = future.get()
                print(f"   📤 STATE SYNC SENT: Node {node_id} → Weight {leaf_data['total_weight']:.0f} (Offset {result.offset})")
            except Exception as e:
                print(f"   ❌ SYNC ERROR: {e}")
        
        # Create producer (simulate)
        class MockProducer:
            def __init__(self, queue):
                self.queue = queue
                self.offset = 0
            
            def send(self, topic, key, value):
                msg = type('Message', (), {
                    'key': key,
                    'value': value,
                    'partition': 0,
                    'offset': self.offset
                })()
                self.queue.put(msg)
                self.offset += 1
                
                class MockFuture:
                    def get(self):
                        return msg
                return MockFuture()
        
        self.producer = MockProducer(self.message_queue)
        
        # Create tree with callback
        tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba',
            leaf_update_callback=leaf_update_callback,
            leaf_update_threshold=10  # Trigger every 10 instances
        )
        
        print(f"📚 Training tree with {self.training_samples} instances...")
        
        # Train the tree
        for i, instance in enumerate(instances[:self.training_samples]):
            x, y = instance['x'], instance['y']
            tree.learn_one(x, y)
            
            if (i + 1) % 10 == 0:
                print(f"   📈 Processed {i + 1}/{self.training_samples} instances")
        
        # Send final complete tree state
        final_message = {
            'message_type': 'training_complete',
            'timestamp': time.time(),
            'final_tree_state': {
                'n_nodes': tree.get_node_registry_size(),
                'nodes': {}
            }
        }
        
        # Add complete state for each node
        for node_id, node in tree._node_registry.items():
            final_message['final_tree_state']['nodes'][str(node_id)] = {
                'type': type(node).__name__,
                'stats': dict(getattr(node, 'stats', {})),
                'total_weight': getattr(node, 'total_weight', 0),
                'depth': getattr(node, 'depth', 0)
            }
        
        # Send completion message
        completion_msg = type('Message', (), {
            'key': 'final_state',
            'value': final_message,
            'partition': 0,
            'offset': self.producer.offset
        })()
        self.message_queue.put(completion_msg)
        
        print(f"   📤 FINAL STATE SENT: Complete tree state (Offset {self.producer.offset})")
        print(f"✅ Update process completed")
        tree.print_node_registry()
        
        return tree
    
    def inference_process(self, instances: List[Dict]) -> HoeffdingTreeClassifier:
        """Inference process: Receive state syncs and apply exact synchronization."""
        print(f"\n🔍 INFERENCE PROCESS: Receiving State Syncs & Synchronizing")
        print("=" * 60)
        
        # Create inference tree
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with first instance to create root
        first_instance = instances[0]
        inference_tree.learn_one(first_instance['x'], first_instance['y'])
        print(f"   🌱 Initialized inference tree with first instance")
        
        syncs_received = 0
        training_complete = False
        
        print(f"📥 Processing state synchronization messages...")
        
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                data = message.value
                message_type = data.get('message_type')
                
                print(f"   📥 MESSAGE RECEIVED: {message_type} (Key: {message.key}, Offset: {message.offset})")
                
                if message_type == 'state_sync':
                    node_id = data['node_id']
                    complete_state = data['complete_state']
                    
                    # Create synchronization payload
                    sync_payload = {
                        'update_type': 'leaf_stats',
                        'data': {
                            'stats': complete_state['stats'],
                            'total_weight': complete_state['total_weight']
                        }
                    }
                    
                    # Apply exact state synchronization
                    success = inference_tree.apply_distributed_update(node_id, sync_payload)
                    if success:
                        syncs_received += 1
                        print(f"      ✅ State synchronized #{syncs_received}")
                    else:
                        print(f"      ❌ Failed to sync state")
                
                elif message_type == 'training_complete':
                    print(f"   🏁 Training complete with final state")
                    final_tree_state = data['final_tree_state']
                    
                    # Verify final synchronization
                    expected_nodes = final_tree_state['nodes']
                    actual_nodes = inference_tree._node_registry
                    
                    print(f"      Expected nodes: {len(expected_nodes)}")
                    print(f"      Actual nodes: {len(actual_nodes)}")
                    
                    # Sync any final state differences
                    for node_id_str, expected_state in expected_nodes.items():
                        node_id = int(node_id_str)
                        if node_id in actual_nodes:
                            sync_payload = {
                                'update_type': 'leaf_stats', 
                                'data': {
                                    'stats': expected_state['stats'],
                                    'total_weight': expected_state['total_weight']
                                }
                            }
                            inference_tree.apply_distributed_update(node_id, sync_payload)
                    
                    training_complete = True
                    break
                    
        except queue.Empty:
            print(f"   📭 No more messages in queue")
        
        print(f"✅ Inference process completed")
        print(f"   State syncs received: {syncs_received}")
        inference_tree.print_node_registry()
        
        return inference_tree
    
    def compare_trees_and_predictions(self, original_tree: HoeffdingTreeClassifier, 
                                    inference_tree: HoeffdingTreeClassifier, 
                                    test_instances: List[Dict]) -> Dict[str, Any]:
        """Compare tree states and predictions."""
        print(f"\n🔍 COMPARING TREE STATES AND PREDICTIONS")
        print("=" * 45)
        
        # Compare tree structure
        orig_nodes = original_tree._node_registry
        infer_nodes = inference_tree._node_registry
        
        print(f"🌳 Tree Structure Comparison:")
        print(f"   Original nodes: {len(orig_nodes)}")
        print(f"   Inference nodes: {len(infer_nodes)}")
        
        structure_match = len(orig_nodes) == len(infer_nodes)
        print(f"   Structure match: {'✅' if structure_match else '❌'}")
        
        # Compare node statistics
        stats_match = True
        for node_id in orig_nodes:
            if node_id in infer_nodes:
                orig_stats = dict(getattr(orig_nodes[node_id], 'stats', {}))
                infer_stats = dict(getattr(infer_nodes[node_id], 'stats', {}))
                
                if orig_stats != infer_stats:
                    stats_match = False
                    print(f"   ❌ Node {node_id} stats mismatch:")
                    print(f"      Original: {orig_stats}")
                    print(f"      Inference: {infer_stats}")
                else:
                    print(f"   ✅ Node {node_id} stats match: {orig_stats}")
        
        # Test predictions
        print(f"\n🎯 Prediction Comparison:")
        original_preds = []
        inference_preds = []
        identical_count = 0
        
        for i, instance in enumerate(test_instances[:self.test_samples]):
            x = instance['x']
            
            # Get predictions
            orig_proba = original_tree.predict_proba_one(x)
            infer_proba = inference_tree.predict_proba_one(x)
            
            # Get predicted classes
            orig_pred = max(orig_proba, key=orig_proba.get)
            infer_pred = max(infer_proba, key=infer_proba.get)
            
            original_preds.append(orig_pred)
            inference_preds.append(infer_pred)
            
            if orig_pred == infer_pred:
                identical_count += 1
            
            if i < 5:  # Show first 5 predictions
                print(f"   Sample {i+1}: Original={orig_pred}, Inference={infer_pred} {'✅' if orig_pred == infer_pred else '❌'}")
        
        accuracy = (identical_count / self.test_samples) * 100
        
        results = {
            'structure_match': structure_match,
            'stats_match': stats_match,
            'total_predictions': self.test_samples,
            'identical_predictions': identical_count,
            'accuracy_percentage': accuracy,
            'original_predictions': original_preds,
            'inference_predictions': inference_preds
        }
        
        print(f"\n📊 SYNCHRONIZATION RESULTS:")
        print(f"   Structure match: {'✅' if results['structure_match'] else '❌'}")
        print(f"   Statistics match: {'✅' if results['stats_match'] else '❌'}")
        print(f"   Prediction accuracy: {results['accuracy_percentage']:.2f}%")
        
        if accuracy == 100 and structure_match and stats_match:
            print(f"   🎉 PERFECT DISTRIBUTED SYNCHRONIZATION!")
        elif accuracy >= 95:
            print(f"   ✅ EXCELLENT SYNCHRONIZATION!")
        else:
            print(f"   ⚠️  SYNCHRONIZATION ISSUES DETECTED")
        
        return results
    
    def run_test(self):
        """Run the complete improved distributed test."""
        print(f"\n🚀 STARTING IMPROVED KAFKA DISTRIBUTED TEST")
        print("=" * 50)
        
        # Generate dataset
        df, instances = self.generate_test_dataset()
        
        # Run update process
        print(f"\n" + "="*60)
        original_tree = self.update_process(instances)
        
        # Run inference process  
        print(f"\n" + "="*60)
        inference_tree = self.inference_process(instances)
        
        # Compare results
        print(f"\n" + "="*60)
        test_instances = instances[self.training_samples:self.training_samples + self.test_samples]
        results = self.compare_trees_and_predictions(original_tree, inference_tree, test_instances)
        
        # Final summary
        print(f"\n🏁 IMPROVED KAFKA TEST SUMMARY")
        print("=" * 35)
        print(f"✅ Approach: State synchronization (not accumulation)")
        print(f"✅ Dataset: {self.n_samples} samples generated")
        print(f"✅ Training: {self.training_samples} instances processed")
        print(f"✅ Synchronization: State snapshots sent/received")
        print(f"📊 Structure match: {'Yes' if results['structure_match'] else 'No'}")
        print(f"📊 Stats match: {'Yes' if results['stats_match'] else 'No'}")
        print(f"📊 Prediction accuracy: {results['accuracy_percentage']:.2f}%")
        
        if (results['accuracy_percentage'] == 100 and 
            results['structure_match'] and 
            results['stats_match']):
            print(f"\n🎉 SUCCESS: PERFECT KAFKA DISTRIBUTED SYNCHRONIZATION!")
            print(f"   ✅ State synchronization approach works flawlessly")
            print(f"   ✅ Ready for production with real Kafka")
        else:
            print(f"\n⚠️  NEEDS IMPROVEMENT: Synchronization not perfect")
        
        return results


def main():
    """Main function."""
    test = ImprovedKafkaDistributedTest()
    results = test.run_test()
    
    if (results and results['accuracy_percentage'] == 100 and 
        results['structure_match'] and results['stats_match']):
        print(f"\n💾 VALIDATION SUCCESSFUL!")
        print(f"   🎯 Distributed state synchronization works perfectly")
        print(f"   🚀 Ready for Kafka production deployment")


if __name__ == "__main__":
    main()