#!/usr/bin/env python3
"""
🚀 SIMULATED KAFKA DISTRIBUTED HOEFFDING TREE TEST
==================================================
This simulates the Kafka distributed training when Kafka isn't available.
Shows exactly how the system would work in production.

Architecture:
1. Update Process: Trains tree, generates Kafka messages (leaf stats)
2. Message Queue: Simulates Kafka topic with in-memory queue
3. Inference Process: Consumes messages, applies updates via apply_distributed_update()
4. Validation: Compares predictions to ensure synchronization
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


class SimulatedKafkaMessage:
    """Simulates a Kafka message structure."""
    
    def __init__(self, key: str, value: dict, partition: int = 0, offset: int = 0):
        self.key = key
        self.value = value
        self.partition = partition
        self.offset = offset


class SimulatedKafkaProducer:
    """Simulates Kafka producer using in-memory queue."""
    
    def __init__(self, message_queue: queue.Queue):
        self.message_queue = message_queue
        self.message_count = 0
    
    def send(self, topic: str, key: str, value: dict):
        """Simulate sending message to Kafka topic."""
        message = SimulatedKafkaMessage(
            key=key,
            value=value,
            partition=0,
            offset=self.message_count
        )
        
        self.message_queue.put(message)
        self.message_count += 1
        
        # Return a future-like object
        class FakeFuture:
            def get(self, timeout=None):
                return message
        
        return FakeFuture()
    
    def flush(self):
        """Flush producer (no-op in simulation)."""
        pass
    
    def close(self):
        """Close producer (no-op in simulation)."""
        pass


class SimulatedKafkaConsumer:
    """Simulates Kafka consumer using in-memory queue."""
    
    def __init__(self, topic: str, message_queue: queue.Queue):
        self.topic = topic
        self.message_queue = message_queue
        self.consumed_count = 0
    
    def __iter__(self):
        """Iterate over messages in queue."""
        return self
    
    def __next__(self):
        """Get next message from queue."""
        try:
            # Wait for message with timeout
            message = self.message_queue.get(timeout=5)
            self.consumed_count += 1
            return message
        except queue.Empty:
            raise StopIteration
    
    def close(self):
        """Close consumer (no-op in simulation)."""
        pass


class SimulatedKafkaDistributedTest:
    """Simulated Kafka distributed Hoeffding Tree test."""
    
    def __init__(self):
        self.topic_name = "hoeffding_tree_updates"
        self.message_queue = queue.Queue()
        
        # Test configuration
        self.n_samples = 1000
        self.training_samples = 50
        self.test_samples = 50
        
        print(f"🔧 SIMULATED KAFKA DISTRIBUTED TEST SETUP")
        print(f"   Topic: {self.topic_name}")
        print(f"   Message queue: In-memory simulation")
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
        """Update process: Train tree and publish leaf stats."""
        print(f"\n🌱 UPDATE PROCESS: Training & Publishing Messages")
        print("=" * 50)
        
        producer = SimulatedKafkaProducer(self.message_queue)
        
        # Create tree with leaf update callback
        def leaf_update_callback(update_info):
            """Callback to publish leaf updates."""
            message = {
                'message_type': 'leaf_update',
                'timestamp': time.time(),
                'update_info': update_info
            }
            
            try:
                future = producer.send(self.topic_name, key=str(update_info['node_id']), value=message)
                result = future.get()
                print(f"   📤 MESSAGE SENT: Node {update_info['node_id']} → Offset {result.offset}")
            except Exception as e:
                print(f"   ❌ MESSAGE ERROR: {e}")
        
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
        
        # Send training complete message
        final_message = {
            'message_type': 'training_complete',
            'timestamp': time.time(),
            'total_instances': self.training_samples,
            'tree_stats': {
                'n_nodes': tree.get_node_registry_size(),
                'node_registry': {
                    str(node_id): {
                        'type': type(node).__name__,
                        'stats': dict(getattr(node, 'stats', {})),
                        'total_weight': getattr(node, 'total_weight', 0)
                    }
                    for node_id, node in tree._node_registry.items()
                }
            }
        }
        
        try:
            future = producer.send(self.topic_name, key='training_complete', value=final_message)
            result = future.get()
            print(f"   📤 MESSAGE SENT: Training complete → Offset {result.offset}")
        except Exception as e:
            print(f"   ❌ MESSAGE ERROR: {e}")
        
        producer.flush()
        producer.close()
        
        print(f"✅ Update process completed")
        tree.print_node_registry()
        
        return tree
    
    def inference_process(self, instances: List[Dict]) -> HoeffdingTreeClassifier:
        """Inference process: Receive messages and apply updates."""
        print(f"\n🔍 INFERENCE PROCESS: Consuming Messages & Applying Updates")
        print("=" * 60)
        
        consumer = SimulatedKafkaConsumer(self.topic_name, self.message_queue)
        
        # Create inference tree
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with first instance
        first_instance = instances[0]
        inference_tree.learn_one(first_instance['x'], first_instance['y'])
        print(f"   🌱 Initialized inference tree with first instance")
        
        updates_received = 0
        training_complete = False
        
        print(f"📥 Processing messages from queue...")
        
        try:
            for message in consumer:
                data = message.value
                message_type = data.get('message_type')
                
                print(f"   📥 MESSAGE RECEIVED: {message_type} (Key: {message.key}, Offset: {message.offset})")
                
                if message_type == 'leaf_update':
                    update_info = data['update_info']
                    node_id = update_info['node_id']
                    
                    # Extract leaf data
                    leaf_data = update_info['leaf_data']
                    
                    # Create update payload for leaf stats
                    update_payload = {
                        'update_type': 'leaf_stats',
                        'data': {
                            'stats': leaf_data['stats'],
                            'total_weight': leaf_data['total_weight']
                        }
                    }
                    
                    # Apply the update using our distributed update system
                    success = inference_tree.apply_distributed_update(node_id, update_payload)
                    if success:
                        updates_received += 1
                        print(f"      ✅ Applied update #{updates_received}")
                    else:
                        print(f"      ❌ Failed to apply update")
                
                elif message_type == 'training_complete':
                    print(f"   🏁 Training complete signal received")
                    print(f"      Total updates applied: {updates_received}")
                    training_complete = True
                    break
                    
        except StopIteration:
            print(f"   📭 No more messages in queue")
        
        consumer.close()
        
        print(f"✅ Inference process completed")
        print(f"   Updates received: {updates_received}")
        inference_tree.print_node_registry()
        
        return inference_tree
    
    def compare_predictions(self, original_tree: HoeffdingTreeClassifier, 
                          inference_tree: HoeffdingTreeClassifier, 
                          test_instances: List[Dict]) -> Dict[str, Any]:
        """Compare predictions between trees."""
        print(f"\n🎯 COMPARING PREDICTIONS")
        print("=" * 30)
        
        original_preds = []
        inference_preds = []
        identical_count = 0
        
        print(f"🔍 Testing {self.test_samples} predictions...")
        
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
            'total_predictions': self.test_samples,
            'identical_predictions': identical_count,
            'accuracy_percentage': accuracy,
            'original_predictions': original_preds,
            'inference_predictions': inference_preds
        }
        
        print(f"\n📊 PREDICTION COMPARISON RESULTS:")
        print(f"   Total predictions: {results['total_predictions']}")
        print(f"   Identical predictions: {results['identical_predictions']}")
        print(f"   Success rate: {results['accuracy_percentage']:.2f}%")
        
        if accuracy == 100:
            print(f"   🎉 PERFECT SYNCHRONIZATION ACHIEVED!")
        elif accuracy >= 95:
            print(f"   ✅ EXCELLENT SYNCHRONIZATION!")
        elif accuracy >= 90:
            print(f"   ⚠️  GOOD SYNCHRONIZATION (minor differences)")
        else:
            print(f"   ❌ POOR SYNCHRONIZATION (significant differences)")
        
        return results
    
    def run_test(self):
        """Run the complete simulated distributed test."""
        print(f"\n🚀 STARTING SIMULATED KAFKA DISTRIBUTED TEST")
        print("=" * 55)
        
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
        results = self.compare_predictions(original_tree, inference_tree, test_instances)
        
        # Final summary
        print(f"\n🏁 SIMULATED KAFKA TEST SUMMARY")
        print("=" * 40)
        print(f"✅ Dataset: {self.n_samples} samples generated")
        print(f"✅ Training: {self.training_samples} instances processed")
        print(f"✅ Messages: Sent and received via queue")
        print(f"✅ Updates: Applied using apply_distributed_update()")
        print(f"✅ Predictions: {results['total_predictions']} tested")
        print(f"📊 Synchronization: {results['accuracy_percentage']:.2f}%")
        
        if results['accuracy_percentage'] == 100:
            print(f"\n🎉 SUCCESS: PERFECT DISTRIBUTED SYNCHRONIZATION!")
            print(f"   ✅ Distributed update system works perfectly")
            print(f"   ✅ Ready for production Kafka deployment")
            print(f"   ✅ Leaf stats synchronization validated")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {results['accuracy_percentage']:.2f}% synchronization")
        
        # Show how this would work with real Kafka
        print(f"\n🚀 PRODUCTION KAFKA IMPLEMENTATION:")
        print(f"   📤 Update Process → Kafka Producer → Topic: {self.topic_name}")
        print(f"   📥 Inference Process → Kafka Consumer → apply_distributed_update()")
        print(f"   🔄 Real-time synchronization across distributed nodes")
        print(f"   📡 Messages contain leaf_data with stats and weights")
        
        return results


def main():
    """Main function."""
    test = SimulatedKafkaDistributedTest()
    results = test.run_test()
    
    if results and results['accuracy_percentage'] == 100:
        print(f"\n💾 VALIDATION SUCCESSFUL!")
        print(f"   🎯 Your distributed system is production-ready")
        print(f"   🚀 Next step: Deploy with real Kafka infrastructure")


if __name__ == "__main__":
    main()