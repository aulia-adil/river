#!/usr/bin/env python3
"""
🎯 KAFKA INFERENCE PROCESS - Distributed Hoeffding Tree
=======================================================
Inference process that consumes leaf updates from Kafka.
Run this after starting the training process to sync the tree.
"""

import sys
import os
import time
import json
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np


class KafkaConfig:
    """Kafka configuration."""
    BOOTSTRAP_SERVERS = ['kafka_service:9092']
    TOPIC_LEAF_UPDATES = 'hoeffding_leaf_updates'
    
    @staticmethod
    def create_consumer(group_id='inference-consumer'):
        """Create Kafka consumer with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                consumer = KafkaConsumer(
                    KafkaConfig.TOPIC_LEAF_UPDATES,
                    bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='earliest',
                    group_id=group_id,
                    enable_auto_commit=True
                )
                print(f"✅ Kafka consumer connected to {KafkaConfig.BOOTSTRAP_SERVERS}")
                return consumer
            except NoBrokersAvailable:
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting for Kafka broker (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    raise


class InferenceProcess:
    """Inference process that consumes updates from Kafka."""
    
    def __init__(self, instances):
        self.instances = instances
        self.tree = None
        self.consumer = None
        self.updates_received = 0
        self.running = False
        self.last_applied_iteration = 0  # Track last applied iteration for ordering
        
    def initialize_tree(self):
        """Initialize inference tree with both classes."""
        print("\n🎯 INFERENCE PROCESS STARTED")
        print("=" * 50)
        
        self.tree = HoeffdingTreeClassifier(grace_period=200, leaf_prediction='nba')
        
        print("🌱 Initializing tree with both classes...")
        
        # Find instances of both classes
        class_instances = {0: None, 1: None}
        for instance in self.instances[:20]:
            if instance['y'] in class_instances and class_instances[instance['y']] is None:
                class_instances[instance['y']] = instance
        
        # Learn from both classes
        for class_label, instance in class_instances.items():
            if instance:
                self.tree.learn_one(instance['x'], instance['y'])
                print(f"   ✅ Initialized with class {class_label}")
        
        print("✅ Inference tree initialized\n")
    
    def consume_and_update(self, expected_updates=10, timeout=30):
        """Consume updates from Kafka and apply to tree."""
        print("📥 Starting Kafka consumer...")
        self.consumer = KafkaConfig.create_consumer()
        self.running = True
        
        print(f"🔄 Waiting for {expected_updates} updates (timeout: {timeout}s)...\n")
        
        start_time = time.time()
        
        try:
            for message in self.consumer:
                if not self.running:
                    break
                
                update_data = message.value
                iteration = update_data.get('iteration', 0)
                 
                # Only apply if this is the next expected iteration
                if iteration != self.last_applied_iteration + 1:
                    print(f"⏭️  Skipping out-of-order update (iteration {iteration}, expected {self.last_applied_iteration + 1})")
                    continue
                
                self.updates_received += 1
                
                print(f"📨 Received update #{self.updates_received}")
                print(f"   Iteration: {iteration}")
                print(f"   Node ID: {update_data.get('node_id')}")
                print(f"   Stats: {update_data.get('stats')}")
                print(f"   Weights: MC={update_data.get('mc_correct_weight')}, NB={update_data.get('nb_correct_weight')}")
                
                # Apply update to tree
                self._apply_update(update_data)
                self.last_applied_iteration = iteration
                
                # Check if we received all expected updates
                if self.updates_received >= expected_updates:
                    print(f"\n✅ Received all {expected_updates} updates!")
                    break
                
                # Check timeout
                if time.time() - start_time > timeout:
                    print(f"\n⏱️ Timeout reached ({timeout}s)")
                    break
                
        except KeyboardInterrupt:
            print("\n⚠️ Consumer interrupted")
        finally:
            self.consumer.close()
            print(f"✅ Consumer closed. Total updates received: {self.updates_received}")
    
    def _apply_update(self, update_data):
        """Apply Kafka update to inference tree."""
        node_id = update_data.get('node_id', 0)
        att_dist_per_class = update_data.get('_att_dist_per_class', {})
        stats_raw = update_data.get('stats', {})
        total_weight = update_data.get('total_weight', 0)
        mc_correct_weight = update_data.get('mc_correct_weight', 0)
        nb_correct_weight = update_data.get('nb_correct_weight', 0)
        
        # Convert stats keys from strings back to floats (JSON converts them to strings)
        stats = {float(k): v for k, v in stats_raw.items()}
        
        # Build the update payload
        update_payload = {
            'node_id': node_id,
            'update_type': 'complete_node',
            'timestamp': update_data.get('timestamp'),
            'data': {
                'leaf_stats': {
                    'stats': stats,
                    'total_weight': total_weight
                },
                'splitter_data': {
                    'splitters': {}
                },
                'naive_bayes_data': {
                    'mc_correct_weight': mc_correct_weight,
                    'nb_correct_weight': nb_correct_weight
                }
            }
        }
        
        # Convert distributions format
        for feature_name, class_distributions in att_dist_per_class.items():
            distributions_dict = {}
            for class_label, dist_params in class_distributions.items():
                distributions_dict[class_label] = {
                    'n_samples': dist_params.get('n_samples'),
                    'mu': dist_params.get('mu'),
                    'sigma': dist_params.get('sigma')
                }
            
            update_payload['data']['splitter_data']['splitters'][feature_name] = {
                'type': 'GaussianSplitter',
                'feature_name': feature_name,
                'gaussian_data': {
                    'distributions': distributions_dict
                }
            }
        
        # Apply the update
        if node_id in self.tree._node_registry:
            success = self.tree.apply_distributed_update(node_id, update_payload)
            print(f"   {'✅' if success else '❌'} Update applied to inference tree")
        else:
            print(f"   ⚠️ Node {node_id} not found in registry")
    
    def show_tree_state(self):
        """Display current tree state."""
        print("\n📊 INFERENCE TREE STATE:")
        print("=" * 50)
        
        if len(self.tree._node_registry) > 0:
            node_id = list(self.tree._node_registry.keys())[0]
            node = self.tree._node_registry[node_id]
            stats = dict(getattr(node, 'stats', {}))
            mc_weight = getattr(node, '_mc_correct_weight', 0)
            nb_weight = getattr(node, '_nb_correct_weight', 0)
            print(f"   Node {node_id}: Stats={stats}, MC={mc_weight}, NB={nb_weight}")
        else:
            print("   No nodes in tree")
        print("=" * 50)
    
    def test_predictions(self, test_instances):
        """Test predictions on sample instances."""
        print("\n🎯 Testing Predictions:")
        print("=" * 50)
        
        for i, instance in enumerate(test_instances):
            x = instance['x']
            y_true = instance['y']
            
            proba = self.tree.predict_proba_one(x)
            pred = max(proba, key=proba.get)
            conf = proba.get(pred, 0)
            
            correct = "✅" if pred == y_true else "❌"
            print(f"   Sample {i+1} {correct}: Predicted={pred}({conf:.3f}), True={y_true}")
        
        print("=" * 50)


def generate_dataset():
    """Generate test dataset (same as training)."""
    print("📊 Generating dataset...")
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
    
    print(f"✅ Generated {len(instances)} instances")
    return instances


def main():
    """Main entry point."""
    print("🎯 KAFKA INFERENCE PROCESS - DISTRIBUTED HOEFFDING TREE")
    print("=" * 60)
    print(f"📡 Kafka Broker: kafka_service:9092")
    print(f"📢 Topic: hoeffding_leaf_updates")
    print("=" * 60)
    
    # Generate dataset (same as training process)
    instances = generate_dataset()
    
    # Create inference process
    inference = InferenceProcess(instances)
    
    # Initialize tree
    inference.initialize_tree()
    
    # Consume updates from Kafka
    inference.consume_and_update(expected_updates=10, timeout=30)
    
    # Show final state
    inference.show_tree_state()
    
    # Test predictions
    test_instances = instances[10:15]
    inference.test_predictions(test_instances)
    
    print("\n🎯 Inference process complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
