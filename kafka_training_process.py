#!/usr/bin/env python3
"""
🚀 KAFKA TRAINING PROCESS - Distributed Hoeffding Tree
======================================================
Training process that publishes leaf updates to Kafka.
Run this first to start training and publishing updates.
"""

import sys
import os
import time
import json
from kafka import KafkaProducer
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
    def create_producer():
        """Create Kafka producer with retry logic."""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=KafkaConfig.BOOTSTRAP_SERVERS,
                    key_serializer=lambda k: k.encode('utf-8') if isinstance(k, str) else k,
                    value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                    acks='all',
                    retries=3
                )
                print(f"✅ Kafka producer connected to {KafkaConfig.BOOTSTRAP_SERVERS}")
                return producer
            except NoBrokersAvailable:
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting for Kafka broker (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(2)
                else:
                    raise


class TrainingProcess:
    """Training process that publishes updates to Kafka."""
    
    def __init__(self, n_training_samples=10):
        self.n_samples = 1000
        self.training_samples = n_training_samples
        self.callback_iteration = 0
        self.producer = None
        
    def generate_dataset(self):
        """Generate test dataset."""
        print("📊 Generating dataset...")
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
        
        print(f"✅ Generated {len(instances)} instances")
        return instances
    
    def train(self, instances):
        """Train tree and publish updates to Kafka."""
        print("\n🌱 TRAINING PROCESS STARTED")
        print("=" * 50)
        
        # Initialize Kafka producer
        self.producer = KafkaConfig.create_producer()
        
        def _leaf_update_callback(update_info):
            """Callback to publish updates to Kafka."""
            self.callback_iteration += 1
            
            # Extract node information
            node_id = update_info.get('node_id')
            leaf_data = update_info.get('leaf_data', {})
            
            print(f"\n📡 Callback #{self.callback_iteration}: Node {node_id}")
            
            # Extract _att_dist_per_class from splitters
            att_dist_data = {}
            
            if 'splitters' in leaf_data:
                for feature_name, splitter_info in leaf_data['splitters'].items():
                    if 'gaussian_data' in splitter_info:
                        gaussian_data = splitter_info['gaussian_data']
                        if 'distributions' in gaussian_data:
                            att_dist_data[feature_name] = gaussian_data['distributions']
            
            # Extract leaf statistics
            stats = leaf_data.get('stats', {})
            total_weight = leaf_data.get('total_weight', 0)
            
            # Extract Naive Bayes weights
            naive_bayes = leaf_data.get('naive_bayes', {})
            mc_correct_weight = naive_bayes.get('mc_correct_weight', 0)
            nb_correct_weight = naive_bayes.get('nb_correct_weight', 0)
            
            # Create Kafka message
            kafka_message = {
                'iteration': self.callback_iteration,
                'node_id': node_id,
                'timestamp': time.time(),
                '_att_dist_per_class': att_dist_data,
                'stats': stats,
                'total_weight': total_weight,
                'mc_correct_weight': mc_correct_weight,
                'nb_correct_weight': nb_correct_weight
            }
            
            # Publish to Kafka with partition key to ensure ordering
            future = self.producer.send(
                KafkaConfig.TOPIC_LEAF_UPDATES,
                key=str(node_id).encode('utf-8'),  # Use node_id as key for partitioning
                value=kafka_message
            )
            
            # Wait for send to complete
            try:
                record_metadata = future.get(timeout=10)
                print(f"   ✅ Published to Kafka: {KafkaConfig.TOPIC_LEAF_UPDATES}")
                print(f"      Partition: {record_metadata.partition}, Offset: {record_metadata.offset}")
                print(f"      Stats: {stats}, Weights: MC={mc_correct_weight}, NB={nb_correct_weight}")
            except Exception as e:
                print(f"   ❌ Failed to publish: {e}")
        
        # Create tree with callback
        tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba',
            leaf_update_threshold=1,
            leaf_update_callback=_leaf_update_callback
        )
        
        # Train the tree
        print(f"📚 Training on {self.training_samples} instances...")
        for i, instance in enumerate(instances[:self.training_samples]):
            tree.learn_one(instance['x'], instance['y'])
            
            if (i + 1) % 10 == 0:
                print(f"   📈 Trained on {i + 1}/{self.training_samples} instances")
        
        # Flush producer
        self.producer.flush()
        print(f"\n✅ Training complete! Published {self.callback_iteration} updates to Kafka")
        print(f"📊 Final tree state:")
        
        # Show final tree state
        if len(tree._node_registry) > 0:
            node_id = list(tree._node_registry.keys())[0]
            node = tree._node_registry[node_id]
            stats = dict(getattr(node, 'stats', {}))
            mc_weight = getattr(node, '_mc_correct_weight', 0)
            nb_weight = getattr(node, '_nb_correct_weight', 0)
            print(f"   Node {node_id}: Stats={stats}, MC={mc_weight}, NB={nb_weight}")
        
        return tree


def main():
    """Main entry point."""
    print("🚀 KAFKA TRAINING PROCESS - DISTRIBUTED HOEFFDING TREE")
    print("=" * 60)
    print(f"📡 Kafka Broker: kafka_service:9092")
    print(f"📢 Topic: hoeffding_leaf_updates")
    print("=" * 60)
    
    # Create training process
    trainer = TrainingProcess(n_training_samples=10)
    
    # Generate dataset
    instances = trainer.generate_dataset()
    
    # Train and publish to Kafka
    tree = trainer.train(instances)
    
    print("\n🎯 Training process complete!")
    print("   The inference process can now consume these updates.")
    print("=" * 60)


if __name__ == "__main__":
    main()
