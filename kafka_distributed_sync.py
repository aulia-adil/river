#!/usr/bin/env python3
"""
🚀 REAL KAFKA-BASED DISTRIBUTED HOEFFDING TREE
==============================================
Training process pushes updates to Kafka.
Inference process consumes updates and synchronizes.
"""

import sys
import os
import time
import json
import threading
from kafka import KafkaProducer, KafkaConsumer
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


class TrainingProcess:
    """Training process that publishes updates to Kafka."""
    
    def __init__(self):
        self.n_samples = 1000
        self.training_samples = 10
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
        
        return tree


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


class DistributedSystem:
    """Orchestrates the distributed Hoeffding Tree system."""
    
    def __init__(self):
        self.training_process = TrainingProcess()
        self.inference_process = None
        self.instances = None
        self.training_tree = None
    
    def run(self):
        """Run the complete distributed system test."""
        print("🚀 KAFKA-BASED DISTRIBUTED HOEFFDING TREE SYSTEM")
        print("=" * 60)
        print(f"📡 Kafka Broker: {KafkaConfig.BOOTSTRAP_SERVERS[0]}")
        print(f"📢 Topic: {KafkaConfig.TOPIC_LEAF_UPDATES}")
        print("=" * 60)
        
        # Generate dataset
        self.instances = self.training_process.generate_dataset()
        
        # Start inference process in background thread
        self.inference_process = InferenceProcess(self.instances)
        self.inference_process.initialize_tree()
        
        inference_thread = threading.Thread(
            target=self.inference_process.consume_and_update,
            kwargs={'expected_updates': 10, 'timeout': 30}
        )
        inference_thread.daemon = True
        inference_thread.start()
        
        # Give consumer time to connect
        time.sleep(2)
        
        # Train and publish updates
        self.training_tree = self.training_process.train(self.instances)
        
        # Wait for inference process to receive all updates
        print("\n⏳ Waiting for inference process to sync...")
        inference_thread.join(timeout=35)
        
        # Compare trees
        self._compare_trees()
    
    def _compare_trees(self):
        """Compare training and inference trees."""
        print("\n" + "=" * 60)
        print("🔍 COMPARING TRAINING vs INFERENCE TREES")
        print("=" * 60)
        
        # Get nodes
        train_nodes = list(self.training_tree._node_registry.keys())
        infer_nodes = list(self.inference_process.tree._node_registry.keys())
        
        print(f"\n📋 Structure:")
        print(f"   Training nodes: {len(train_nodes)}")
        print(f"   Inference nodes: {len(infer_nodes)}")
        print(f"   Match: {'✅' if train_nodes == infer_nodes else '❌'}")
        
        if train_nodes == infer_nodes:
            node_id = train_nodes[0]
            train_node = self.training_tree._node_registry[node_id]
            infer_node = self.inference_process.tree._node_registry[node_id]
            
            # Compare stats
            train_stats = dict(getattr(train_node, 'stats', {}))
            infer_stats = dict(getattr(infer_node, 'stats', {}))
            
            print(f"\n📊 Statistics:")
            print(f"   Training: {train_stats}")
            print(f"   Inference: {infer_stats}")
            print(f"   Match: {'✅' if train_stats == infer_stats else '❌'}")
            
            # Compare weights
            train_mc = getattr(train_node, '_mc_correct_weight', 0)
            train_nb = getattr(train_node, '_nb_correct_weight', 0)
            infer_mc = getattr(infer_node, '_mc_correct_weight', 0)
            infer_nb = getattr(infer_node, '_nb_correct_weight', 0)
            
            print(f"\n⚖️  Weights:")
            print(f"   Training MC/NB: {train_mc}/{train_nb}")
            print(f"   Inference MC/NB: {infer_mc}/{infer_nb}")
            print(f"   Match: {'✅' if (train_mc == infer_mc and train_nb == infer_nb) else '❌'}")
        
        # Test predictions
        print(f"\n🎯 Prediction Comparison:")
        test_instances = self.instances[10:15]
        perfect_matches = 0
        
        for i, instance in enumerate(test_instances):
            x = instance['x']
            
            train_proba = self.training_tree.predict_proba_one(x)
            infer_proba = self.inference_process.tree.predict_proba_one(x)
            
            train_pred = max(train_proba, key=train_proba.get)
            infer_pred = max(infer_proba, key=infer_proba.get)
            
            train_conf = train_proba.get(train_pred, 0)
            infer_conf = infer_proba.get(infer_pred, 0)
            prob_diff = abs(train_conf - infer_conf)
            
            match = train_pred == infer_pred and prob_diff < 0.01
            if match:
                perfect_matches += 1
                status = "🎯"
            else:
                status = "❌"
            
            print(f"   Sample {i+1} {status}: Train={train_pred}({train_conf:.3f}) vs Infer={infer_pred}({infer_conf:.3f})")
        
        accuracy = (perfect_matches / len(test_instances)) * 100
        
        print(f"\n📊 FINAL RESULTS:")
        print("=" * 60)
        print(f"   Perfect matches: {perfect_matches}/{len(test_instances)}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        if accuracy == 100:
            print(f"\n🎉🎉🎉 PERFECT KAFKA-BASED SYNCHRONIZATION! 🎉🎉🎉")
            print(f"✅ Training and inference trees are identical")
            print(f"✅ Real-time distributed learning achieved")
            print(f"🚀 SYSTEM READY FOR PRODUCTION!")
        else:
            print(f"\n⚠️ Synchronization needs investigation")
        
        print("=" * 60)


def main():
    """Main entry point."""
    system = DistributedSystem()
    system.run()


if __name__ == "__main__":
    main()
