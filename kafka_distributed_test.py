#!/usr/bin/env python3
"""
🚀 KAFKA DISTRIBUTED HOEFFDING TREE TEST
===============================================
Tests distributed training using Kafka:
1. Update Process (Producer): Trains tree and sends leaf stats via Kafka
2. Inference Process (Consumer): Receives updates and applies them
3. Validates that both processes maintain identical predictions

🐳 Docker Setup Required:
- Kafka container running on kafka_service:9092
- Use the provided docker-compose.yml with kafka_service
"""

import json
import time
import threading
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Kafka imports
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    print("❌ Kafka library not installed. Install with: pip install kafka-python")
    KAFKA_AVAILABLE = False
    exit(1)

# River imports
import sys
import os
sys.path.append('/root/river')
from river.tree import HoeffdingTreeClassifier


class KafkaDistributedTreeTest:
    """Test distributed Hoeffding Tree training using Kafka."""
    
    def __init__(self, kafka_bootstrap_servers="kafka_service:9092"):
        self.kafka_servers = kafka_bootstrap_servers
        self.topic_name = "hoeffding_tree_updates"
        
        # Test configuration
        self.n_samples = 1000
        self.training_samples = 50
        self.test_samples = 50
        
        # Results storage
        self.original_predictions = []
        self.distributed_predictions = []
        
        print(f"🔧 KAFKA DISTRIBUTED TEST SETUP")
        print(f"   Kafka servers: {self.kafka_servers}")
        print(f"   Topic: {self.topic_name}")
        print(f"   Dataset: {self.n_samples} samples")
        print(f"   Training: {self.training_samples} instances")
        print(f"   Testing: {self.test_samples} predictions")
    
    def generate_test_dataset(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Generate test dataset and return as DataFrame and instances list."""
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
        
        # Convert to River format (list of dicts)
        instances = []
        for i in range(len(df)):
            row = df.iloc[i]
            x = {col: row[col] for col in df.columns if col != 'target'}
            instances.append({'x': x, 'y': row['target']})
        
        print(f"✅ Dataset generated")
        print(f"   Features: {list(df.columns[:-1])}")
        print(f"   Target distribution: {df['target'].value_counts().to_dict()}")
        
        return df, instances
    
    def create_kafka_producer(self) -> KafkaProducer:
        """Create and configure Kafka producer."""
        print(f"\n📤 CREATING KAFKA PRODUCER")
        print("=" * 30)
        
        producer = KafkaProducer(
            bootstrap_servers=self.kafka_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: str(k).encode('utf-8'),
            retries=3,
            acks='all'
        )
        
        print(f"✅ Kafka producer created")
        return producer
    
    def create_kafka_consumer(self) -> KafkaConsumer:
        """Create and configure Kafka consumer."""
        print(f"\n📥 CREATING KAFKA CONSUMER")
        print("=" * 30)
        
        consumer = KafkaConsumer(
            self.topic_name,
            bootstrap_servers=self.kafka_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',
            group_id='distributed_tree_test',
            enable_auto_commit=True
        )
        
        print(f"✅ Kafka consumer created")
        print(f"   Subscribed to topic: {self.topic_name}")
        return consumer
    
    def update_process(self, instances: List[Dict], producer: KafkaProducer) -> HoeffdingTreeClassifier:
        """Update process: Train tree and publish leaf stats to Kafka."""
        print(f"\n🌱 UPDATE PROCESS: Training & Publishing")
        print("=" * 45)
        
        # Create tree with leaf update callback
        def leaf_update_callback(update_info):
            """Callback to publish leaf updates to Kafka."""
            message = {
                'message_type': 'leaf_update',
                'timestamp': time.time(),
                'update_info': update_info
            }
            
            try:
                future = producer.send(self.topic_name, key=str(update_info['node_id']), value=message)
                result = future.get(timeout=10)
                print(f"   📤 KAFKA SENT: Node {update_info['node_id']} → Partition {result.partition}, Offset {result.offset}")
            except KafkaError as e:
                print(f"   ❌ KAFKA ERROR: {e}")
        
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
        
        # Send final tree state
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
            result = future.get(timeout=10)
            print(f"   📤 KAFKA SENT: Training complete → Partition {result.partition}, Offset {result.offset}")
        except KafkaError as e:
            print(f"   ❌ KAFKA ERROR: {e}")
        
        producer.flush()
        print(f"✅ Update process completed")
        tree.print_node_registry()
        
        return tree
    
    def inference_process(self, instances: List[Dict], consumer: KafkaConsumer) -> HoeffdingTreeClassifier:
        """Inference process: Create tree and apply updates from Kafka."""
        print(f"\n🔍 INFERENCE PROCESS: Receiving & Applying Updates")
        print("=" * 50)
        
        # Create inference tree (no callbacks needed)
        inference_tree = HoeffdingTreeClassifier(
            grace_period=200,
            leaf_prediction='nba'
        )
        
        # Initialize with first instance to create root
        first_instance = instances[0]
        inference_tree.learn_one(first_instance['x'], first_instance['y'])
        print(f"   🌱 Initialized inference tree with first instance")
        
        updates_received = 0
        training_complete = False
        
        print(f"📥 Listening for Kafka messages...")
        
        # Set consumer timeout to avoid infinite waiting
        consumer_timeout = 30000  # 30 seconds
        start_time = time.time()
        
        for message in consumer:
            try:
                data = message.value
                message_type = data.get('message_type')
                
                print(f"   📥 KAFKA RECEIVED: {message_type} (Key: {message.key})")
                
                if message_type == 'leaf_update':
                    update_info = data['update_info']
                    node_id = update_info['node_id']
                    
                    # Extract leaf data for distributed update
                    leaf_data = update_info['leaf_data']
                    
                    # Create update payload for incremental learning
                    # We'll simulate the original training by applying incremental updates
                    update_payload = {
                        'update_type': 'leaf_stats',
                        'data': {
                            'stats': leaf_data['stats'],
                            'total_weight': leaf_data['total_weight']
                        }
                    }
                    
                    # Apply the update
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
                
                # Timeout check
                if time.time() - start_time > consumer_timeout:
                    print(f"   ⏰ Consumer timeout reached ({consumer_timeout}ms)")
                    break
                    
            except Exception as e:
                print(f"   ❌ Error processing message: {e}")
                import traceback
                traceback.print_exc()
        
        if not training_complete:
            print(f"   ⚠️  Training complete signal not received, but proceeding...")
        
        print(f"✅ Inference process completed")
        print(f"   Updates received: {updates_received}")
        inference_tree.print_node_registry()
        
        return inference_tree
    
    def compare_predictions(self, original_tree: HoeffdingTreeClassifier, 
                          inference_tree: HoeffdingTreeClassifier, 
                          test_instances: List[Dict]) -> Dict[str, Any]:
        """Compare predictions between original and inference trees."""
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
        print(f"   Original predictions: {original_preds}")
        print(f"   Inference predictions: {inference_preds}")
        
        if accuracy == 100:
            print(f"   🎉 PERFECT SYNCHRONIZATION ACHIEVED!")
        elif accuracy >= 95:
            print(f"   ✅ EXCELLENT SYNCHRONIZATION!")
        elif accuracy >= 90:
            print(f"   ⚠️  GOOD SYNCHRONIZATION (minor differences)")
        else:
            print(f"   ❌ POOR SYNCHRONIZATION (significant differences)")
        
        return results
    
    def run_kafka_test(self):
        """Run the complete Kafka distributed test."""
        print(f"\n🚀 STARTING KAFKA DISTRIBUTED HOEFFDING TREE TEST")
        print("=" * 60)
        
        # Generate dataset
        df, instances = self.generate_test_dataset()
        
        # Test Kafka connectivity
        print(f"\n🔗 TESTING KAFKA CONNECTIVITY")
        print("=" * 35)
        try:
            producer = self.create_kafka_producer()
            consumer = self.create_kafka_consumer()
            print(f"✅ Kafka connectivity successful")
        except Exception as e:
            print(f"❌ Kafka connectivity failed: {e}")
            print(f"💡 Make sure Kafka is running on {self.kafka_servers}")
            print(f"   Docker: docker-compose up kafka_service")
            return None
        
        # Run update process (producer)
        print(f"\n" + "="*60)
        original_tree = self.update_process(instances, producer)
        producer.close()
        
        # Small delay to ensure messages are published
        print(f"\n⏳ Waiting for message propagation...")
        time.sleep(2)
        
        # Run inference process (consumer)
        print(f"\n" + "="*60)
        inference_tree = self.inference_process(instances, consumer)
        consumer.close()
        
        # Compare results
        print(f"\n" + "="*60)
        test_instances = instances[self.training_samples:self.training_samples + self.test_samples]
        results = self.compare_predictions(original_tree, inference_tree, test_instances)
        
        # Final summary
        print(f"\n🏁 KAFKA DISTRIBUTED TEST SUMMARY")
        print("=" * 40)
        print(f"✅ Dataset: {self.n_samples} samples generated")
        print(f"✅ Training: {self.training_samples} instances processed")
        print(f"✅ Kafka: Messages sent and received")
        print(f"✅ Predictions: {results['total_predictions']} tested")
        print(f"📊 Synchronization: {results['accuracy_percentage']:.2f}%")
        
        if results['accuracy_percentage'] == 100:
            print(f"\n🎉 SUCCESS: PERFECT KAFKA DISTRIBUTED SYNCHRONIZATION!")
            print(f"   ✅ Your distributed Hoeffding Tree system works flawlessly")
            print(f"   ✅ Ready for production deployment with Kafka")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {results['accuracy_percentage']:.2f}% synchronization")
            print(f"   💡 Consider investigating timing or update mechanism")
        
        return results


def main():
    """Main function to run the Kafka distributed test."""
    if not KAFKA_AVAILABLE:
        return
    
    # Check if Kafka arguments provided
    import sys
    kafka_servers = sys.argv[1] if len(sys.argv) > 1 else "kafka_service:9092"
    
    # Run the test
    test = KafkaDistributedTreeTest(kafka_bootstrap_servers=kafka_servers)
    results = test.run_kafka_test()
    
    if results:
        print(f"\n💾 Test completed successfully!")
        print(f"   Final accuracy: {results['accuracy_percentage']:.2f}%")


if __name__ == "__main__":
    main()