"""
Example demonstrating distributed Hoeffding Tree training with Kafka-like split broadcasting.

This example shows how to use the split_callback mechanism in HoeffdingTreeClassifier
to implement your thesis idea of broadcasting tree splits to multiple inference processes.
"""

import json
from collections import defaultdict
from river import tree
from river.datasets import synth


class MockKafkaProducer:
    """Mock Kafka producer for demonstration purposes."""
    
    def __init__(self):
        self.messages = []
    
    def send(self, topic, message):
        """Simulate sending a message to Kafka."""
        self.messages.append({
            'topic': topic,
            'message': message,
            'timestamp': len(self.messages)
        })
        print(f"📤 Sent to {topic}: {message['split_type']} - Feature: {message.get('split_feature', 'N/A')}")


class MockKafkaConsumer:
    """Mock Kafka consumer for demonstration purposes."""
    
    def __init__(self, producer, topic):
        self.producer = producer
        self.topic = topic
        self.last_consumed = -1
        self.tree_structure = {}  # Local copy of tree structure
    
    def poll(self):
        """Simulate polling for new messages."""
        new_messages = []
        for i, msg in enumerate(self.producer.messages[self.last_consumed + 1:], 
                               self.last_consumed + 1):
            if msg['topic'] == self.topic:
                new_messages.append(msg)
                self.last_consumed = i
                self._apply_split_update(msg['message'])
        return new_messages
    
    def _apply_split_update(self, split_info):
        """Apply the split update to local tree structure."""
        tree_id = split_info['tree_id']
        if tree_id not in self.tree_structure:
            self.tree_structure[tree_id] = {'nodes': {}, 'splits': 0}
        
        self.tree_structure[tree_id]['splits'] += 1
        print(f"📥 Consumer applied split #{self.tree_structure[tree_id]['splits']} "
              f"on feature '{split_info.get('split_feature', 'N/A')}'")


def create_split_callback(kafka_producer, topic="tree_splits"):
    """
    Factory function to create a split callback that publishes to Kafka.
    
    In a real implementation, this would serialize the node information
    and publish it to a Kafka topic for consumption by inference processes.
    """
    def split_callback(split_info):
        # In a real implementation, you'd need to serialize the node objects
        # For this demo, we'll just extract the key information
        serializable_info = {
            'split_type': split_info['split_type'],
            'split_feature': split_info['split_feature'],
            'tree_id': split_info['tree_id'],
            'num_new_leaves': len(split_info['new_leaves']),
            'split_depth': split_info['new_split_node'].depth,
            # In practice, you'd serialize the actual node structures here
            # This is where the real complexity of your thesis lies!
        }
        
        kafka_producer.send(topic, serializable_info)
    
    return split_callback


def main():
    """Demonstrate the distributed Hoeffding Tree concept."""
    
    print("🌳 Distributed Hoeffding Tree Training Demo")
    print("=" * 50)
    
    # Set up mock Kafka infrastructure
    kafka_producer = MockKafkaProducer()
    
    # Create the training tree with split callback
    training_tree = tree.HoeffdingTreeClassifier(
        grace_period=50,  # Smaller grace period for more frequent splits
        split_callback=create_split_callback(kafka_producer),
        nominal_attributes=['elevel', 'car', 'zipcode']
    )
    
    # Create mock inference consumers
    consumer1 = MockKafkaConsumer(kafka_producer, "tree_splits")
    consumer2 = MockKafkaConsumer(kafka_producer, "tree_splits")
    
    print("\n🚀 Starting training process...")
    
    # Generate some training data
    gen = synth.Agrawal(classification_function=0, seed=42)
    
    # Train the model and observe splits being broadcast
    for i, (x, y) in enumerate(gen.take(500)):
        training_tree.learn_one(x, y)
        
        # Periodically let consumers poll for updates
        if i % 100 == 0:
            print(f"\n📊 After {i+1} training instances:")
            consumer1.poll()
            consumer2.poll()
    
    print(f"\n✅ Training complete!")
    print(f"📈 Total messages sent: {len(kafka_producer.messages)}")
    print(f"🌳 Tree has {len(training_tree.classes)} classes")
    
    # Final consumer poll
    print(f"\n📊 Final consumer updates:")
    consumer1.poll()
    consumer2.poll()
    
    print(f"\nConsumer 1 processed {consumer1.tree_structure.get(id(training_tree), {}).get('splits', 0)} splits")
    print(f"Consumer 2 processed {consumer2.tree_structure.get(id(training_tree), {}).get('splits', 0)} splits")


if __name__ == "__main__":
    main()