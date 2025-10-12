"""
Test script to demonstrate the Node ID system in HoeffdingTreeClassifier.
This shows O(1) lookup capability for distributed training scenarios.
"""

import sys
import os
sys.path.append('/root/river')

def test_node_id_system():
    """Test the node ID system with a simple tree."""
    print("🧪 TESTING NODE ID SYSTEM")
    print("=" * 50)
    
    try:
        # Import River components
        from river.tree import HoeffdingTreeClassifier
        from river.datasets import synth
        
        # Create tree with small grace period for quick splits
        tree = HoeffdingTreeClassifier(
            grace_period=5,  # Small grace period to trigger splits quickly
            delta=1e-5,
            leaf_prediction='nba'
        )
        
        print("✅ Created HoeffdingTreeClassifier with Node ID system")
        print(f"   Initial registry size: {tree.get_node_registry_size()}")
        
        # Generate some synthetic data
        dataset = synth.Agrawal(classification_function=0, seed=42)
        
        print("\n📚 Training tree and observing node IDs...")
        
        # Train with several samples
        for i, (x, y) in enumerate(dataset.take(20)):
            print(f"\n--- Training instance {i+1} ---")
            print(f"Features: {dict(list(x.items())[:3])}...")  # Show first 3 features
            print(f"Label: {y}")
            
            tree.learn_one(x, y)
            
            # Show registry status after each instance
            print(f"Registry size: {tree.get_node_registry_size()}")
            
            # Show all node IDs after every 5 instances
            if (i + 1) % 5 == 0:
                print(f"\n📋 Node Registry Status after {i+1} instances:")
                tree.print_node_registry()
        
        print(f"\n🎯 FINAL NODE REGISTRY:")
        tree.print_node_registry()
        
        # Test O(1) lookup
        print(f"\n🔍 TESTING O(1) NODE LOOKUP:")
        print("=" * 40)
        
        all_ids = tree.get_all_node_ids()
        print(f"All node IDs: {all_ids}")
        
        for node_id in all_ids[:3]:  # Test first 3 nodes
            node = tree.get_node_by_id(node_id)
            if node:
                node_type = type(node).__name__
                depth = getattr(node, 'depth', 'N/A')
                print(f"   ID {node_id}: {node_type} at depth {depth}")
                
                # Show node-specific data
                if hasattr(node, 'stats'):
                    print(f"      Stats: {node.stats}")
                if hasattr(node, 'feature'):
                    print(f"      Split feature: {node.feature}")
                if hasattr(node, 'splitters') and node.splitters:
                    print(f"      Splitters: {list(node.splitters.keys())}")
            else:
                print(f"   ID {node_id}: Node not found!")
        
        # Test prediction with node tracking
        print(f"\n🔮 TESTING PREDICTION WITH NODE TRACKING:")
        print("=" * 45)
        
        test_instance = {'age': 35, 'elevel': 1, 'car': 2, 'zipcode': 3}
        print(f"Test instance: {test_instance}")
        
        prediction = tree.predict_proba_one(test_instance)
        print(f"Prediction: {prediction}")
        
        # Test node update functionality
        print(f"\n🔄 TESTING NODE UPDATE SYSTEM:")
        print("=" * 35)
        
        if all_ids:
            test_node_id = all_ids[0]
            update_data = {'new_weight': 100.0, 'updated_stats': {0: 50, 1: 50}}
            success = tree.update_node_in_registry(test_node_id, update_data)
            print(f"Update result: {'Success' if success else 'Failed'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   This requires full River installation")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demonstrate_kafka_integration():
    """Show how the node ID system integrates with Kafka distributed training."""
    print(f"\n📡 KAFKA INTEGRATION DEMONSTRATION:")
    print("=" * 40)
    
    print("With Node IDs, your Kafka messages can now include:")
    print("1. node_id: Unique identifier for O(1) lookup")
    print("2. node_type: Leaf or Split node")  
    print("3. update_type: split, update, delete")
    print("4. node_data: Complete node state")
    
    sample_kafka_message = {
        "message_type": "node_split",
        "tree_id": "training_process_1",
        "timestamp": 1697123456.789,
        "split_info": {
            "original_leaf_id": 5,  # O(1) lookup on inference side
            "new_split_node_id": 12,
            "new_leaf_ids": [13, 14],
            "split_feature": "age",
            "split_threshold": 35.5,
            "original_leaf_data": {
                "stats": {0: 25, 1: 30},
                "splitters": {
                    "age": {"type": "GaussianSplitter", "means": {0: 28.5, 1: 42.3}},
                    "salary": {"type": "GaussianSplitter", "variances": {0: 1000, 1: 2000}}
                }
            }
        }
    }
    
    import json
    print("\nSample Kafka message structure:")
    print(json.dumps(sample_kafka_message, indent=2))
    
    print(f"\n🚀 BENEFITS FOR DISTRIBUTED TRAINING:")
    print("=" * 40)
    print("✅ O(1) node lookup instead of tree traversal")
    print("✅ Direct node updates without searching")
    print("✅ Precise change tracking for each node")
    print("✅ Efficient synchronization between processes")
    print("✅ Easy debugging with unique identifiers")

if __name__ == "__main__":
    print("🚀 NODE ID SYSTEM DEMONSTRATION")
    print("=" * 50)
    
    success = test_node_id_system()
    
    if success:
        demonstrate_kafka_integration()
        print(f"\n🎉 SUCCESS!")
        print("   Node ID system is working and ready")
        print("   for your distributed Hoeffding Tree thesis!")
    else:
        print(f"\n⚠️  Note: Run with full River installation:")
        print("   /root/.cache/pypoetry/virtualenvs/river-Llde-jfc-py3.12/bin/python /root/river/test_node_ids.py")
    
    print(f"\n🎯 USAGE SUMMARY:")
    print("- tree.get_node_by_id(node_id) - O(1) lookup")
    print("- tree.get_all_node_ids() - List all IDs")
    print("- tree.print_node_registry() - Debug view")
    print("- tree.update_node_in_registry() - Update tracking")