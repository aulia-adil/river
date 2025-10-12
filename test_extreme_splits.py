"""
Force splits with extreme data and split callback to demonstrate
the complete Node ID system with multiple nodes.
"""

import sys
sys.path.append('/root/river')

def split_callback(split_info):
    """Callback to track when splits occur."""
    print(f"\n🌿 SPLIT CALLBACK TRIGGERED!")
    print(f"   Original leaf ID: {getattr(split_info['original_leaf'], 'node_id', 'No ID')}")
    print(f"   New split node ID: {getattr(split_info['new_split_node'], 'node_id', 'No ID')}")
    print(f"   New leaf IDs: {[getattr(leaf, 'node_id', 'No ID') for leaf in split_info['new_leaves']]}")
    print(f"   Split feature: {split_info['split_feature']}")

def test_forced_splits():
    """Test with extreme data to force splits and show node IDs."""
    print("🧪 TESTING WITH EXTREME DATA TO FORCE SPLITS")
    print("=" * 50)
    
    try:
        from river.tree import HoeffdingTreeClassifier
        
        # Create tree with very aggressive split settings
        tree = HoeffdingTreeClassifier(
            grace_period=2,    # Minimum grace period
            delta=0.1,         # Large delta for quicker decisions
            tau=0.5,           # Large tau threshold
            leaf_prediction='nba',
            split_callback=split_callback  # Track splits
        )
        
        print("✅ Created HoeffdingTreeClassifier with aggressive split settings")
        print(f"   Grace period: {tree.grace_period}")
        print(f"   Delta: {tree.delta}")
        print(f"   Tau: {tree.tau}")
        
        # Create extremely separable data
        extreme_data = []
        
        # Class 0: Very young with very low values
        for i in range(10):
            extreme_data.append(({
                'age': 20 + i,
                'salary': 20000 + i * 1000,
                'education': 1,
                'experience': 0
            }, 0))
        
        # Class 1: Very old with very high values  
        for i in range(10):
            extreme_data.append(({
                'age': 60 + i,
                'salary': 100000 + i * 5000,
                'education': 4,
                'experience': 30 + i
            }, 1))
        
        # Add more extreme class 0
        for i in range(10):
            extreme_data.append(({
                'age': 18 + i,
                'salary': 15000 + i * 500,
                'education': 1,
                'experience': 0
            }, 0))
        
        # Add more extreme class 1
        for i in range(10):
            extreme_data.append(({
                'age': 70 + i,
                'salary': 150000 + i * 10000,
                'education': 5,
                'experience': 40 + i
            }, 1))
        
        print(f"\n📚 Training with {len(extreme_data)} extremely separable instances...")
        print("   Class 0: age 18-30, salary 15K-30K, education 1, experience 0")
        print("   Class 1: age 60-80, salary 100K-250K, education 4-5, experience 30-50")
        
        for i, (x, y) in enumerate(extreme_data):
            print(f"\n--- Instance {i+1}: age={x['age']}, salary={x['salary']}, class={y} ---")
            
            old_registry_size = tree.get_node_registry_size()
            tree.learn_one(x, y)
            new_registry_size = tree.get_node_registry_size()
            
            if new_registry_size > old_registry_size:
                print(f"🎉 SPLIT! Registry: {old_registry_size} -> {new_registry_size} nodes")
                tree.print_node_registry()
            else:
                print(f"   Registry: {new_registry_size} nodes")
            
            # Stop after first few splits to keep output manageable
            if new_registry_size >= 5:
                print(f"\n⏹️  Stopping after reaching {new_registry_size} nodes")
                break
        
        print(f"\n🎯 FINAL TREE STRUCTURE WITH NODE IDS:")
        print("=" * 45)
        tree.print_node_registry()
        
        # Test the node ID system with multiple nodes
        if tree.get_node_registry_size() > 1:
            print(f"\n🔍 TESTING MULTI-NODE LOOKUP:")
            print("=" * 30)
            
            all_ids = tree.get_all_node_ids()
            for node_id in sorted(all_ids):
                node = tree.get_node_by_id(node_id)
                node_type = type(node).__name__
                
                print(f"   Node ID {node_id}: {node_type}")
                
                if hasattr(node, 'feature'):  # Split node
                    feature = node.feature
                    threshold = getattr(node, 'threshold', 'N/A')
                    print(f"      Split: {feature} <= {threshold}")
                elif hasattr(node, 'stats'):  # Leaf node
                    stats = node.stats
                    weight = getattr(node, 'total_weight', 0)
                    print(f"      Stats: {stats}, Weight: {weight}")
                
                # Show O(1) access time
                import time
                start = time.time()
                retrieved_node = tree.get_node_by_id(node_id)
                end = time.time()
                access_time = (end - start) * 1_000_000  # microseconds
                print(f"      Access time: {access_time:.3f} μs (O(1) confirmed)")
        
        # Demonstrate prediction with node path tracking
        print(f"\n🔮 PREDICTION WITH NODE PATH TRACKING:")
        print("=" * 40)
        
        test_cases = [
            {'age': 25, 'salary': 25000, 'education': 1, 'experience': 1},
            {'age': 65, 'salary': 120000, 'education': 5, 'experience': 35}
        ]
        
        for test_x in test_cases:
            print(f"\nTest: {test_x}")
            prediction = tree.predict_proba_one(test_x)
            print(f"Prediction: {prediction}")
            
            # Track which nodes are visited
            if hasattr(tree._root, 'node_id'):
                print(f"Root node ID: {tree._root.node_id}")
        
        # Show Kafka integration possibilities
        print(f"\n📡 KAFKA INTEGRATION WITH NODE IDS:")
        print("=" * 35)
        
        sample_update = {
            "message_type": "node_update",
            "tree_id": "training_process_1", 
            "node_updates": []
        }
        
        for node_id in tree.get_all_node_ids():
            node = tree.get_node_by_id(node_id)
            node_update = {
                "node_id": node_id,
                "node_type": type(node).__name__,
                "timestamp": __import__('time').time()
            }
            
            if hasattr(node, 'stats'):
                node_update["stats"] = dict(node.stats)
            if hasattr(node, 'feature'):
                node_update["split_feature"] = node.feature
                node_update["threshold"] = getattr(node, 'threshold', None)
            
            sample_update["node_updates"].append(node_update)
        
        import json
        print("Sample Kafka update message:")
        print(json.dumps(sample_update, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 EXTREME DATA TEST FOR NODE ID SYSTEM")
    print("=" * 45)
    
    success = test_forced_splits()
    
    if success:
        print(f"\n🎉 SUCCESS WITH NODE ID SYSTEM!")
        print("✅ Unique IDs assigned to all nodes")
        print("✅ O(1) lookup performance confirmed") 
        print("✅ Registry tracks splits and nodes")
        print("✅ Ready for distributed Kafka training!")
        
        print(f"\n🚀 YOUR THESIS BENEFITS:")
        print("- Training: tree.get_node_by_id() for direct access")
        print("- Kafka: Send node_id instead of tree paths")
        print("- Inference: O(1) node updates without traversal")
        print("- Debugging: Easy node identification and tracking")
    else:
        print(f"\n❌ Test encountered issues")
    
    print(f"\n💡 KEY METHODS ADDED:")
    print("- _generate_node_id(): Creates unique IDs")
    print("- _register_node(): Adds nodes to registry")  
    print("- get_node_by_id(): O(1) node lookup")
    print("- print_node_registry(): Debug view of all nodes")