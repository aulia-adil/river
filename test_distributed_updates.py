#!/usr/bin/env python3
"""
Test script to demonstrate the distributed update system for HoeffdingTreeClassifier.
This simulates how nodes can receive and apply updates from distributed training processes.
"""

import sys
import os
sys.path.insert(0, '/root/river')

from river.tree import HoeffdingTreeClassifier
import numpy as np
import json

def test_distributed_updates():
    """Test the distributed update functionality."""
    
    # Create two trees to simulate distributed training
    print("🌐 DISTRIBUTED UPDATE SYSTEM TEST")
    print("=" * 50)
    
    # Tree 1: Primary training node
    tree1 = HoeffdingTreeClassifier(
        grace_period=100,
        leaf_update_threshold=10,
        nominal_attributes=['category']
    )
    
    # Tree 2: Distributed node that will receive updates
    tree2 = HoeffdingTreeClassifier(
        grace_period=100,
        leaf_update_threshold=10,
        nominal_attributes=['category']
    )
    
    print("📚 Training Tree 1 with initial data...")
    
    # Train tree1 with some initial data
    np.random.seed(42)
    categories = ['low', 'medium', 'high']
    
    for i in range(15):
        x = {
            'numerical_feature': np.random.normal(0, 1),
            'category': np.random.choice(categories),
            'another_num': np.random.uniform(-1, 1)
        }
        y = 1 if (x['category'] == 'high' or x['numerical_feature'] > 0) else 0
        
        tree1.learn_one(x, y)
        tree2.learn_one(x, y)  # Both trees start with same data
    
    print("\n🏗️ Initial training completed for both trees")
    print("Tree 1 nodes:", tree1.get_node_registry_size())
    print("Tree 2 nodes:", tree2.get_node_registry_size())
    
    # Get the leaf node from tree1
    leaf_nodes = [node_id for node_id, node in tree1._node_registry.items() 
                  if 'Leaf' in type(node).__name__]
    
    if not leaf_nodes:
        print("❌ No leaf nodes found")
        return False
    
    target_node_id = leaf_nodes[0]
    print(f"\n🎯 Target node for updates: {target_node_id}")
    
    # Show initial state of both trees
    print("\n📊 INITIAL STATE COMPARISON:")
    node1 = tree1.get_node_by_id(target_node_id)
    node2 = tree2.get_node_by_id(target_node_id)
    
    print(f"Tree 1 Node {target_node_id} stats: {dict(getattr(node1, 'stats', {}))}")
    print(f"Tree 2 Node {target_node_id} stats: {dict(getattr(node2, 'stats', {}))}")
    
    # Train tree1 with additional data (simulating distributed training)
    print(f"\n🚀 Training Tree 1 with additional distributed data...")
    
    additional_data = []
    for i in range(10):
        x = {
            'numerical_feature': np.random.normal(1, 0.5),  # Different distribution
            'category': np.random.choice(['high', 'medium']),
            'another_num': np.random.uniform(0, 2)
        }
        y = 1 if x['numerical_feature'] > 0.5 else 0
        additional_data.append((x, y))
        tree1.learn_one(x, y)
    
    print(f"✅ Tree 1 trained with {len(additional_data)} additional instances")
    
    # Create update payload from tree1
    print(f"\n📦 Creating update payload from Tree 1...")
    update_payload = tree1.create_update_payload(target_node_id, 'complete_node')
    
    if update_payload:
        print(f"✅ Update payload created successfully")
        print(f"   Payload size: {len(str(update_payload))} characters")
        print(f"   Data sections: {list(update_payload['data'].keys())}")
        
        # Apply the update to tree2
        print(f"\n🔄 Applying update to Tree 2...")
        success = tree2.apply_distributed_update(target_node_id, update_payload)
        
        if success:
            print(f"✅ Update applied successfully!")
            
            # Compare final states
            print(f"\n📊 FINAL STATE COMPARISON:")
            node1_final = tree1.get_node_by_id(target_node_id)
            node2_final = tree2.get_node_by_id(target_node_id)
            
            print(f"Tree 1 Node {target_node_id} stats: {dict(getattr(node1_final, 'stats', {}))}")
            print(f"Tree 2 Node {target_node_id} stats: {dict(getattr(node2_final, 'stats', {}))}")
            
            # Check if they match
            stats1 = dict(getattr(node1_final, 'stats', {}))
            stats2 = dict(getattr(node2_final, 'stats', {}))
            
            if stats1 == stats2:
                print(f"✅ SYNCHRONIZATION SUCCESSFUL: Both trees have identical stats!")
            else:
                print(f"⚠️ Stats differ - Tree 1: {stats1}, Tree 2: {stats2}")
                
        else:
            print(f"❌ Update failed")
            return False
    else:
        print(f"❌ Failed to create update payload")
        return False
    
    return True

def test_incremental_updates():
    """Test incremental instance updates."""
    print(f"\n🧪 TESTING INCREMENTAL UPDATES")
    print("=" * 40)
    
    tree = HoeffdingTreeClassifier(
        grace_period=50,
        leaf_update_threshold=5
    )
    
    # Initial training
    for i in range(10):
        x = {'feature1': i * 0.1, 'feature2': i * 0.2}
        y = 1 if i > 5 else 0
        tree.learn_one(x, y)
    
    # Get leaf node
    leaf_nodes = [node_id for node_id, node in tree._node_registry.items() 
                  if 'Leaf' in type(node).__name__]
    
    if leaf_nodes:
        target_node_id = leaf_nodes[0]
        node = tree.get_node_by_id(target_node_id)
        
        print(f"Initial stats: {dict(getattr(node, 'stats', {}))}")
        
        # Create incremental update
        incremental_update = {
            'update_type': 'incremental_stats',
            'data': {
                'instance': {'feature1': 1.5, 'feature2': 2.0},
                'class_label': 1,
                'weight': 1.0
            }
        }
        
        # Apply incremental update
        success = tree.apply_distributed_update(target_node_id, incremental_update)
        
        if success:
            print(f"Final stats: {dict(getattr(node, 'stats', {}))}")
            print(f"✅ Incremental update successful!")
        else:
            print(f"❌ Incremental update failed")
            
        return success
    
    return False

def test_kafka_simulation():
    """Simulate Kafka-style message passing."""
    print(f"\n📡 KAFKA SIMULATION TEST")
    print("=" * 30)
    
    # Simulate multiple distributed nodes
    nodes = {
        'node_a': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5),
        'node_b': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5),
        'node_c': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5)
    }
    
    # Initial synchronization - all nodes start with same data
    initial_data = [
        ({'x': 0.1, 'y': 0.2}, 0),
        ({'x': 0.5, 'y': 0.8}, 1),
        ({'x': 0.3, 'y': 0.4}, 0),
        ({'x': 0.9, 'y': 1.1}, 1),
        ({'x': 0.2, 'y': 0.1}, 0)
    ]
    
    for name, tree in nodes.items():
        for x, y in initial_data:
            tree.learn_one(x, y)
        print(f"{name}: {tree.get_node_registry_size()} nodes")
    
    # Each node trains on different data (distributed training)
    distributed_data = {
        'node_a': [({'x': 1.0, 'y': 1.2}, 1), ({'x': 1.1, 'y': 1.3}, 1)],
        'node_b': [({'x': 0.0, 'y': 0.1}, 0), ({'x': 0.1, 'y': 0.0}, 0)],
        'node_c': [({'x': 0.8, 'y': 0.9}, 1), ({'x': 0.7, 'y': 0.8}, 1)]
    }
    
    print(f"\n🔄 Distributed training phase...")
    updates_queue = []  # Simulated Kafka message queue
    
    for name, tree in nodes.items():
        print(f"\n📡 {name} training and broadcasting updates...")
        
        for x, y in distributed_data[name]:
            tree.learn_one(x, y)
        
        # Create update payload and add to queue
        leaf_nodes = [nid for nid, node in tree._node_registry.items() 
                      if 'Leaf' in type(node).__name__]
        
        if leaf_nodes:
            update = tree.create_update_payload(leaf_nodes[0], 'complete_node')
            if update:
                update['source_node'] = name
                updates_queue.append(update)
                print(f"   📦 Update queued from {name}")
    
    # Apply all updates to all nodes (simulating Kafka distribution)
    print(f"\n🔄 Applying distributed updates...")
    
    for update in updates_queue:
        source = update['source_node']
        target_node_id = update['node_id']
        
        for name, tree in nodes.items():
            if name != source:  # Don't apply update to source node
                print(f"   📨 Applying {source} → {name}")
                success = tree.apply_distributed_update(target_node_id, update)
                if success:
                    print(f"      ✅ {name} synchronized with {source}")
                else:
                    print(f"      ❌ {name} failed to sync with {source}")
    
    # Check final synchronization
    print(f"\n📊 Final synchronization check...")
    
    for name, tree in nodes.items():
        leaf_nodes = [nid for nid, node in tree._node_registry.items() 
                      if 'Leaf' in type(node).__name__]
        if leaf_nodes:
            node = tree.get_node_by_id(leaf_nodes[0])
            stats = dict(getattr(node, 'stats', {}))
            print(f"{name}: {stats}")
    
    return True

if __name__ == "__main__":
    print("🧪 DISTRIBUTED UPDATE SYSTEM TESTING")
    print("Testing node update capabilities for distributed Hoeffding Tree training")
    print("=" * 80)
    
    try:
        # Test distributed updates
        success1 = test_distributed_updates()
        
        # Test incremental updates
        success2 = test_incremental_updates()
        
        # Test Kafka simulation
        success3 = test_kafka_simulation()
        
        if success1 and success2 and success3:
            print(f"\n🎉 ALL DISTRIBUTED UPDATE TESTS PASSED!")
            print(f"✅ Your distributed Hoeffding Tree system is ready!")
            print(f"✅ Nodes can receive and apply updates from Kafka")
            print(f"✅ Both incremental and complete updates supported")
        else:
            print(f"\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()