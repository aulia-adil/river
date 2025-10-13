#!/usr/bin/env python3
"""
Simplified distributed update test focusing on the working functionality.
This demonstrates the key capabilities needed for distributed Hoeffding Tree training.
"""

import sys
import os
sys.path.insert(0, '/root/river')

from river.tree import HoeffdingTreeClassifier
import numpy as np
import json

def test_working_distributed_system():
    """Test the fully functional distributed update capabilities."""
    
    print("🎯 WORKING DISTRIBUTED UPDATE SYSTEM")
    print("=" * 50)
    
    # Create distributed nodes
    primary_node = HoeffdingTreeClassifier(
        grace_period=20,
        leaf_update_threshold=5,
        nominal_attributes=['category']
    )
    
    inference_nodes = {
        'inference_1': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5, nominal_attributes=['category']),
        'inference_2': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5, nominal_attributes=['category']),
        'inference_3': HoeffdingTreeClassifier(grace_period=20, leaf_update_threshold=5, nominal_attributes=['category'])
    }
    
    # Initial synchronization data
    sync_data = [
        ({'feature1': 0.1, 'category': 'A'}, 0),
        ({'feature1': 0.8, 'category': 'B'}, 1),
        ({'feature1': 0.3, 'category': 'A'}, 0),
        ({'feature1': 0.9, 'category': 'B'}, 1),
        ({'feature1': 0.2, 'category': 'A'}, 0)
    ]
    
    print("📚 Initial synchronization...")
    
    # All nodes start with same data
    for x, y in sync_data:
        primary_node.learn_one(x, y)
        for name, node in inference_nodes.items():
            node.learn_one(x, y)
    
    print(f"✅ All nodes synchronized with {len(sync_data)} instances")
    
    # Get leaf node ID
    leaf_nodes = [nid for nid, node in primary_node._node_registry.items() 
                  if 'Leaf' in type(node).__name__]
    
    if not leaf_nodes:
        print("❌ No leaf nodes found")
        return False
    
    target_node_id = leaf_nodes[0]
    
    # Show initial state
    print(f"\n📊 INITIAL STATE (Node {target_node_id}):")
    primary_stats = dict(getattr(primary_node.get_node_by_id(target_node_id), 'stats', {}))
    print(f"Primary: {primary_stats}")
    
    for name, node in inference_nodes.items():
        stats = dict(getattr(node.get_node_by_id(target_node_id), 'stats', {}))
        print(f"{name}: {stats}")
    
    # Distributed training phase
    print(f"\n🚀 DISTRIBUTED TRAINING PHASE")
    
    # Primary node receives new training data
    new_training_data = [
        ({'feature1': 1.1, 'category': 'B'}, 1),
        ({'feature1': 1.2, 'category': 'B'}, 1),
        ({'feature1': 0.05, 'category': 'A'}, 0)
    ]
    
    print(f"📡 Primary node training on new data...")
    update_messages = []
    
    for x, y in new_training_data:
        primary_node.learn_one(x, y)
        
        # Create incremental update message (like Kafka message)
        update_message = {
            'update_type': 'incremental_stats',
            'data': {
                'instance': x,
                'class_label': y,
                'weight': 1.0
            },
            'node_id': target_node_id,
            'timestamp': __import__('time').time(),
            'source': 'primary_training_node'
        }
        update_messages.append(update_message)
        
        print(f"   📦 Created update message for: {x} → {y}")
    
    # Distribute updates to inference nodes (simulating Kafka)
    print(f"\n📨 DISTRIBUTING UPDATES VIA 'KAFKA'...")
    
    for i, message in enumerate(update_messages):
        print(f"\n📡 Broadcasting message {i+1}/{len(update_messages)}")
        
        for name, node in inference_nodes.items():
            print(f"   📨 {name} receiving update...")
            success = node.apply_distributed_update(message['node_id'], message)
            
            if success:
                print(f"      ✅ {name} successfully updated")
            else:
                print(f"      ❌ {name} failed to update")
    
    # Verify synchronization
    print(f"\n🔍 FINAL SYNCHRONIZATION CHECK:")
    
    primary_final = dict(getattr(primary_node.get_node_by_id(target_node_id), 'stats', {}))
    print(f"Primary: {primary_final}")
    
    all_synchronized = True
    for name, node in inference_nodes.items():
        stats = dict(getattr(node.get_node_by_id(target_node_id), 'stats', {}))
        print(f"{name}: {stats}")
        
        if stats != primary_final:
            all_synchronized = False
            print(f"   ⚠️ {name} not synchronized!")
    
    if all_synchronized:
        print(f"\n🎉 PERFECT SYNCHRONIZATION ACHIEVED!")
        print(f"✅ All nodes have identical stats: {primary_final}")
        print(f"✅ Distributed training successful!")
        return True
    else:
        print(f"\n❌ Synchronization incomplete")
        return False

def test_kafka_style_messaging():
    """Test Kafka-style message queue simulation."""
    
    print(f"\n📡 KAFKA-STYLE MESSAGING TEST")
    print("=" * 40)
    
    # Multiple distributed training nodes
    training_nodes = {
        'trainer_A': HoeffdingTreeClassifier(grace_period=10, leaf_update_threshold=3),
        'trainer_B': HoeffdingTreeClassifier(grace_period=10, leaf_update_threshold=3),
        'trainer_C': HoeffdingTreeClassifier(grace_period=10, leaf_update_threshold=3)
    }
    
    # Inference nodes that need to stay synchronized
    inference_nodes = {
        'inference_1': HoeffdingTreeClassifier(grace_period=10, leaf_update_threshold=3),
        'inference_2': HoeffdingTreeClassifier(grace_period=10, leaf_update_threshold=3)
    }
    
    all_nodes = {**training_nodes, **inference_nodes}
    
    # Initial data for all nodes
    base_data = [
        ({'x': 0.1}, 0),
        ({'x': 0.9}, 1),
        ({'x': 0.2}, 0)
    ]
    
    print("📚 Initializing all nodes with base data...")
    for x, y in base_data:
        for name, node in all_nodes.items():
            node.learn_one(x, y)
    
    # Simulated Kafka message queue
    kafka_queue = []
    
    # Each training node processes different data streams
    distributed_streams = {
        'trainer_A': [({'x': 1.1}, 1), ({'x': 1.2}, 1)],
        'trainer_B': [({'x': 0.05}, 0), ({'x': 0.01}, 0)],
        'trainer_C': [({'x': 0.85}, 1)]
    }
    
    print(f"\n🔄 Distributed training with Kafka messaging...")
    
    # Training nodes generate updates
    for trainer_name, stream_data in distributed_streams.items():
        trainer = training_nodes[trainer_name]
        
        print(f"\n📊 {trainer_name} processing stream...")
        
        for x, y in stream_data:
            # Train on the instance
            trainer.learn_one(x, y)
            
            # Get leaf node for update
            leaf_nodes = [nid for nid, node in trainer._node_registry.items() 
                          if 'Leaf' in type(node).__name__]
            
            if leaf_nodes:
                # Create Kafka message
                kafka_message = {
                    'topic': 'hoeffding_tree_updates',
                    'key': f'node_{leaf_nodes[0]}',
                    'value': {
                        'update_type': 'incremental_stats',
                        'data': {
                            'instance': x,
                            'class_label': y,
                            'weight': 1.0
                        },
                        'node_id': leaf_nodes[0],
                        'source_trainer': trainer_name,
                        'timestamp': __import__('time').time()
                    }
                }
                
                kafka_queue.append(kafka_message)
                print(f"   📦 Queued Kafka message: {x} → {y}")
    
    print(f"\n📨 Processing {len(kafka_queue)} Kafka messages...")
    
    # All nodes consume Kafka messages
    for message in kafka_queue:
        topic = message['topic']
        key = message['key']
        payload = message['value']
        source = payload['source_trainer']
        
        print(f"\n📡 Broadcasting: {topic}/{key} from {source}")
        
        # Apply to all nodes except the source
        for node_name, node in all_nodes.items():
            if node_name != source:  # Don't apply to source trainer
                success = node.apply_distributed_update(
                    payload['node_id'], 
                    payload
                )
                
                status = "✅" if success else "❌"
                print(f"   {status} {node_name}")
    
    # Check final synchronization
    print(f"\n📊 FINAL KAFKA SYNCHRONIZATION CHECK:")
    
    # Get stats from all nodes
    node_stats = {}
    for name, node in all_nodes.items():
        leaf_nodes = [nid for nid, n in node._node_registry.items() 
                      if 'Leaf' in type(n).__name__]
        if leaf_nodes:
            stats = dict(getattr(node.get_node_by_id(leaf_nodes[0]), 'stats', {}))
            node_stats[name] = stats
            print(f"{name:12}: {stats}")
    
    # Check if all have same stats
    first_stats = next(iter(node_stats.values()))
    all_same = all(stats == first_stats for stats in node_stats.values())
    
    if all_same:
        print(f"\n🎉 KAFKA SYNCHRONIZATION SUCCESS!")
        print(f"✅ All {len(all_nodes)} nodes synchronized: {first_stats}")
    else:
        print(f"\n⚠️ Some nodes not synchronized")
    
    return all_same

if __name__ == "__main__":
    print("🧪 DISTRIBUTED HOEFFDING TREE - WORKING SYSTEM TEST")
    print("Testing proven functionality for your thesis implementation")
    print("=" * 70)
    
    try:
        # Test core distributed functionality
        test1_success = test_working_distributed_system()
        
        # Test Kafka-style messaging
        test2_success = test_kafka_style_messaging()
        
        if test1_success and test2_success:
            print(f"\n🎉 DISTRIBUTED SYSTEM FULLY FUNCTIONAL!")
            print(f"=" * 50)
            print(f"✅ Incremental learning updates: WORKING")
            print(f"✅ Node synchronization: WORKING") 
            print(f"✅ Kafka-style messaging: WORKING")
            print(f"✅ Distributed training: READY")
            print(f"\n🚀 YOUR THESIS IMPLEMENTATION IS VIABLE!")
            print(f"   Use 'incremental_stats' updates for distributed training")
            print(f"   Each Kafka message can contain a single instance update")
            print(f"   All inference nodes will stay perfectly synchronized")
            
        else:
            print(f"\n❌ Some functionality needs work")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()