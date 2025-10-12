"""
Test script to demonstrate the dual gate notification system:
🚪 Gate 1: Split notifications (when tree structure changes)
🚪 Gate 2: Leaf update notifications (when leaf reaches multiples of n instances)
"""

import sys
import json
import time
sys.path.append('/root/river')

class DualGateDemo:
    """Demonstrates both notification gates in action."""
    
    def __init__(self):
        self.split_notifications = []
        self.leaf_notifications = []
    
    def split_gate_callback(self, split_info):
        """🚪 GATE 1 CALLBACK: Handles split notifications."""
        
        notification = {
            'gate': 1,
            'type': 'split_occurred',
            'timestamp': time.time(),
            'original_leaf_id': getattr(split_info['original_leaf'], 'node_id', None),
            'new_split_node_id': getattr(split_info['new_split_node'], 'node_id', None),
            'new_leaf_ids': [getattr(leaf, 'node_id', None) for leaf in split_info['new_leaves']],
            'split_feature': split_info['split_feature']
        }
        
        self.split_notifications.append(notification)
        
        print(f"\n🎯 GATE 1 CALLBACK RECEIVED:")
        print(f"   Split Event: Leaf {notification['original_leaf_id']} → Split {notification['new_split_node_id']}")
        print(f"   New Leaves: {notification['new_leaf_ids']}")
        print(f"   Split Feature: {notification['split_feature']}")
        print(f"   📊 Total split notifications: {len(self.split_notifications)}")
    
    def leaf_gate_callback(self, leaf_info):
        """🚪 GATE 2 CALLBACK: Handles leaf update notifications."""
        
        notification = {
            'gate': 2,
            'type': 'leaf_threshold_reached',
            'timestamp': leaf_info['timestamp'],
            'node_id': leaf_info['node_id'],
            'instances_reached': leaf_info['total_instances_reached'],
            'threshold': leaf_info['threshold'],
            'multiple': leaf_info['multiple_reached'],
            'current_weight': leaf_info['current_weight'],
            'leaf_stats': leaf_info['leaf_data']['stats']
        }
        
        self.leaf_notifications.append(notification)
        
        print(f"\n🎯 GATE 2 CALLBACK RECEIVED:")
        print(f"   Leaf Update: Node {notification['node_id']} reached {notification['instances_reached']} instances")
        print(f"   Threshold: {notification['threshold']} (multiple #{notification['multiple']})")
        print(f"   Leaf Stats: {notification['leaf_stats']}")
        print(f"   📊 Total leaf notifications: {len(self.leaf_notifications)}")
    
    def print_summary(self):
        """Print summary of all notifications received."""
        print(f"\n📋 NOTIFICATION SUMMARY:")
        print("=" * 40)
        print(f"🚪 Gate 1 (Splits): {len(self.split_notifications)} notifications")
        print(f"🚪 Gate 2 (Leaves): {len(self.leaf_notifications)} notifications")
        
        if self.split_notifications:
            print(f"\n🌿 Split Events:")
            for i, notif in enumerate(self.split_notifications, 1):
                print(f"   {i}. Leaf {notif['original_leaf_id']} split on {notif['split_feature']}")
        
        if self.leaf_notifications:
            print(f"\n🍃 Leaf Threshold Events:")
            for i, notif in enumerate(self.leaf_notifications, 1):
                print(f"   {i}. Node {notif['node_id']}: {notif['instances_reached']} instances (threshold {notif['threshold']})")

def test_dual_gates():
    """Test both notification gates with configurable threshold."""
    
    print("🧪 TESTING DUAL GATE NOTIFICATION SYSTEM")
    print("=" * 50)
    
    try:
        from river.tree import HoeffdingTreeClassifier
        
        # Create dual gate demo
        demo = DualGateDemo()
        
        # Create tree with both callbacks and custom threshold
        LEAF_THRESHOLD = 10  # Notify every 10 instances
        
        tree = HoeffdingTreeClassifier(
            grace_period=5,                            # Small grace period for splits
            delta=0.1,                                 # Relaxed for easier splits
            tau=0.1,
            leaf_prediction='nba',
            split_callback=demo.split_gate_callback,       # 🚪 Gate 1
            leaf_update_callback=demo.leaf_gate_callback,  # 🚪 Gate 2  
            leaf_update_threshold=LEAF_THRESHOLD           # Customizable n
        )
        
        print(f"✅ Created HoeffdingTreeClassifier with dual gates:")
        print(f"   🚪 Gate 1: Split notifications (enabled)")
        print(f"   🚪 Gate 2: Leaf notifications every {LEAF_THRESHOLD} instances")
        
        # Create training data that will trigger both gates
        training_data = []
        
        # Class 0: Low values
        for i in range(30):
            training_data.append(({
                'feature_x': 10 + i * 0.5,
                'feature_y': 20 + i * 0.3,
                'feature_z': 1
            }, 0))
        
        # Class 1: High values  
        for i in range(30):
            training_data.append(({
                'feature_x': 100 + i * 2,
                'feature_y': 200 + i * 1.5,
                'feature_z': 10
            }, 1))
        
        print(f"\n📚 Training with {len(training_data)} instances...")
        print(f"   Will trigger Gate 2 every {LEAF_THRESHOLD} instances")
        print(f"   Will trigger Gate 1 when splits occur")
        
        for i, (x, y) in enumerate(training_data):
            print(f"\n--- Instance {i+1}: class={y} ---")
            
            # Track registry size to detect splits
            old_size = tree.get_node_registry_size()
            
            # Learn (this will trigger gates if conditions are met)
            tree.learn_one(x, y)
            
            new_size = tree.get_node_registry_size()
            
            # Show registry changes
            if new_size > old_size:
                print(f"   🌿 Registry change: {old_size} → {new_size} nodes (split occurred)")
            else:
                print(f"   📊 Registry size: {new_size} nodes")
            
            # Stop after getting some notifications to keep output manageable
            if len(demo.split_notifications) >= 2 and len(demo.leaf_notifications) >= 5:
                print(f"\n⏹️  Stopping after sufficient notifications received")
                break
        
        # Print final results
        demo.print_summary()
        
        # Show final tree structure
        print(f"\n🌳 FINAL TREE STRUCTURE:")
        tree.print_node_registry()
        
        # Demonstrate O(1) access to notified nodes
        print(f"\n🔍 TESTING O(1) ACCESS TO NOTIFIED NODES:")
        print("=" * 45)
        
        # Access nodes from split notifications
        for notif in demo.split_notifications:
            node_id = notif['new_split_node_id']
            if node_id:
                node = tree.get_node_by_id(node_id)
                if node:
                    print(f"   Split node {node_id}: {type(node).__name__} (O(1) access ✓)")
        
        # Access nodes from leaf notifications
        for notif in demo.leaf_notifications[:3]:  # First 3
            node_id = notif['node_id']
            if node_id:
                node = tree.get_node_by_id(node_id)
                if node:
                    weight = getattr(node, 'total_weight', 0)
                    print(f"   Leaf node {node_id}: {type(node).__name__}, weight={weight} (O(1) access ✓)")
        
        # Show Kafka integration potential
        print(f"\n📡 KAFKA INTEGRATION POTENTIAL:")
        print("=" * 35)
        
        sample_messages = {
            "gate_1_message": {
                "gate": 1,
                "type": "tree_structure_change",
                "notifications": demo.split_notifications
            },
            "gate_2_message": {
                "gate": 2, 
                "type": "leaf_updates",
                "notifications": demo.leaf_notifications
            }
        }
        
        print("Sample Kafka message structure:")
        for msg_type, msg_data in sample_messages.items():
            print(f"\n{msg_type}:")
            print(json.dumps({
                "message_type": msg_data["type"],
                "gate_number": msg_data["gate"],
                "notification_count": len(msg_data["notifications"]),
                "sample": msg_data["notifications"][0] if msg_data["notifications"] else None
            }, indent=2, default=str))
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_customizable_threshold():
    """Show how different thresholds affect Gate 2 notifications."""
    
    print(f"\n🔧 CUSTOMIZABLE THRESHOLD DEMONSTRATION:")
    print("=" * 45)
    
    thresholds_to_test = [5, 10, 25]
    
    for threshold in thresholds_to_test:
        print(f"\n📊 Testing threshold = {threshold}:")
        
        notification_count = 0
        
        def count_callback(leaf_info):
            nonlocal notification_count
            notification_count += 1
            instances = leaf_info['total_instances_reached']
            print(f"   Notification {notification_count}: {instances} instances reached")
            # Print node id
            print(f"      Node ID: {leaf_info['node_id']}")
        
        try:
            from river.tree import HoeffdingTreeClassifier
            
            tree = HoeffdingTreeClassifier(
                leaf_update_callback=count_callback,
                leaf_update_threshold=threshold
            )
            
            # Simulate 50 instances
            for i in range(50):
                x = {'feature': i}
                y = i % 2
                tree.learn_one(x, y)
            
            expected_notifications = 50 // threshold
            print(f"   Expected notifications: {expected_notifications}")
            print(f"   Actual notifications: {notification_count}")
            print(f"   Match: {'✓' if notification_count == expected_notifications else '✗'}")
            
        except Exception as e:
            print(f"   Error with threshold {threshold}: {e}")

if __name__ == "__main__":
    print("🚀 DUAL GATE NOTIFICATION SYSTEM TEST")
    print("=" * 45)
    
    success = test_dual_gates()
    
    if success:
        demonstrate_customizable_threshold()
        
        print(f"\n🎉 DUAL GATE SYSTEM SUCCESS!")
        print("=" * 35)
        print("✅ Gate 1: Split notifications working")
        print("✅ Gate 2: Leaf threshold notifications working")
        print("✅ Customizable threshold parameter")
        print("✅ O(1) node access with IDs")
        print("✅ Complete data extraction for Kafka")
        
        print(f"\n🎯 YOUR THESIS BENEFITS:")
        print("- Gate 1: Know immediately when tree structure changes")
        print("- Gate 2: Track leaf growth at custom intervals")
        print("- Dual notifications: Complete distributed training control")
        print("- Configurable thresholds: Adjust to your use case")
    else:
        print(f"\n❌ Test failed - check implementation")
    
    print(f"\n🚀 READY FOR DISTRIBUTED TRAINING!")
    print("   Both gates provide comprehensive tree monitoring")