"""
Test script to demonstrate the detailed debug output of HoeffdingTreeClassifier.
This will show you exactly how the tree learns and makes decisions!

MODIFIED AGAIN: The dataset is now much larger (100 instances) and perfectly
interleaved to guarantee the root leaf sees a mix of classes, forcing a split.
"""

import sys
import random
# This path is for demonstration purposes in an isolated environment.
# You would typically have river installed in your environment.
sys.path.insert(0, '/root/river')

# Mock the imports to avoid dependency issues if river isn't fully installed
class MockSynth:
    def __init__(self, classification_function=0, seed=42):
        print("💡 Initializing MockSynth with a large, interleaved dataset to FORCE a split.")
        print("   The rule remains: income < 50000 -> class 0, else class 1.")
        
        # --- NEW LARGER, INTERLEAVED DATASET ---
        # Create two distinct groups of data
        low_income_samples = [
            ({'age': random.randint(20, 35), 'income': random.randint(20000, 49000), 'education': random.choice(['low', 'medium'])}, 0)
            for _ in range(1000)
        ]
        
        high_income_samples = [
            ({'age': random.randint(36, 60), 'income': random.randint(51000, 100000), 'education': random.choice(['medium', 'high'])}, 1)
            for _ in range(1000)
        ]
        
        # Interleave the data to ensure the first leaf sees a mix of classes
        self.data = []
        for s0, s1 in zip(low_income_samples, high_income_samples):
            self.data.append(s0)
            self.data.append(s1)
            
        self.index = 0
        print(f"   Generated {len(self.data)} total interleaved instances.")
    
    def take(self, n):
        for _ in range(min(n, len(self.data))):
            if self.index < len(self.data):
                yield self.data[self.index]
                self.index += 1

def simple_callback(split_info):
    """Simple callback to show distributed training capability"""
    print(f"\n**************************************************")
    print(f"  🚀 NODE SPLIT! DISTRIBUTED CALLBACK TRIGGERED!")
    print(f"     A leaf node has seen enough data to become an internal node.")
    print(f"     Best split found on feature: '{split_info['split_feature']}'")
    print(f"     Number of new leaves created: {len(split_info['new_leaves'])}")
    print(f"     Tree ID: {split_info['tree_id']}")
    print(f"**************************************************\n")

def test_hoeffding_tree():
    print("🌳 HOEFFDING TREE CLASSIFIER DEBUG TEST")
    print("=" * 60)
    
    try:
        # Try to import the modified classifier
        from river.tree import HoeffdingTreeClassifier
        print("✅ Successfully imported HoeffdingTreeClassifier")

        notification_count = 0

        def count_callback(leaf_info):
            nonlocal notification_count
            notification_count += 1
            instances = leaf_info['total_instances_reached']
            print(f"   Notification {notification_count}: {instances} instances reached")
            # Print node id
            print(f"      Node ID: {leaf_info['node_id']}")
        
        # Create classifier with debug callback
        clf = HoeffdingTreeClassifier(
            grace_period=3,   # Will not check for a split until it sees 10 samples
            delta=1e-7,        # Standard confidence level for splitting
            split_criterion='info_gain',
            leaf_update_callback=count_callback,
            leaf_update_threshold=50,
            nominal_attributes=['education']
        )
        
        print(f"🔧 Created classifier with grace_period={clf.grace_period}")
        print(f"   Split callback enabled: {clf.split_callback is not None}")
        
        # Generate simple training data
        gen = MockSynth()
        # Train with enough instances to pass the grace period
        num_instances_to_train = 2000
        data = list(gen.take(num_instances_to_train))
        
        print(f"\n📊 Training with {len(data)} instances...")
        
        # Train the model
        for i, (x, y) in enumerate(data):
            print(f"\n{'='*50}")
            print(f"TRAINING INSTANCE {i+1}/{num_instances_to_train} | x={x}, y={y}")
            print(f"{'='*50}")
            
            clf.learn_one(x, y)
            
            # Make a prediction to see traversal logic in debug output
            if i >= clf.grace_period:
                print(f"\n--- Making prediction on the same instance to see tree traversal ---")
                pred = clf.predict_proba_one(x)
                
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"Final tree statistics:")
        print(f"  - Total weight seen: {clf._train_weight_seen_by_model}")
        print(f"  - Active leaves (branches): {clf._n_active_leaves}")
        print(f"  - Inactive leaves: {getattr(clf, '_n_inactive_leaves', 0)}")
        print(f"  - Tree depth: {clf.height}")
        
        # Test final predictions
        print(f"\n🔮 FINAL PREDICTIONS TEST:")
        test_instances = [
            {'age': 40, 'income': 85000, 'education': 'high'},   # Should be class 1
            {'age': 25, 'income': 30000, 'education': 'low'}    # Should be class 0
        ]
        
        for test_x in test_instances:
            print(f"\n" + "="*40)
            print(f"Predicting for instance: {test_x}")
            pred_proba = clf.predict_proba_one(test_x)
            prediction = max(pred_proba, key=pred_proba.get)
            print(f"Predicted probabilities: {pred_proba}")
            print(f"Final Prediction: Class {prediction}")
        

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        print("   This might be expected if 'river' is not installed.")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hoeffding_tree()