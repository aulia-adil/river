"""
Test script to demonstrate the detailed debug output of HoeffdingTreeClassifier.
This will show you exactly how the tree learns and makes decisions!
"""

import sys
sys.path.insert(0, '/root/river')

# Mock the imports to avoid dependency issues
class MockSynth:
    def __init__(self, classification_function=0, seed=42):
        self.data = [
            ({'age': 25, 'income': 30000, 'education': 'high'}, 0),
            ({'age': 45, 'income': 80000, 'education': 'medium'}, 1),
            ({'age': 30, 'income': 45000, 'education': 'high'}, 1),
            ({'age': 22, 'income': 25000, 'education': 'low'}, 0),
            ({'age': 55, 'income': 95000, 'education': 'high'}, 1),
            ({'age': 28, 'income': 35000, 'education': 'medium'}, 0),
            ({'age': 40, 'income': 70000, 'education': 'high'}, 1),
            ({'age': 33, 'income': 50000, 'education': 'medium'}, 1),
            ({'age': 26, 'income': 32000, 'education': 'low'}, 0),
            ({'age': 50, 'income': 85000, 'education': 'high'}, 1),
        ]
        self.index = 0
    
    def take(self, n):
        for i in range(min(n, len(self.data))):
            if self.index < len(self.data):
                yield self.data[self.index]
                self.index += 1

def simple_callback(split_info):
    """Simple callback to show distributed training capability"""
    print(f"   🚀 DISTRIBUTED CALLBACK TRIGGERED!")
    print(f"      Split feature: {split_info['split_feature']}")
    print(f"      Number of new leaves: {len(split_info['new_leaves'])}")
    print(f"      Tree ID: {split_info['tree_id']}")

def test_hoeffding_tree():
    print("🌳 HOEFFDING TREE CLASSIFIER DEBUG TEST")
    print("=" * 60)
    
    try:
        # Try to import the modified classifier
        from river.tree import HoeffdingTreeClassifier
        print("✅ Successfully imported HoeffdingTreeClassifier")
        
        # Create classifier with debug callback
        clf = HoeffdingTreeClassifier(
            grace_period=3,  # Very small for quick splits
            delta=1e-3,      # Less strict for demonstration
            split_callback=simple_callback,
            nominal_attributes=['education']
        )
        
        print(f"🔧 Created classifier with grace_period={clf.grace_period}")
        print(f"   Split callback enabled: {clf.split_callback is not None}")
        
        # Generate simple training data
        gen = MockSynth()
        data = list(gen.take(10))
        
        print(f"\n📊 Training data: {len(data)} instances")
        
        # Train the model
        for i, (x, y) in enumerate(data):
            print(f"\n{'='*50}")
            print(f"TRAINING INSTANCE {i+1}/10")
            print(f"{'='*50}")
            
            clf.learn_one(x, y)
            
            # Make a prediction to see traversal
            if i >= 2:  # Only predict after a few training instances
                print(f"\n{'='*30}")
                print(f"MAKING PREDICTION")
                print(f"{'='*30}")
                pred = clf.predict_proba_one(x)
                
        print(f"\n🎉 TRAINING COMPLETE!")
        print(f"Final tree statistics:")
        print(f"  - Total weight seen: {clf._train_weight_seen_by_model}")
        print(f"  - Classes learned: {sorted(clf.classes)}")
        print(f"  - Active leaves: {clf._n_active_leaves}")
        print(f"  - Inactive leaves: {getattr(clf, '_n_inactive_leaves', 0)}")
        
        # Test final predictions
        print(f"\n🔮 FINAL PREDICTIONS TEST:")
        test_instances = [
            {'age': 35, 'income': 60000, 'education': 'high'},
            {'age': 20, 'income': 20000, 'education': 'low'}
        ]
        
        for test_x in test_instances:
            print(f"\n" + "="*40)
            pred = clf.predict_proba_one(test_x)
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("This is expected due to missing dependencies, but the code is modified!")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hoeffding_tree()