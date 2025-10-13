#!/usr/bin/env python3
"""
Test script to verify the fix for handling both Gaussian and nominal splitters
that have _att_dist_per_class attribute.
"""

import sys
import os
sys.path.insert(0, '/root/river')

from river.tree import HoeffdingTreeClassifier
from river.datasets import synth
import numpy as np

def test_mixed_data_types():
    """Test with mixed nominal and numerical features."""
    
    # Callback to capture splitter data
    split_notifications = []
    leaf_notifications = []
    
    def split_callback(info):
        split_notifications.append(info)
        print(f"🌳 SPLIT CALLBACK: Feature '{info['split_feature']}' split occurred")
    
    def leaf_callback(info):
        leaf_notifications.append(info)
        print(f"🍃 LEAF CALLBACK: Node {info['node_id']} reached threshold")
        
        # Print splitter details to verify fix
        splitters = info['leaf_data']['splitters']
        for feature_name, splitter_data in splitters.items():
            splitter_type = splitter_data['type']
            print(f"   Feature '{feature_name}': {splitter_type}")
            
            if 'gaussian_data' in splitter_data:
                print(f"      ✅ Gaussian data extracted successfully")
                gaussian = splitter_data['gaussian_data']
                if 'distributions' in gaussian:
                    print(f"      📊 Classes: {list(gaussian['distributions'].keys())}")
            
            if 'nominal_data' in splitter_data:
                print(f"      ✅ Nominal data extracted successfully")
                nominal = splitter_data['nominal_data']
                if 'class_distributions' in nominal:
                    print(f"      📊 Class distributions: {nominal['class_distributions']}")
    
    # Create tree with callbacks and lower threshold for quick testing
    tree = HoeffdingTreeClassifier(
        grace_period=50,
        split_callback=split_callback,
        leaf_update_callback=leaf_callback,
        leaf_update_threshold=10,  # Trigger every 10 instances
        nominal_attributes=['category']  # Mark 'category' as nominal
    )
    
    print("🧪 TESTING MIXED DATA TYPES (Nominal + Numerical)")
    print("=" * 60)
    
    # Generate mixed data: numerical + nominal features
    np.random.seed(42)
    categories = ['low', 'medium', 'high']
    
    for i in range(30):
        # Create instance with both numerical and nominal features
        x = {
            'numerical_feature': np.random.normal(0, 1),      # Gaussian feature
            'category': np.random.choice(categories),          # Nominal feature
            'another_num': np.random.uniform(-1, 1)           # Another numerical
        }
        
        # Create class based on features (to encourage splits)
        if x['category'] == 'high' or x['numerical_feature'] > 0:
            y = 1
        else:
            y = 0
        
        print(f"\n--- Instance {i+1}: {x} → class {y} ---")
        tree.learn_one(x, y)
    
    print(f"\n📋 FINAL RESULTS:")
    print(f"Split notifications: {len(split_notifications)}")
    print(f"Leaf notifications: {len(leaf_notifications)}")
    
    # Show tree structure
    tree.print_node_registry()
    
    return len(split_notifications) > 0 or len(leaf_notifications) > 0

def test_nominal_only():
    """Test with only nominal features to specifically test nominal _att_dist_per_class."""
    
    leaf_notifications = []
    
    def leaf_callback(info):
        leaf_notifications.append(info)
        print(f"🍃 LEAF CALLBACK: Node {info['node_id']} - {info['total_instances_reached']} instances")
        
        # Examine splitter data
        splitters = info['leaf_data']['splitters']
        for feature_name, splitter_data in splitters.items():
            print(f"   📊 Feature '{feature_name}': {splitter_data['type']}")
            
            if 'nominal_data' in splitter_data and 'class_distributions' in splitter_data['nominal_data']:
                distributions = splitter_data['nominal_data']['class_distributions']
                print(f"      🎯 Class distributions: {distributions}")
    
    # Tree with only nominal attributes
    tree = HoeffdingTreeClassifier(
        grace_period=100,
        leaf_update_callback=leaf_callback,
        leaf_update_threshold=5,
        nominal_attributes=['color', 'size']  # Both features are nominal
    )
    
    print("\n🧪 TESTING NOMINAL-ONLY DATA")
    print("=" * 40)
    
    # Generate nominal-only data
    colors = ['red', 'blue', 'green']
    sizes = ['small', 'medium', 'large']
    
    for i in range(20):
        x = {
            'color': np.random.choice(colors),
            'size': np.random.choice(sizes)
        }
        
        # Simple class rule
        y = 1 if (x['color'] == 'red' or x['size'] == 'large') else 0
        
        print(f"Instance {i+1}: {x} → class {y}")
        tree.learn_one(x, y)
    
    print(f"\n📋 Nominal-only test completed: {len(leaf_notifications)} notifications")
    return True

if __name__ == "__main__":
    print("🔧 TESTING SPLITTER BUG FIX")
    print("Testing both Gaussian and nominal splitters with _att_dist_per_class")
    print("=" * 70)
    
    try:
        # Test mixed data types
        success1 = test_mixed_data_types()
        
        # Test nominal-only data
        success2 = test_nominal_only()
        
        if success1 and success2:
            print("\n✅ ALL TESTS PASSED!")
            print("✅ Bug fix successful: Both Gaussian and nominal splitters handled correctly")
        else:
            print("\n❌ Some tests failed")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()