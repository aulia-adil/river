"""
Node Data Structure Inspector for Hoeffding Tree
This will show you exactly what data is inside each node type.
"""

def inspect_node_callback(split_info):
    """
    Callback that thoroughly inspects node data structures.
    This shows you exactly what you'll need to serialize for Kafka.
    """
    print(f"\n🔬 DEEP NODE INSPECTION FOR KAFKA SERIALIZATION")
    print("=" * 60)
    
    # 1. Original Leaf Node Analysis
    original_leaf = split_info['original_leaf']
    print(f"📄 ORIGINAL LEAF NODE STRUCTURE:")
    print(f"   Class: {type(original_leaf).__module__}.{type(original_leaf).__name__}")
    
    # Get all attributes
    leaf_attrs = {}
    for attr in dir(original_leaf):
        if not attr.startswith('_') and not callable(getattr(original_leaf, attr)):
            try:
                value = getattr(original_leaf, attr)
                leaf_attrs[attr] = value
            except:
                leaf_attrs[attr] = "Cannot access"
    
    print(f"   Attributes: {list(leaf_attrs.keys())}")
    for key, value in leaf_attrs.items():
        if key in ['stats', 'depth', 'total_weight', 'last_split_attempt_at']:
            print(f"     {key}: {value}")
    
    # 2. New Split Node Analysis
    new_split = split_info['new_split_node']
    print(f"\n🌿 NEW SPLIT NODE STRUCTURE:")
    print(f"   Class: {type(new_split).__module__}.{type(new_split).__name__}")
    
    split_attrs = {}
    for attr in dir(new_split):
        if not attr.startswith('_') and not callable(getattr(new_split, attr)):
            try:
                value = getattr(new_split, attr)
                split_attrs[attr] = value
            except:
                split_attrs[attr] = "Cannot access"
    
    print(f"   Attributes: {list(split_attrs.keys())}")
    for key, value in split_attrs.items():
        if key in ['feature', 'threshold', 'depth', 'stats', 'children']:
            if key == 'children':
                print(f"     {key}: [{len(value)} children] - {[type(c).__name__ for c in value]}")
            else:
                print(f"     {key}: {value}")
    
    # 3. New Leaf Nodes Analysis
    print(f"\n🌱 NEW LEAF NODES STRUCTURE:")
    for i, new_leaf in enumerate(split_info['new_leaves']):
        print(f"   Leaf {i}:")
        print(f"     Class: {type(new_leaf).__module__}.{type(new_leaf).__name__}")
        
        new_leaf_attrs = {}
        for attr in dir(new_leaf):
            if not attr.startswith('_') and not callable(getattr(new_leaf, attr)):
                try:
                    value = getattr(new_leaf, attr)
                    new_leaf_attrs[attr] = value
                except:
                    new_leaf_attrs[attr] = "Cannot access"
        
        for key, value in new_leaf_attrs.items():
            if key in ['stats', 'depth', 'total_weight', 'parent']:
                if key == 'parent':
                    print(f"       {key}: {type(value).__name__ if value else None}")
                else:
                    print(f"       {key}: {value}")
    
    # 4. Split Decision Analysis
    split_decision = split_info['split_decision']
    print(f"\n📋 SPLIT DECISION STRUCTURE:")
    print(f"   Class: {type(split_decision).__module__}.{type(split_decision).__name__}")
    
    decision_attrs = {}
    for attr in dir(split_decision):
        if not attr.startswith('_') and not callable(getattr(split_decision, attr)):
            try:
                value = getattr(split_decision, attr)
                decision_attrs[attr] = value
            except:
                decision_attrs[attr] = "Cannot access"
    
    print(f"   Attributes: {list(decision_attrs.keys())}")
    for key, value in decision_attrs.items():
        if key in ['feature', 'merit', 'threshold', 'children_stats', 'numerical_feature', 'multiway_split']:
            print(f"     {key}: {value}")
    
    # 5. What You Need to Serialize for Kafka
    print(f"\n📦 SERIALIZATION REQUIREMENTS FOR KAFKA:")
    print("=" * 45)
    
    serializable_data = {
        'tree_id': split_info['tree_id'],
        'split_type': split_info['split_type'],
        'split_feature': split_info['split_feature'],
        'parent_branch': split_info['parent_branch'],
        
        # Original leaf data
        'original_leaf': {
            'type': type(original_leaf).__name__,
            'depth': getattr(original_leaf, 'depth', None),
            'stats': getattr(original_leaf, 'stats', {}),
            'total_weight': getattr(original_leaf, 'total_weight', 0)
        },
        
        # New split node data
        'new_split_node': {
            'type': type(new_split).__name__,
            'feature': getattr(new_split, 'feature', None),
            'threshold': getattr(new_split, 'threshold', None),
            'depth': getattr(new_split, 'depth', None),
            'stats': getattr(new_split, 'stats', {}),
            'max_branches': new_split.max_branches() if hasattr(new_split, 'max_branches') else None
        },
        
        # New leaf nodes data
        'new_leaves': [
            {
                'type': type(leaf).__name__,
                'depth': getattr(leaf, 'depth', None),
                'stats': getattr(leaf, 'stats', {}),
                'total_weight': getattr(leaf, 'total_weight', 0)
            }
            for leaf in split_info['new_leaves']
        ],
        
        # Split decision data
        'split_decision': {
            'feature': getattr(split_decision, 'feature', None),
            'merit': getattr(split_decision, 'merit', None),
            'threshold': getattr(split_decision, 'threshold', None),
            'numerical_feature': getattr(split_decision, 'numerical_feature', None),
            'multiway_split': getattr(split_decision, 'multiway_split', None),
            'children_stats': getattr(split_decision, 'children_stats', None)
        }
    }
    
    print("MINIMAL SERIALIZABLE DATA STRUCTURE:")
    import json
    try:
        # Try to JSON serialize to see what's serializable
        json_str = json.dumps(serializable_data, indent=2, default=str)
        print(json_str[:1000] + "..." if len(json_str) > 1000 else json_str)
    except Exception as e:
        print(f"JSON serialization issue: {e}")
        print("Raw structure:")
        for key, value in serializable_data.items():
            print(f"  {key}: {type(value)} = {str(value)[:100]}...")
    
    print(f"\n✅ This is the data you'll send over Kafka!")
    return True


def test_node_inspection():
    """Test with a simple example to see node structures."""
    print("🧪 NODE STRUCTURE INSPECTION TEST")
    print("=" * 40)
    
    try:
        # Since we can't import river due to dependencies, 
        # let's just show what the callback structure would look like
        print("📋 Expected callback structure when a split occurs:")
        
        example_callback_info = {
            'split_type': 'node_split',
            'tree_id': 140234567890,
            'split_feature': 'age',
            'parent_branch': 0,
            'original_leaf': "LeafNaiveBayesAdaptive instance",
            'new_split_node': "NumericBinaryBranch instance", 
            'new_leaves': ["LeafNaiveBayesAdaptive instance", "LeafNaiveBayesAdaptive instance"],
            'split_decision': "SplitSuggestion instance"
        }
        
        print("Key components you'll be serializing:")
        for key, value in example_callback_info.items():
            print(f"  {key}: {value}")
            
        print("\n🔑 Critical data for inference processes:")
        critical_data = [
            "Split feature name and threshold",
            "Branch type (NumericBinary, NominalMultiway, etc.)",
            "Node depth and position in tree",
            "Class statistics from original leaf",
            "Initial statistics for new leaf nodes",
            "Parent-child relationships"
        ]
        
        for item in critical_data:
            print(f"  • {item}")
            
        print("\n💡 For your thesis implementation:")
        print("   1. Extract these key attributes from the callback")
        print("   2. Serialize them (JSON, pickle, protobuf, etc.)")
        print("   3. Send via Kafka to inference processes") 
        print("   4. Inference processes reconstruct tree nodes")
        print("   5. Apply the split to their local tree copies")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_node_inspection()