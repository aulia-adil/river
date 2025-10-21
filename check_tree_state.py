"""
Check the complete state of the reconstructed tree
"""

import json
from pathlib import Path
from river import tree
from river.tree.nodes.branch import NumericBinaryBranch
from river.tree.nodes.htc_nodes import LeafMajorityClass

def load_and_check():
    """Load split event and check the reconstructed tree state"""
    
    # Load the split event
    split_file = 'json_data/verification/split_event.json'
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    print("="*80)
    print("📄 SPLIT EVENT DATA (from JSON)")
    print("="*80)
    print(f"\nEvent type: {split_data['event_type']}")
    print(f"Original leaf ID: {split_data['original_leaf_id']}")
    print(f"Splitter type: {split_data['splitter_type']}")
    
    print(f"\n🌳 SPLIT NODE (from JSON):")
    split_node = split_data['split_node']
    print(f"   Node ID: {split_node['node_id']}")
    print(f"   Node Type: {split_node['node_type']}")
    print(f"   Depth: {split_node['depth']}")
    print(f"   Stats: {split_node['stats']}")
    print(f"   Feature: {split_node['branch_params']['feature']}")
    print(f"   Threshold: {split_node['branch_params']['threshold']}")
    
    print(f"\n🍃 NEW LEAVES (from JSON):")
    for i, leaf in enumerate(split_data['new_leaves']):
        print(f"\n   Leaf {i} (will be node_id={leaf['node_id']}):")
        print(f"      Branch index: {leaf['branch_index']} ({'LEFT' if leaf['branch_index'] == 0 else 'RIGHT'})")
        print(f"      Node type: {leaf['node_type']}")
        print(f"      Depth: {leaf['depth']}")
        print(f"      Stats: {leaf['stats']}")
        print(f"      Total weight: {leaf['total_weight']}")
    
    # Now reconstruct the tree
    print("\n" + "="*80)
    print("🔧 RECONSTRUCTING TREE")
    print("="*80)
    
    model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        leaf_prediction='nba'
    )
    
    print(f"Initial model: {model.n_nodes} nodes")
    
    # Create leaves
    from river.tree.nodes.htc_nodes import LeafNaiveBayesAdaptive
    
    leaves = []
    for leaf_info in split_data['new_leaves']:
        stats = {int(k): v for k, v in leaf_info['stats'].items()}
        leaf = LeafNaiveBayesAdaptive(
            stats=stats, 
            depth=leaf_info['depth'], 
            splitter=model.splitter  # Use model's splitter template
        )
        leaves.append(leaf)
        print(f"Created leaf: stats={stats}, depth={leaf_info['depth']}, splitter={type(leaf.splitter).__name__}")
    
    # Create split node
    stats_dict = {int(k): v for k, v in split_node['stats'].items()}
    split_node_obj = NumericBinaryBranch(
        stats=stats_dict,
        feature=split_node['branch_params']['feature'],
        threshold=split_node['branch_params']['threshold'],
        depth=split_node['depth'],
        left=leaves[0],
        right=leaves[1]
    )
    
    # Set as root
    model._root = split_node_obj
    
    # Register nodes
    if hasattr(model, '_register_node'):
        model._register_node(split_node_obj, split_node['node_id'])
        for leaf, leaf_info in zip(leaves, split_data['new_leaves']):
            model._register_node(leaf, leaf_info['node_id'])
    
    print(f"\nReconstructed model: {model.n_nodes} nodes, height {model.height}")
    
    # Now inspect the actual tree state
    print("\n" + "="*80)
    print("🔍 ACTUAL TREE STATE (after reconstruction)")
    print("="*80)
    
    root = model._root
    print(f"\n🌳 ROOT NODE:")
    print(f"   Type: {type(root).__name__}")
    print(f"   Node ID: {getattr(root, 'node_id', 'NOT SET')}")
    print(f"   Depth: {root.depth}")
    print(f"   Stats: {root.stats}")
    print(f"   Total weight: {root.total_weight}")
    print(f"   Feature: {root.feature}")
    print(f"   Threshold: {root.threshold}")
    print(f"   Number of children: {len(root.children)}")
    
    # Check each child
    for i, child in enumerate(root.children):
        branch_name = "LEFT" if i == 0 else "RIGHT"
        print(f"\n🍃 {branch_name} CHILD (index {i}):")
        print(f"   Type: {type(child).__name__}")
        print(f"   Node ID: {getattr(child, 'node_id', 'NOT SET')}")
        print(f"   Depth: {child.depth}")
        print(f"   Stats: {child.stats}")
        print(f"   Total weight: {child.total_weight}")
        
        # Check if it has splitter
        if hasattr(child, 'splitter'):
            print(f"   Splitter: {type(child.splitter).__name__ if child.splitter else None}")
        
        # Check if it has splitters dict (for tracking features)
        if hasattr(child, 'splitters'):
            print(f"   Splitters dict: {len(child.splitters) if child.splitters else 0} features tracked")
    
    # Verify correctness
    print("\n" + "="*80)
    print("✅ VERIFICATION")
    print("="*80)
    
    # Check if stats match
    json_split_stats = split_node['stats']
    actual_split_stats = {str(k): v for k, v in root.stats.items()}
    
    print(f"\n🌳 Split node stats match: {json_split_stats == actual_split_stats}")
    print(f"   JSON:   {json_split_stats}")
    print(f"   Actual: {actual_split_stats}")
    
    # Check each leaf
    for i, (leaf_info, actual_leaf) in enumerate(zip(split_data['new_leaves'], root.children)):
        json_stats = leaf_info['stats']
        actual_stats = {str(k): v for k, v in actual_leaf.stats.items()}
        
        match = json_stats == actual_stats
        branch_name = "LEFT" if i == 0 else "RIGHT"
        
        print(f"\n🍃 {branch_name} leaf stats match: {match}")
        print(f"   JSON:   {json_stats}")
        print(f"   Actual: {actual_stats}")
        
        # Check if both have same classes
        json_classes = set(json_stats.keys())
        actual_classes = set(actual_stats.keys())
        
        if json_classes != actual_classes:
            print(f"   ⚠  CLASS MISMATCH!")
            print(f"   JSON has classes: {json_classes}")
            print(f"   Actual has classes: {actual_classes}")
            print(f"   Missing in actual: {json_classes - actual_classes}")
            print(f"   Extra in actual: {actual_classes - json_classes}")
    
    # Check total weights
    print(f"\n📊 Total weights:")
    for i, (leaf_info, actual_leaf) in enumerate(zip(split_data['new_leaves'], root.children)):
        branch_name = "LEFT" if i == 0 else "RIGHT"
        json_weight = leaf_info['total_weight']
        actual_weight = actual_leaf.total_weight
        
        match = abs(json_weight - actual_weight) < 1e-10
        print(f"   {branch_name}: JSON={json_weight:.4f}, Actual={actual_weight:.4f}, Match={match}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    load_and_check()
