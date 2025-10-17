# 📊 SPLIT EVENT: Data Requirements for Distribution

## 🎯 The Problem

When a split happens in the **training tree**, we need to send enough information to the **inference tree** so it can replicate the exact same split structure.

---

## 🔍 What Happens During a Split?

Looking at `_attempt_to_split()` in `hoeffding_tree_classifier.py`:

1. **Original Leaf** gets replaced by a **Split Node** (branch)
2. Split Node has 2+ **New Leaf Children**
3. Original leaf is removed from node registry
4. Split node and new leaves are registered with unique IDs

---

## 📦 CRITICAL DATA TO CONVEY

### 1️⃣ **Split Node Identity**
```python
{
    'node_id': 123,                    # Unique ID of the new split node
    'node_type': 'NumericBinaryBranch', # Type of branch created
    'original_leaf_id': 42,            # ID of leaf that was replaced
}
```

### 2️⃣ **Split Decision Parameters**

The split node needs these parameters to route instances correctly:

#### For **NumericBinaryBranch** (most common):
```python
{
    'feature': 'feature_0',            # Which feature to split on
    'threshold': 0.567,                # Split point (value <= threshold → left)
    'depth': 1,                        # Depth in tree
}
```

#### For **NominalBinaryBranch**:
```python
{
    'feature': 'category_feature',
    'value': 'A',                      # (feature == value → left, else → right)
    'depth': 1,
}
```

#### For **NumericMultiwayBranch**:
```python
{
    'feature': 'feature_0',
    'radius': 0.5,                     # Divides range into slots
    'slot_ids': [0, 1, 2, 3],         # List of slot IDs
    'depth': 1,
}
```

#### For **NominalMultiwayBranch**:
```python
{
    'feature': 'category_feature',
    'feature_values': ['A', 'B', 'C'], # One branch per category
    'depth': 1,
}
```

### 3️⃣ **Split Node Statistics**
```python
{
    'stats': {                         # Class distribution at split point
        '0.0': 6.0,
        '1.0': 4.0
    }
}
```

### 4️⃣ **New Leaf Children**

For **each** new leaf created (usually 2 for binary split):

```python
{
    'node_id': 124,                    # Unique ID for this leaf
    'node_type': 'LeafNaiveBayesAdaptive',  # Type of leaf
    'depth': 2,                        # Depth (parent_depth + 1)
    'branch_index': 0,                 # Which branch (0=left, 1=right for binary)
    
    # Initial statistics from split
    'stats': {
        '0.0': 3.0,                    # Class distribution for this branch
        '1.0': 2.0
    },
    'total_weight': 5.0,
    
    # Naive Bayes weights
    'mc_correct_weight': 0.0,
    'nb_correct_weight': 0.0,
    
    # Splitter data (feature distributions)
    'splitters': {
        'feature_0': {
            'type': 'GaussianSplitter',
            'gaussian_data': {
                'distributions': {
                    '0.0': {'n_samples': 3.0, 'mu': 0.234, 'sigma': 0.123},
                    '1.0': {'n_samples': 2.0, 'mu': 0.789, 'sigma': 0.098}
                },
                'min_per_class': {'0.0': 0.1, '1.0': 0.5},
                'max_per_class': {'0.0': 0.4, '1.0': 0.9}
            }
        },
        'feature_1': {
            'type': 'NominalSplitter',
            'nominal_data': {
                'class_distributions': {
                    '0.0': {'A': 2, 'B': 1},
                    '1.0': {'A': 1, 'B': 1}
                },
                'unique_values': ['A', 'B'],
                'total_weight': 5.0
            }
        }
    }
}
```

### 5️⃣ **Parent Context**
```python
{
    'parent_node_id': 100,            # ID of parent (None if root)
    'parent_branch': 1,               # Which branch of parent leads to this split
}
```

---

## 🏗️ Complete Split Event Payload Structure

```python
split_payload = {
    'event_type': 'split',
    'timestamp': 1697500000.0,
    'tree_id': id(tree),
    
    # What got replaced
    'original_leaf_id': 42,
    
    # New split node created
    'split_node': {
        'node_id': 123,
        'node_type': 'NumericBinaryBranch',  # or other branch type
        'depth': 1,
        'stats': {'0.0': 6.0, '1.0': 4.0},
        
        # Branch-specific parameters
        'branch_params': {
            'feature': 'feature_0',
            'threshold': 0.567,        # For NumericBinaryBranch
            # or 'value': 'A',         # For NominalBinaryBranch
            # or 'radius': 0.5,        # For NumericMultiwayBranch
            # or 'feature_values': [...] # For NominalMultiwayBranch
        }
    },
    
    # New leaf children
    'new_leaves': [
        {
            'node_id': 124,
            'node_type': 'LeafNaiveBayesAdaptive',
            'depth': 2,
            'branch_index': 0,  # Left child
            'stats': {'0.0': 3.0, '1.0': 2.0},
            'total_weight': 5.0,
            'mc_correct_weight': 0.0,
            'nb_correct_weight': 0.0,
            'splitters': { ... }  # Complete splitter data
        },
        {
            'node_id': 125,
            'node_type': 'LeafNaiveBayesAdaptive',
            'depth': 2,
            'branch_index': 1,  # Right child
            'stats': {'0.0': 3.0, '1.0': 2.0},
            'total_weight': 5.0,
            'mc_correct_weight': 0.0,
            'nb_correct_weight': 0.0,
            'splitters': { ... }  # Complete splitter data
        }
    ],
    
    # Parent context
    'parent_context': {
        'parent_node_id': 100,  # None if replacing root
        'parent_branch': 1,     # None if replacing root
    }
}
```

---

## 🔧 How to Apply This on Inference Side

```python
def apply_split_update(inference_tree, split_payload):
    """
    Apply a split update from training tree to inference tree.
    """
    
    # 1. Find the original leaf in inference tree
    original_leaf_id = split_payload['original_leaf_id']
    original_leaf = inference_tree.get_node_by_id(original_leaf_id)
    
    if original_leaf is None:
        print(f"❌ Original leaf {original_leaf_id} not found!")
        return False
    
    # 2. Create the new split node
    split_info = split_payload['split_node']
    branch_type = split_info['node_type']
    branch_params = split_info['branch_params']
    
    # Instantiate the correct branch type
    if branch_type == 'NumericBinaryBranch':
        from river.tree.nodes.branch import NumericBinaryBranch
        
        # Create placeholder leaves first
        left_leaf = create_leaf_from_payload(split_payload['new_leaves'][0])
        right_leaf = create_leaf_from_payload(split_payload['new_leaves'][1])
        
        new_split = NumericBinaryBranch(
            stats=split_info['stats'],
            feature=branch_params['feature'],
            threshold=branch_params['threshold'],
            depth=split_info['depth'],
            left=left_leaf,
            right=right_leaf
        )
    
    # ... similar for other branch types ...
    
    # 3. Register the new split node with the SAME node_id
    inference_tree._register_node(new_split, node_id=split_info['node_id'])
    
    # 4. Register new leaves with their node_ids
    for leaf_payload in split_payload['new_leaves']:
        leaf = new_split.children[leaf_payload['branch_index']]
        inference_tree._register_node(leaf, node_id=leaf_payload['node_id'])
    
    # 5. Replace in parent or set as root
    parent_info = split_payload['parent_context']
    if parent_info['parent_node_id'] is None:
        inference_tree._root = new_split
    else:
        parent = inference_tree.get_node_by_id(parent_info['parent_node_id'])
        parent.children[parent_info['parent_branch']] = new_split
    
    # 6. Remove old leaf from registry
    inference_tree.remove_node_from_registry(original_leaf_id)
    
    return True
```

---

## ⚠️ Critical Notes

1. **Node IDs Must Match**: Training and inference trees must use the same node IDs
2. **Order Matters**: Apply split updates in the order they occur in training
3. **Leaf Statistics**: New leaves inherit portions of parent's statistics based on split
4. **Splitter State**: Each new leaf gets complete splitter data from split decision
5. **Branch Type Detection**: Infer branch type from parameters (numeric vs nominal, binary vs multiway)

---

## 🎯 Summary

**Minimum Required Data for Split Replication:**

✅ Original leaf ID (what's being replaced)  
✅ New split node ID + type + parameters (feature, threshold/value)  
✅ Split node statistics  
✅ New leaf children (2+ leaves with complete state)  
✅ Parent context (where to attach the split)  

With this data, the inference tree can **perfectly reconstruct** the split that occurred in the training tree! 🚀
