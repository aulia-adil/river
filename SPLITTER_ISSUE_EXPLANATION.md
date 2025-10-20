# Splitter Issue: Why Inference Trees Can't Learn After Reconstruction

## The Problem

When reconstructing a Hoeffding Tree from split event JSON data, the reconstructed tree could:
- ✅ Make predictions (inference)
- ❌ Learn from new data (`learn_one()` failed)

### Error Message
```
AttributeError: 'NoneType' object has no attribute 'clone'
```

This occurred in `river/tree/nodes/leaf.py` at line 108 in the `update_splitters()` method.

## Root Cause

### What is a Splitter?

A **splitter** (also called Attribute Observer or AO) is an object that tracks statistics for each feature to determine when and how to split a node. Examples include:
- `GaussianSplitter` - for numeric attributes in classification
- `NominalSplitterClassif` - for categorical attributes

### The Splitter's Role

Each leaf node has:
1. `self.splitter` - A template splitter instance (e.g., `GaussianSplitter`)
2. `self.splitters` - A dictionary `{feature_id: splitter_instance}` tracking each feature

When a leaf observes a new sample via `learn_one()`:
1. It calls `update_splitters(x, y, w, tree.nominal_attributes)`
2. For each feature in the sample:
   - If the feature is already tracked: update existing splitter
   - If it's a new feature: **clone the template splitter** (`self.splitter.clone()`)

### Why Inference Works But Learning Doesn't

- **Inference (prediction)** only reads the `stats` dictionary (class counts) - no splitter needed
- **Learning** must update splitters for every feature to prepare for future splits - requires `self.splitter`

## The Bug in Reconstruction

### Original (Broken) Code

```python
# In reconstruct_and_predict()
leaf = LeafMajorityClass(stats=stats, depth=leaf_info['depth'], splitter=None)
                                                                    # ^^^^^^^^ BUG!
```

This passed `splitter=None`, so when the leaf tried to learn:
```python
# In leaf.py update_splitters()
splitter = self.splitter.clone()  # CRASH! self.splitter is None
```

### How Normal Trees Work

When a tree naturally creates leaves via `_new_leaf()`:
```python
# In hoeffding_tree_classifier.py
def _new_leaf(self, initial_stats=None, parent=None):
    # ...
    leaf = LeafMajorityClass(initial_stats, depth, self.splitter)
                                                    # ^^^^^^^^^^
                                                    # Uses tree's splitter!
    return leaf
```

The tree passes its configured splitter (e.g., `GaussianSplitter`) to every leaf.

### The Fix

```python
# In reconstruct_and_predict()
leaf = LeafMajorityClass(
    stats=stats, 
    depth=leaf_info['depth'], 
    splitter=inference_model.splitter  # ✅ Use the model's splitter!
)
```

Also capture the splitter type in the split event JSON for validation:
```python
splitter_type = type(new_leaves[0].splitter).__name__  # e.g., "GaussianSplitter"
```

## Key Insights

1. **Splitter is configuration, not state**: The splitter template itself (`GaussianSplitter()`) doesn't hold learned data - it's just a factory for creating feature-specific splitters.

2. **Each tree has one splitter template**: All leaves in a tree share the same splitter template type.

3. **Splitter instances are per-feature**: Each feature gets its own splitter instance (via `.clone()`), stored in `leaf.splitters[feature_id]`.

4. **Split events don't need splitter data**: Since the splitter is just configuration (not learned state), we only need to capture its type, not serialize its state.

## Why This Matters for Distributed Learning

In a distributed Hoeffding Tree system:
- **Split events** update tree structure but don't need splitter state
- **Leaf update events** (to be implemented) would transfer learned feature statistics
- The inference nodes must be initialized with proper splitters to continue learning after receiving structure updates

## Summary

The splitter being `None` prevented learning because:
1. Learning requires updating feature statistics via splitters
2. New features trigger cloning the splitter template
3. Can't clone `None`

The solution:
1. Pass the tree's splitter when reconstructing leaves
2. Capture splitter type in split events for validation
3. Ensure all reconstructed leaves have proper splitter templates

Now reconstructed trees can both predict **and** learn! 🎉
