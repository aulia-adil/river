# Refactored to Use `apply_distributed_update`

## What Changed

Refactored `inference_from_json.py` to use the built-in `apply_distributed_update()` method instead of custom `_update_splitters()` logic.

---

## Before (Custom Implementation)

```python
def apply_leaf_update(self, update_data):
    # ... manual stats update ...
    node.stats = new_stats
    
    # ... manual weights update ...
    node._mc_correct_weight = ...
    node._nb_correct_weight = ...
    
    # ... manual splitters update ...
    self._update_splitters(node, leaf_data['splitters'])

def _update_splitters(self, node, splitters_data):
    # Custom logic to update Gaussian distributions
    for feature_name, splitter_info in splitters_data.items():
        # ... 50+ lines of custom code ...
        splitter._att_dist_per_class[class_label] = Gaussian._from_state(...)
```

**Problems:**
- ❌ Duplicate code (same logic exists in HoeffdingTree)
- ❌ Harder to maintain
- ❌ Bypasses built-in validation
- ❌ Missing features of the official method

---

## After (Using Built-in Method)

```python
def apply_leaf_update(self, update_data):
    node_id = update_data['node_id']
    leaf_data = update_data['leaf_data']
    
    # Get node
    node = self.model.get_node_by_id(node_id)
    
    # CRITICAL: Create splitters if they don't exist
    if 'splitters' in leaf_data and leaf_data['splitters']:
        if not hasattr(node, 'splitters') or node.splitters is None:
            node.splitters = {}
        
        # Clone splitters from template for missing features
        for feature_name in leaf_data['splitters'].keys():
            if feature_name not in node.splitters:
                node.splitters[feature_name] = self.model.splitter.clone()
    
    # Build update payload
    update_payload = {
        'update_type': 'complete_node',
        'data': {
            'leaf_stats': {
                'stats': {int(k): v for k, v in leaf_data['stats'].items()},
                'total_weight': leaf_data.get('total_weight', 0)
            },
            'naive_bayes_data': {
                'mc_correct_weight': leaf_data.get('mc_correct_weight', 0),
                'nb_correct_weight': leaf_data.get('nb_correct_weight', 0)
            },
            'splitter_data': {
                'splitters': leaf_data.get('splitters', {})
            }
        }
    }
    
    # Use built-in method
    success = self.model.apply_distributed_update(node_id, update_payload)
```

**Benefits:**
- ✅ Uses official River implementation
- ✅ Single source of truth
- ✅ Automatic validation and error handling
- ✅ Consistent with training process
- ✅ Gets future improvements for free
- ✅ Shorter, cleaner code

---

## Key Insight: Pre-create Splitters

The critical step that makes this work:

```python
# BEFORE calling apply_distributed_update, ensure splitters exist
for feature_name in leaf_data['splitters'].keys():
    if feature_name not in node.splitters:
        # Clone from template
        node.splitters[feature_name] = self.model.splitter.clone()
```

**Why this is necessary:**
1. After a split, new leaves have **empty** `splitters` dict: `{}`
2. `apply_distributed_update` expects splitters to **already exist**
3. It only updates existing splitters, doesn't create them
4. So we clone from the template first

---

## What `apply_distributed_update` Does

### For `update_type='complete_node'`:

1. **Leaf Stats Update** (`_apply_leaf_stats_update`):
   - Synchronizes `node.stats` (replaces, doesn't accumulate)
   - Validates `total_weight` matches
   
2. **Splitter Data Update** (`_apply_splitter_data_update`):
   - Updates Gaussian distributions using `Gaussian._from_state()`
   - Synchronizes `_att_dist_per_class`
   - Updates `_min_per_class` and `_max_per_class`
   - Handles class label conversions (string ↔ int)
   
3. **Naive Bayes Update** (`_apply_naive_bayes_update`):
   - Updates `_mc_correct_weight`
   - Updates `_nb_correct_weight`

---

## Update Payload Format

The format expected by `apply_distributed_update`:

```json
{
  "update_type": "complete_node",
  "timestamp": 1761140918.68,
  "data": {
    "leaf_stats": {
      "stats": {
        "1": 281.22,
        "0": 216.0
      },
      "total_weight": 497.22
    },
    "naive_bayes_data": {
      "mc_correct_weight": 0.0,
      "nb_correct_weight": 0.0
    },
    "splitter_data": {
      "splitters": {
        "salary": {
          "type": "GaussianSplitter",
          "feature_name": "salary",
          "gaussian_data": {
            "distributions": {
              "0": {
                "n_samples": 1.0,
                "mu": 130241.37,
                "sigma": 0.0
              }
            },
            "min_per_class": {"0": 130241.37},
            "max_per_class": {"0": 130241.37}
          }
        }
      }
    }
  }
}
```

**This matches exactly what `capture_split_and_updates.py` produces!**

---

## Test Results

### Before Refactor:
- Custom `_update_splitters()` method
- ~50 lines of manual Gaussian reconstruction code
- Worked but duplicated River internals

### After Refactor:
- Uses `apply_distributed_update()`
- ~30 lines (much cleaner)
- **Same results**: 70% accuracy on 10 test samples

```
✅ Successfully applied update to node 1
✅ Successfully applied update to node 2
...
📊 PREDICTION SUMMARY
Total samples: 10
Correct predictions: 7
Accuracy: 70.00%
```

---

## Code Comparison

### Lines of Code:
- **Before**: ~90 lines (`apply_leaf_update` + `_update_splitters`)
- **After**: ~40 lines (`apply_leaf_update` only)
- **Reduction**: 55% less code

### Complexity:
- **Before**: Manual Gaussian state management, class label conversion, min/max updates
- **After**: Build payload dict, call one method

### Maintainability:
- **Before**: If River updates `_from_state()` logic, we need to update our code too
- **After**: Automatically gets River improvements

---

## Usage Pattern

This pattern can be reused anywhere you need to apply distributed updates:

```python
# 1. Get node
node = model.get_node_by_id(node_id)

# 2. Pre-create splitters if needed
if node.splitters is None:
    node.splitters = {}
for feature_name in update_data['splitters'].keys():
    if feature_name not in node.splitters:
        node.splitters[feature_name] = model.splitter.clone()

# 3. Build payload
update_payload = {
    'update_type': 'complete_node',  # or 'leaf_stats', 'splitter_data', etc.
    'data': {
        'leaf_stats': {...},
        'splitter_data': {...},
        'naive_bayes_data': {...}
    }
}

# 4. Apply
success = model.apply_distributed_update(node_id, update_payload)
```

---

## Summary

| Aspect | Custom | Built-in Method |
|--------|--------|----------------|
| Code lines | ~90 | ~40 |
| Complexity | High | Low |
| Maintainability | Manual sync needed | Automatic |
| Future-proof | No | Yes |
| Validation | Manual | Built-in |
| Error handling | Custom | Built-in |
| Test results | 70% accuracy | 70% accuracy ✅ |

**Recommendation:** Always use `apply_distributed_update()` instead of manually updating node internals. It's the official way to synchronize distributed trees in River! 🎯
