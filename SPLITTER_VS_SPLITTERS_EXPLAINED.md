# Splitter vs Splitters: Understanding the Difference

## Question
> "Why does `update_model._node_registry.get(0).splitter._att_dist_per_class` return `{}`?"

## Answer
Because you're accessing the **template** (`splitter`) instead of the **actual data** (`splitters`).

---

## The Two Different Attributes

### 1. `node.splitter` (singular) - The TEMPLATE
**What it is:** A shared template instance used for cloning  
**Purpose:** Configuration/factory pattern  
**Data:** Empty - has NO learned distributions  
**Type:** Single `GaussianSplitter` instance

```python
node.splitter                # Template object
├─ _att_dist_per_class: {}   # ← Always empty!
├─ _min_per_class: {}        # ← Always empty!
├─ _max_per_class: {}        # ← Always empty!
└─ n_splits: 10              # Configuration parameter
```

**When created:**
```python
# In HoeffdingTreeClassifier.__init__()
self.splitter = GaussianSplitter()  # Template for all nodes

# In _new_leaf()
leaf = LeafNaiveBayesAdaptive(stats, depth, self.splitter)  # Pass template
```

### 2. `node.splitters` (plural) - The ACTUAL DATA
**What it is:** Dictionary mapping feature names to feature-specific splitters  
**Purpose:** Store learned distributions for each feature  
**Data:** Contains actual Gaussian distributions  
**Type:** `dict[str, GaussianSplitter]`

```python
node.splitters                      # Dictionary of splitters
├─ 'salary': GaussianSplitter       # Feature-specific splitter
│  ├─ _att_dist_per_class:          # ← HAS DATA!
│  │  ├─ 0: Gaussian(μ=86890, σ=37324)
│  │  └─ 1: Gaussian(μ=86570, σ=39128)
│  ├─ _min_per_class: {0: 22174, 1: 20682}
│  └─ _max_per_class: {0: 149988, 1: 149652}
├─ 'age': GaussianSplitter          # Another feature
│  └─ _att_dist_per_class: {0: ..., 1: ...}
└─ ... (more features)
```

**When created/populated:**
```python
# In Leaf.learn_one() - called for each training sample
for att_idx, att_val in x.items():
    if att_idx not in self.splitters:
        # First time seeing this feature - clone template
        self.splitters[att_idx] = self.splitter.clone()
    
    # Update the distribution
    self.splitters[att_idx].update(att_val, y, w)
```

---

## Example from Your Notebook

### ❌ WRONG: Accessing Template
```python
node_0 = update_model._node_registry.get(0)
node_0.splitter._att_dist_per_class
# Returns: {}  ← Empty! No data!
```

**Why it's empty:**
- `node.splitter` is just the template
- It never gets updated with data
- It's only used for cloning

### ✅ CORRECT: Accessing Actual Data
```python
node_0 = update_model._node_registry.get(0)

# Access specific feature's splitter
node_0.splitters['salary']._att_dist_per_class
# Returns: {1: 𝒩(μ=86570.484, σ=39128.275), 
#           0: 𝒩(μ=86890.263, σ=37324.100)}

# Loop through all features
for feature_name, feature_splitter in node_0.splitters.items():
    print(f"{feature_name}: {feature_splitter._att_dist_per_class}")
```

---

## Why This Design?

### Memory Efficiency
- One template shared by all nodes (saves memory)
- Each node only stores feature-specific data

### Lazy Initialization
- `splitters` dict starts empty
- Only creates splitter for a feature when first seen
- Saves memory for sparse features

### Separation of Concerns
- **Template** (`splitter`): Configuration
- **Data** (`splitters`): Learned statistics

---

## Code Flow Example

```python
# 1. Model initialization
model = HoeffdingTreeClassifier()
model.splitter = GaussianSplitter()  # Template created

# 2. First leaf creation
leaf = model._new_leaf()
leaf.splitter = model.splitter       # ← Template (empty)
leaf.splitters = {}                  # ← Empty dict

# 3. First sample arrives: x = {'age': 25, 'salary': 50000}, y = 1
leaf.learn_one(x, y)
# In learn_one():
#   for feature in ['age', 'salary']:
#     leaf.splitters[feature] = leaf.splitter.clone()  # Clone template
#     leaf.splitters[feature].update(value, y)         # Add data

# Result after first sample:
leaf.splitter._att_dist_per_class      # Still {}
leaf.splitters['age']._att_dist_per_class   # {1: Gaussian(...)}
leaf.splitters['salary']._att_dist_per_class # {1: Gaussian(...)}

# 4. Second sample: x = {'age': 45, 'salary': 80000}, y = 0
leaf.learn_one(x, y)

# Result after second sample:
leaf.splitter._att_dist_per_class      # Still {}
leaf.splitters['age']._att_dist_per_class   # {1: G(...), 0: G(...)}
leaf.splitters['salary']._att_dist_per_class # {1: G(...), 0: G(...)}
```

---

## When to Use Each

### Use `node.splitter` when:
- ✅ Creating new leaves (need template for cloning)
- ✅ Checking configuration (n_splits, etc.)
- ✅ Reconstructing tree from JSON (pass template to new leaves)

### Use `node.splitters[feature]` when:
- ✅ Accessing learned distributions
- ✅ Evaluating split candidates
- ✅ Serializing leaf state for updates
- ✅ Making Naive Bayes predictions

---

## Impact on Your Code

### In `verify_split_only.py`:
```python
# CORRECT ✅
leaf = LeafNaiveBayesAdaptive(
    stats=stats, 
    depth=depth, 
    splitter=inference_model.splitter  # Pass template
)
# After reconstruction, leaf.splitters will be {} (empty)
# Will build up as learning continues
```

### In `capture_split_and_updates.py`:
```python
# When capturing leaf updates:
leaf_data = {
    'stats': node.stats,
    'splitters': {}  # Dictionary to populate
}

# CORRECT ✅: Loop through actual splitters, not template
for feat_name, feat_splitter in node.splitters.items():
    leaf_data['splitters'][feat_name] = {
        'distributions': feat_splitter._att_dist_per_class,  # Has data!
        'min_per_class': feat_splitter._min_per_class,
        'max_per_class': feat_splitter._max_per_class
    }
```

---

## Summary

| Attribute | Type | Purpose | Has Data? | When to Use |
|-----------|------|---------|-----------|-------------|
| `node.splitter` | Single GaussianSplitter | Template for cloning | ❌ No (always empty) | Configuration, creating new splitters |
| `node.splitters` | dict[str, GaussianSplitter] | Learned distributions | ✅ Yes (populated by learning) | Accessing distributions, making predictions |

**Key Takeaway:** 
- `splitter` (singular) = Template (empty)
- `splitters` (plural) = Actual data (populated)

Always use `node.splitters[feature_name]` to access learned distributions! 🎯
