# Why Is There No Splitter Data in Split Events?

## Question
> "Why is there no splitter data here?" (in the split event JSON)

## Short Answer
**Split events intentionally do NOT contain splitter distribution data** because new leaves have **empty splitters** at the moment of split. The splitter distributions are built incrementally as new samples arrive and are synchronized via **leaf update events**.

---

## The Two Event Types

### 1. Split Event (Structure Change)
**When:** A leaf splits into a branch with new child leaves  
**Purpose:** Communicate tree structure changes  
**Frequency:** Rare (only when splits occur)

**What's Included:**
- ✅ Node IDs (for registry)
- ✅ Branch structure (feature, threshold)
- ✅ Class statistics (`stats`)
- ✅ Splitter type (template name)
- ❌ **NO splitter distributions**

### 2. Leaf Update Event (Statistics Sync)
**When:** Periodically after learning continues  
**Purpose:** Synchronize learned statistics  
**Frequency:** Regular (every N samples)

**What's Included:**
- ✅ Updated class statistics
- ✅ **Splitter distributions** (Gaussian μ, σ, n)
- ✅ Min/max values per class
- ✅ All feature distributions

---

## Proof: Splitters Are Empty at Split Time

### Experiment Result:
```
🔍 CHECKING SPLITTERS IMMEDIATELY AFTER SPLIT

LEFT LEAF (node_id=1):
  Stats: {1: 281.22, 0: 215.0}        ✅ Has stats
  Splitters dict: 0 features          ✅ EMPTY!

RIGHT LEAF (node_id=2):
  Stats: {1: 103.78}                  ✅ Has stats  
  Splitters dict: 0 features          ✅ EMPTY!
```

**Conclusion:** At the exact moment of split, new leaves have:
- Stats: ✅ Populated (from split calculation)
- Splitters: ❌ Empty dictionary `{}`

---

## Why This Happens

### Code Analysis: `_new_leaf()` in HoeffdingTree

```python
def _new_leaf(self, initial_stats=None, parent=None):
    return LeafNaiveBayesAdaptive(
        stats=initial_stats,      # ← Populated from split calculation
        depth=parent.depth + 1,
        splitter=self.splitter    # ← Template only (no data!)
    )
```

**What gets created:**
- `stats`: `{1: 281.22, 0: 215.0}` ← From `split_decision.children_stats`
- `splitters`: `{}` ← Empty dictionary
- `splitter`: `GaussianSplitter()` ← Template for future use

### How Splitters Get Built

Splitters are populated **incrementally** in `Leaf.learn_one()`:

```python
def learn_one(self, x, y, *, w=1.0, tree=None):
    # Update stats
    self.stats[y] = self.stats.get(y, 0.0) + w
    
    # Update splitters (builds distributions incrementally)
    for att_idx, att_val in x.items():
        if att_idx not in self.splitters:
            # First time seeing feature → create splitter
            self.splitters[att_idx] = self.splitter.clone()
        
        # Update Gaussian distribution
        self.splitters[att_idx].update(att_val, y, w)
```

**Timeline:**
- **T=0 (split time):** `splitters = {}` ← Empty
- **T=1 (first sample):** `splitters = {"age": GaussianSplitter(...)}` 
- **T=10 (10 samples later):** All features have distributions → Ready for update event

---

## The Splitter Data You Saw

The splitter data you showed comes from **leaf update events**, not split events:

```json
{
  "event_type": "leaf_update",    ← Different event type!
  "node_id": 1,
  "stats": {...},
  "splitters": {                   ← This is from update events
    "salary": {
      "gaussian_data": {
        "distributions": {
          "0": {"n_samples": 5.0, "mu": 100830.94, "sigma": 50621.71}
        }
      }
    }
  }
}
```

This event is sent **AFTER** the split, once leaves have learned from new samples.

---

## Complete Event Sequence

### Distributed Learning Flow:

```
Training Process                    Inference Process
================                    =================

1. Train samples 1-600
   ↓
2. Split occurs!
   New leaves created with:
   - stats: {1: 281.22, 0: 215.0}
   - splitters: {}  ← EMPTY
   ↓
3. Publish SPLIT EVENT              → Receive SPLIT EVENT
   {                                   ↓
     "stats": {...},                4. Reconstruct tree:
     NO splitters                      - Create split node
   }                                   - Create leaves with stats
                                       - Leaves have empty splitters ✅
                                       
5. Continue learning (601-610)
   Leaves build splitter data:
   splitters = {
     "age": {...},
     "salary": {...}
   }
   ↓
6. Publish LEAF UPDATE              → Receive LEAF UPDATE
   {                                   ↓
     "stats": {...},                7. Update leaf:
     "splitters": {...}  ← NOW         - Update stats
   }                                   - Create splitters from data ✅
                                       
8. Continue learning...
   ↓
9. Next LEAF UPDATE                 → Update again...
```

---

## Why Stats But Not Splitters?

### Stats Are Available at Split Time

**Where stats come from:**
```python
# In GaussianSplitter._class_dists_from_binary_split()
def _class_dists_from_binary_split(self, split_value):
    lhs_dist = {}
    rhs_dist = {}
    
    for k, estimator in self._att_dist_per_class.items():
        # Calculate expected samples per class in each child
        if split_value >= self._max_per_class[k]:
            lhs_dist[k] = estimator.n_samples  # All left
        else:
            lhs_dist[k] = estimator.cdf(split_value) * estimator.n_samples
            rhs_dist[k] = estimator.n_samples - lhs_dist[k]
    
    return [lhs_dist, rhs_dist]  # ← These become the stats!
```

**Result:** `[{1: 281.22, 0: 215.0}, {1: 103.78}]`

**These stats represent:**
- **Probabilistic estimate** of class distribution in each child
- Calculated from **parent's splitter** (which has data)
- Available **immediately** at split time ✅

### Splitters Are NOT Available

**Child splitters are:**
- Built from **actual samples** that arrive at the child
- Start **empty** (`{}`) at split time
- Populated **incrementally** as learning continues
- **NOT available** at split time ❌

---

## What Should Split Events Contain?

### Current Implementation ✅ CORRECT

```json
{
  "event_type": "split",
  "original_leaf_id": 0,
  "splitter_type": "GaussianSplitter",  ← Template type
  "split_node": {
    "node_id": 3,
    "feature": "age",
    "threshold": 63.64,
    "stats": {"1": 385.0, "0": 215.0}   ← Parent stats
  },
  "new_leaves": [
    {
      "node_id": 1,
      "stats": {"1": 281.22, "0": 215.0}  ← Child stats
    }
  ]
}
```

**This contains everything needed to:**
1. ✅ Create the split node structure
2. ✅ Create leaves with correct stats
3. ✅ Know what splitter template to use
4. ✅ Make Naive Bayes predictions (needs stats only)

**What's NOT needed:**
- ❌ Splitter distributions (will come in update events)
- ❌ mc_correct_weight (reset to 0 for new leaves)
- ❌ nb_correct_weight (reset to 0 for new leaves)

---

## Architecture Benefits

This two-event design is **elegant**:

### 1. Immediate Structure Updates
- Split events are small
- Inference gets tree structure immediately
- Can make predictions right away (using stats)

### 2. Gradual Statistics Sync
- Update events sent periodically
- Bandwidth efficient (can batch updates)
- Only send updated distributions, not entire tree

### 3. Separation of Concerns
- **Structure changes** (rare, critical) → Split events
- **Statistics updates** (frequent, incremental) → Update events

### 4. Inference Can Work with Partial Info
- After split: Can predict using majority class or NB with just stats
- After updates: Can predict with more accurate distributions
- After many updates: Converges to training tree's accuracy

---

## Summary

### Is it correct that split events have no splitter data?
**YES!** ✅ This is **expected and correct**.

### Why don't they have splitter data?
Because new leaves have **empty splitters** at split time. Distributions are built incrementally.

### Where does splitter data come from?
**Leaf update events** sent after the split as learning continues.

### Is my current implementation correct?
**YES!** ✅ Your split event contains exactly what it should:
- Stats (for inference)
- Structure (for tree reconstruction)  
- NO splitters (because they don't exist yet)

The splitter data you saw is from **leaf update events**, which is the **correct place** for it! 🎯

---

## Files to Check

1. **Split event:** `json_data/verification/split_event.json`
   - Contains: stats, structure, NO splitters ✅

2. **Leaf updates:** `json_data/update_leaf/update_leaf_*.json`
   - Contains: stats, splitters, distributions ✅

**Both are correct!** Your implementation properly separates structure updates (splits) from statistics updates (leaf updates).
