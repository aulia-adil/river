# 📊 Split Event + Leaf Updates - Data Capture Summary

## ✅ Mission Accomplished!

Successfully captured **1 split event** and **10 subsequent leaf updates** from a Hoeffding Tree training session.

---

## 📁 Files Created

### Split Event
- **File**: `json_data/split_event/split_1.json`
- **Event**: Tree split on feature `age` with threshold `63.636`
- **Split Type**: `NumericBinaryBranch` (binary split)
- **Original Leaf**: ID=0 (replaced)
- **New Split Node**: ID=3 (depth=0, root node)
- **New Leaf Children**:
  - **Left Child (ID=1)**: age ≤ 63.636 - Stats: {class 1: 281.22, class 0: 215.0}
  - **Right Child (ID=2)**: age > 63.636 - Stats: {class 1: 103.78}

### Leaf Updates (10 files)
Located in `json_data/update_leaf/`:

1. **update_leaf_1.json** - Node 1: Stats={1: 281.22, 0: 216.0}
2. **update_leaf_2.json** - Node 1: Stats={1: 281.22, 0: 217.0}
3. **update_leaf_3.json** - Node 1: Stats={1: 281.22, 0: 218.0}
4. **update_leaf_4.json** - Node 2: Stats={1: 104.78} (first update to right child!)
5. **update_leaf_5.json** - Node 1: Stats={1: 281.22, 0: 219.0}
6. **update_leaf_6.json** - Node 1: Stats={1: 281.22, 0: 220.0}
7. **update_leaf_7.json** - Node 1: Stats={1: 282.22, 0: 220.0} (class 1 increased!)
8. **update_leaf_8.json** - Node 1: Stats={1: 283.22, 0: 220.0}
9. **update_leaf_9.json** - Node 1: Stats={1: 284.22, 0: 220.0}
10. **update_leaf_10.json** - Node 1: Stats={1: 284.22, 0: 221.0}

---

## 🔍 What's Included in Each JSON?

### Split Event (`split_1.json`)
```json
{
  "event_type": "split",
  "timestamp": 1760692141.86,
  "original_leaf_id": 0,
  "split_node": {
    "node_id": 3,
    "node_type": "NumericBinaryBranch",
    "depth": 0,
    "stats": {"1": 385.0, "0": 215.0},
    "branch_params": {
      "feature": "age",
      "threshold": 63.63636363636363
    }
  },
  "new_leaves": [
    {
      "node_id": 1,
      "node_type": "LeafNaiveBayesAdaptive",
      "depth": 1,
      "branch_index": 0,
      "stats": {"1": 281.22, "0": 215.0},
      "total_weight": 496.22,
      "mc_correct_weight": 0.0,
      "nb_correct_weight": 0.0,
      "splitters": {}  // Initially empty
    },
    {
      "node_id": 2,
      "branch_index": 1,
      "stats": {"1": 103.78},
      ...
    }
  ],
  "parent_context": {
    "parent_node_id": null,
    "parent_branch": null
  }
}
```

### Leaf Update (`update_leaf_1.json`)
```json
{
  "event_type": "leaf_update",
  "iteration": 1,
  "timestamp": 1760692141.86,
  "node_id": 1,
  "node_type": "LeafNaiveBayesAdaptive",
  "leaf_data": {
    "depth": 1,
    "stats": {"1": 281.22, "0": 216.0},
    "total_weight": 497.22,
    "mc_correct_weight": 0.0,
    "nb_correct_weight": 0.0,
    "splitters": {
      "salary": {
        "type": "GaussianSplitter",
        "feature_name": "salary",
        "gaussian_data": {
          "distributions": {
            "0": {
              "n_samples": 1.0,
              "mu": 130241.377,
              "sigma": 0.0
            }
          },
          "min_per_class": {"0": 130241.377},
          "max_per_class": {"0": 130241.377}
        }
      },
      "commission": {...},
      "age": {...},
      // ... 9 features total
    }
  }
}
```

---

## 🎯 Key Observations

### Split Event Analysis
1. **Trigger**: Split occurred after 600 instances (grace_period=200)
2. **Split Feature**: `age` with threshold `63.636`
3. **Reason**: Best information gain - older people (>63) vs younger (≤63)
4. **Tree Structure Change**: 
   - Before: 1 root leaf (ID=0)
   - After: 1 split node (ID=3) + 2 leaf children (ID=1, ID=2)

### Leaf Update Patterns
1. **Left Child (ID=1)** - Received **9 out of 10 updates** (ages ≤ 63)
   - Class distribution balanced: ~281 (class 1) vs ~216-221 (class 0)
   - Most training instances go to this branch
   
2. **Right Child (ID=2)** - Received **1 update** (ages > 63)
   - Dominated by class 1: 104.78 instances
   - Fewer instances in this region

3. **MC/NB Weights**: Track which prediction method (Majority Class vs Naive Bayes) performs better
   - Start at 0.0 for new leaves
   - Update #7 onwards: Both MC and NB weights increase (predictions being evaluated)

### Complete Data for Synchronization
Each leaf update contains:
- ✅ **Stats**: Class distribution counts
- ✅ **Total Weight**: Sum of all instance weights
- ✅ **MC/NB Weights**: Performance tracking for adaptive prediction
- ✅ **Splitters**: Complete Gaussian distributions for all 9 features
  - Salary, commission, age, elevel, car, zipcode, hvalue, hyears, loan
  - Each with: n_samples, μ (mean), σ (standard deviation), min/max per class

---

## 🚀 Use Cases

### 1. Distributed Training
- **Training Node**: Publishes split event to Kafka
- **Inference Nodes**: Consume split event and reconstruct tree structure
- Result: All nodes have identical tree topology

### 2. Model Synchronization
- **Training Node**: Publishes leaf updates after every N instances
- **Inference Nodes**: Apply updates to keep their leaf statistics current
- Result: All nodes maintain synchronized probability distributions

### 3. Model Versioning
- Save split events and updates as checkpoints
- Replay updates to reconstruct tree state at any point in time
- Enables time-travel debugging and A/B testing

---

## 📈 Training Statistics

- **Total Instances**: 610
- **Split Occurred At**: Instance 600
- **Instances After Split**: 10
- **Left Branch Traffic**: 90% (9/10 updates)
- **Right Branch Traffic**: 10% (1/10 updates)
- **Features**: 9 numeric features from Agrawal dataset
- **Classes**: Binary (0, 1)

---

## 🎓 Thesis Relevance

This data demonstrates the **complete distributed synchronization protocol**:

1. **Split Broadcast**: When training tree splits, send complete split info
   - Original leaf ID being replaced
   - New split node parameters (feature, threshold, type)
   - All new leaf children with initial stats
   
2. **Incremental Updates**: After split, send periodic leaf updates
   - Updated class distributions
   - Updated feature distributions (Gaussians)
   - Performance metrics (MC/NB weights)

3. **Node ID System**: O(1) lookup for applying updates
   - Each node has unique ID
   - Updates reference node ID for direct access
   - No tree traversal needed

**Result**: Inference nodes can perfectly replicate training tree structure and statistics! 🎉

---

**Generated**: December 16, 2024
**Dataset**: Agrawal (classification_function=0, seed=42)
**Tree Type**: HoeffdingTreeClassifier with Naive Bayes Adaptive leaves
