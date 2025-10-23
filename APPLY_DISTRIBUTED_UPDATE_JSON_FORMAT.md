# `apply_distributed_update()` JSON Format Reference

This document describes the JSON format for using the `apply_distributed_update()` method to synchronize distributed Hoeffding Trees.

---

## Method Signature

```python
model.apply_distributed_update(node_id, update_payload)
```

**Parameters:**
- `node_id` (int): The ID of the node to update
- `update_payload` (dict): Update data in the format described below

**Returns:**
- `bool`: True if update successful, False otherwise

---

## Update Types

The method supports 5 different update types:

1. **`leaf_stats`** - Update class counts and weights only
2. **`splitter_data`** - Update feature distributions only
3. **`naive_bayes_data`** - Update Naive Bayes weights only
4. **`complete_node`** - Update everything at once (recommended)
5. **`incremental_stats`** - Apply single instance learning

---

## 1. Complete Node Update (Recommended)

**Use case:** Full synchronization of a leaf node's state

```json
{
  "update_type": "complete_node",
  "timestamp": 1729700000.0,
  "data": {
    "leaf_stats": {
      "stats": {
        "0": 215.0,
        "1": 281.2230429243434
      },
      "total_weight": 496.2230429243434
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
                "n_samples": 5.0,
                "mu": 100830.94,
                "sigma": 50621.71
              },
              "1": {
                "n_samples": 10.0,
                "mu": 85430.22,
                "sigma": 38920.15
              }
            },
            "min_per_class": {
              "0": 27039.26,
              "1": 20682.66
            },
            "max_per_class": {
              "0": 144608.49,
              "1": 149652.35
            }
          }
        },
        "age": {
          "type": "GaussianSplitter",
          "feature_name": "age",
          "gaussian_data": {
            "distributions": {
              "0": {
                "n_samples": 5.0,
                "mu": 45.6,
                "sigma": 4.56
              },
              "1": {
                "n_samples": 10.0,
                "mu": 52.3,
                "sigma": 12.8
              }
            },
            "min_per_class": {
              "0": 40,
              "1": 20
            },
            "max_per_class": {
              "0": 59,
              "1": 80
            }
          }
        }
      }
    }
  }
}
```

---

## 2. Leaf Stats Update Only

**Use case:** Update class counts without touching distributions

```json
{
  "update_type": "leaf_stats",
  "timestamp": 1729700000.0,
  "data": {
    "stats": {
      "0": 220.0,
      "1": 290.0
    },
    "total_weight": 510.0
  }
}
```

**What happens:**
- Replaces `node.stats` with new values (synchronization, not accumulation)
- Validates that `total_weight` matches sum of stats

---

## 3. Splitter Data Update Only

**Use case:** Update Gaussian distributions without touching class counts

### For Gaussian Splitters:

```json
{
  "update_type": "splitter_data",
  "timestamp": 1729700000.0,
  "data": {
    "splitters": {
      "salary": {
        "type": "GaussianSplitter",
        "feature_name": "salary",
        "gaussian_data": {
          "distributions": {
            "0": {
              "n_samples": 5.0,
              "mu": 100830.94,
              "sigma": 50621.71
            }
          },
          "min_per_class": {
            "0": 27039.26
          },
          "max_per_class": {
            "0": 144608.49
          }
        }
      }
    }
  }
}
```

### For Nominal Splitters:

```json
{
  "update_type": "splitter_data",
  "timestamp": 1729700000.0,
  "data": {
    "splitters": {
      "zipcode": {
        "type": "NominalSplitter",
        "feature_name": "zipcode",
        "nominal_data": {
          "class_distributions": {
            "0": {
              "zone_A": 10,
              "zone_B": 5,
              "zone_C": 15
            },
            "1": {
              "zone_A": 20,
              "zone_B": 8,
              "zone_C": 12
            }
          },
          "unique_values": ["zone_A", "zone_B", "zone_C"],
          "total_weight": 70
        }
      }
    }
  }
}
```

---

## 4. Naive Bayes Data Update Only

**Use case:** Update Naive Bayes correctness weights

```json
{
  "update_type": "naive_bayes_data",
  "timestamp": 1729700000.0,
  "data": {
    "mc_correct_weight": 120.5,
    "nb_correct_weight": 145.3
  }
}
```

**What happens:**
- Updates `node._mc_correct_weight` (Majority Class correctness)
- Updates `node._nb_correct_weight` (Naive Bayes correctness)
- Used by Naive Bayes Adaptive leaves to choose prediction method

---

## 5. Incremental Stats Update

**Use case:** Apply learning from a single distributed instance

```json
{
  "update_type": "incremental_stats",
  "timestamp": 1729700000.0,
  "data": {
    "instance": {
      "age": 45,
      "salary": 85000,
      "loan": 250000
    },
    "class_label": 1,
    "weight": 1.0
  }
}
```

**What happens:**
- Calls `node.learn_one(x, y, w)` directly
- Updates stats, splitters, and Naive Bayes weights incrementally

---

## Field Descriptions

### Common Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `update_type` | string | ✅ Yes | One of: `leaf_stats`, `splitter_data`, `naive_bayes_data`, `complete_node`, `incremental_stats` |
| `timestamp` | float | ⚠️ Optional | Unix timestamp of when update was created |
| `data` | object | ✅ Yes | Update-specific data (format depends on `update_type`) |

### Leaf Stats Fields

| Field | Type | Description |
|-------|------|-------------|
| `stats` | object | Class counts: `{class_label: count}` |
| `total_weight` | float | Total weight of all samples (should equal sum of stats) |

### Gaussian Splitter Fields

| Field | Type | Description |
|-------|------|-------------|
| `distributions` | object | Per-class distributions: `{class_label: {n_samples, mu, sigma}}` |
| `n_samples` | float | Number of samples observed for this class |
| `mu` | float | Mean (μ) of the Gaussian distribution |
| `sigma` | float | Standard deviation (σ) of the Gaussian distribution |
| `min_per_class` | object | Minimum observed value per class: `{class_label: min_value}` |
| `max_per_class` | object | Maximum observed value per class: `{class_label: max_value}` |

### Nominal Splitter Fields

| Field | Type | Description |
|-------|------|-------------|
| `class_distributions` | object | Category counts per class: `{class_label: {category: count}}` |
| `unique_values` | array | List of all observed categories |
| `total_weight` | float | Total weight observed for this feature |

### Naive Bayes Fields

| Field | Type | Description |
|-------|------|-------------|
| `mc_correct_weight` | float | Cumulative weight of correct Majority Class predictions |
| `nb_correct_weight` | float | Cumulative weight of correct Naive Bayes predictions |

---

## Usage Examples

### Example 1: Python Usage

```python
from river import tree

# Create model
model = tree.HoeffdingTreeClassifier()

# ... train model, get splits ...

# Create update payload
update_payload = {
    "update_type": "complete_node",
    "data": {
        "leaf_stats": {
            "stats": {1: 281.22, 0: 215.0},
            "total_weight": 496.22
        },
        "splitter_data": {
            "splitters": {
                "age": {
                    "type": "GaussianSplitter",
                    "gaussian_data": {
                        "distributions": {
                            "0": {"n_samples": 5, "mu": 45.6, "sigma": 4.56}
                        },
                        "min_per_class": {"0": 40},
                        "max_per_class": {"0": 59}
                    }
                }
            }
        },
        "naive_bayes_data": {
            "mc_correct_weight": 0.0,
            "nb_correct_weight": 0.0
        }
    }
}

# Apply update to node 1
success = model.apply_distributed_update(node_id=1, update_payload=update_payload)

if success:
    print("✅ Update applied successfully")
else:
    print("❌ Update failed")
```

### Example 2: From JSON File

```python
import json
from river import tree

model = tree.HoeffdingTreeClassifier()

# Load update from JSON file
with open('leaf_update_001.json', 'r') as f:
    update_data = json.load(f)

# Extract node_id and payload
node_id = update_data['node_id']
update_payload = {
    'update_type': 'complete_node',
    'data': {
        'leaf_stats': {
            'stats': update_data['leaf_data']['stats'],
            'total_weight': update_data['leaf_data']['total_weight']
        },
        'splitter_data': {
            'splitters': update_data['leaf_data']['splitters']
        },
        'naive_bayes_data': {
            'mc_correct_weight': update_data['leaf_data']['mc_correct_weight'],
            'nb_correct_weight': update_data['leaf_data']['nb_correct_weight']
        }
    }
}

# Apply
success = model.apply_distributed_update(node_id, update_payload)
```

### Example 3: Kafka-style Distributed Learning

```python
from river import tree
import json

# Training process
training_model = tree.HoeffdingTreeClassifier(
    leaf_update_callback=lambda info: send_to_kafka(info)  # Send updates to Kafka
)

# Train
for x, y in dataset:
    training_model.learn_one(x, y)

# ----

# Inference process (different machine)
inference_model = tree.HoeffdingTreeClassifier()

# Receive update from Kafka
kafka_message = receive_from_kafka()

# Apply update
node_id = kafka_message['node_id']
update_payload = {
    'update_type': 'complete_node',
    'data': {
        'leaf_stats': {
            'stats': kafka_message['leaf_data']['stats'],
            'total_weight': kafka_message['leaf_data']['total_weight']
        },
        'splitter_data': {
            'splitters': kafka_message['leaf_data']['splitters']
        },
        'naive_bayes_data': {
            'mc_correct_weight': kafka_message['leaf_data']['mc_correct_weight'],
            'nb_correct_weight': kafka_message['leaf_data']['nb_correct_weight']
        }
    }
}

inference_model.apply_distributed_update(node_id, update_payload)
```

---

## Important Notes

### 1. Pre-create Splitters

**CRITICAL:** Before calling `apply_distributed_update`, ensure splitters exist for all features in the payload:

```python
node = model.get_node_by_id(node_id)

# Create missing splitters
if 'splitters' in update_payload['data']['splitter_data']:
    if not hasattr(node, 'splitters') or node.splitters is None:
        node.splitters = {}
    
    for feature_name in update_payload['data']['splitter_data']['splitters'].keys():
        if feature_name not in node.splitters:
            # Clone from model's splitter template
            node.splitters[feature_name] = model.splitter.clone()

# Now apply the update
model.apply_distributed_update(node_id, update_payload)
```

### 2. Class Label Format

Class labels can be strings or numbers in JSON, but will be converted:
- String digits (`"0"`, `"1"`) → integers
- Numeric strings (`"0.5"`) → floats
- Other strings → kept as strings

### 3. Gaussian Parameters

- `mu`: Mean of the distribution
- `sigma`: **Standard deviation** (σ), not variance
- The method internally converts to variance (σ²) when needed

### 4. Synchronization vs. Accumulation

- **`leaf_stats`**: REPLACES stats (synchronization)
- **`splitter_data`**: REPLACES Gaussian distributions (exact sync)
- **`naive_bayes_data`**: REPLACES weights (synchronization)
- **`incremental_stats`**: ACCUMULATES (adds to existing)

### 5. Error Handling

The method returns `False` if:
- Node ID not found in registry
- Unknown update type
- Missing required fields
- Exception during update

Always check the return value:

```python
success = model.apply_distributed_update(node_id, update_payload)
if not success:
    print("Update failed - check logs for details")
```

---

## Compatibility with Leaf Update Callbacks

The JSON format produced by `leaf_update_callback` can be used directly with `apply_distributed_update`:

```python
# Training model captures updates
def capture_update(info):
    update_payload = {
        'update_type': 'complete_node',
        'data': {
            'leaf_stats': {
                'stats': info['leaf_data']['stats'],
                'total_weight': info['leaf_data']['total_weight']
            },
            'splitter_data': {
                'splitters': info['leaf_data']['splitters']
            },
            'naive_bayes_data': {
                'mc_correct_weight': info['leaf_data']['naive_bayes']['mc_correct_weight'],
                'nb_correct_weight': info['leaf_data']['naive_bayes']['nb_correct_weight']
            }
        }
    }
    
    # Send to inference process
    send_to_inference(info['node_id'], update_payload)

training_model = tree.HoeffdingTreeClassifier(
    leaf_update_callback=capture_update
)

# Inference model applies updates
def receive_and_apply(node_id, update_payload):
    # Pre-create splitters (important!)
    node = inference_model.get_node_by_id(node_id)
    if node and 'splitter_data' in update_payload['data']:
        if not hasattr(node, 'splitters'):
            node.splitters = {}
        for feature in update_payload['data']['splitter_data']['splitters'].keys():
            if feature not in node.splitters:
                node.splitters[feature] = inference_model.splitter.clone()
    
    # Apply
    inference_model.apply_distributed_update(node_id, update_payload)

inference_model = tree.HoeffdingTreeClassifier()
```

---

## Summary

### Quick Reference Table

| Update Type | Use Case | Stats | Splitters | NB Weights |
|-------------|----------|-------|-----------|------------|
| `leaf_stats` | Update class counts only | ✅ | ❌ | ❌ |
| `splitter_data` | Update distributions only | ❌ | ✅ | ❌ |
| `naive_bayes_data` | Update NB weights only | ❌ | ❌ | ✅ |
| `complete_node` | Full synchronization | ✅ | ✅ | ✅ |
| `incremental_stats` | Single instance learning | ✅ | ✅ | ✅ |

### Recommended Usage

**For distributed learning synchronization:**
- Use `complete_node` type
- Include all three data sections
- Pre-create splitters before applying
- Apply updates periodically (e.g., every 50 samples)

**For single instance replication:**
- Use `incremental_stats` type
- Send individual samples as they arrive
- Useful for real-time synchronization

**For selective updates:**
- Use specific update types (`leaf_stats`, `splitter_data`, `naive_bayes_data`)
- Useful when only partial state needs synchronization
- Can reduce network bandwidth
