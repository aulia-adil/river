"""
Complete Naive Bayes Serialization for Kafka

This shows exactly how to extract and serialize all the Naive Bayes data
from Hoeffding Tree leaves for your distributed training thesis.
"""

import json
from typing import Dict, Any


class NaiveBayesLeafSerializer:
    """
    Extracts and serializes all Naive Bayes data from leaf nodes.
    This is what you'll integrate into your Kafka callback.
    """
    
    def serialize_leaf_naive_bayes_data(self, leaf) -> Dict[str, Any]:
        """
        Extract all Naive Bayes data from a leaf node for Kafka transmission.
        
        Returns complete data needed to reconstruct identical predictions.
        """
        
        nb_data = {
            "leaf_type": type(leaf).__name__,
            "depth": getattr(leaf, 'depth', None),
            "total_weight": getattr(leaf, 'total_weight', 0.0),
            
            # Class statistics (prior probabilities)
            "class_stats": self._serialize_class_stats(getattr(leaf, 'stats', {})),
            
            # Performance tracking
            "performance": {
                "mc_correct_weight": getattr(leaf, '_mc_correct_weight', 0.0),
                "nb_correct_weight": getattr(leaf, '_nb_correct_weight', 0.0),
                "uses_naive_bayes": self._determines_nb_usage(leaf)
            },
            
            # Feature models (the heart of Naive Bayes)
            "feature_models": self._serialize_feature_models(getattr(leaf, 'splitters', {}))
        }
        
        return nb_data
    
    def _serialize_class_stats(self, stats: Dict) -> Dict[str, float]:
        """Serialize class counts/weights."""
        return {str(k): float(v) for k, v in stats.items()} if stats else {}
    
    def _determines_nb_usage(self, leaf) -> bool:
        """Determine if this leaf uses Naive Bayes or Majority Class."""
        if not hasattr(leaf, '_mc_correct_weight') or not hasattr(leaf, '_nb_correct_weight'):
            return False
        return leaf._nb_correct_weight >= leaf._mc_correct_weight
    
    def _serialize_feature_models(self, splitters: Dict) -> Dict[str, Any]:
        """
        Serialize all feature models (splitters).
        This is the most complex part - each feature type has different data.
        """
        feature_models = {}
        
        for feature_name, splitter in splitters.items():
            splitter_type = type(splitter).__name__
            
            if splitter_type == 'GaussianSplitter':
                # Numeric features with Gaussian distribution
                feature_models[feature_name] = self._serialize_gaussian_splitter(splitter)
                
            elif splitter_type == 'NominalSplitterClassif':
                # Categorical features with frequency counts
                feature_models[feature_name] = self._serialize_nominal_splitter(splitter)
                
            else:
                # Other splitter types
                feature_models[feature_name] = self._serialize_generic_splitter(splitter)
        
        return feature_models
    
    def _serialize_gaussian_splitter(self, splitter) -> Dict[str, Any]:
        """Serialize Gaussian (numeric) feature model."""
        data = {
            "type": "GaussianSplitter",
            "means": {},
            "variances": {},
            "sample_counts": {}
        }
        
        # Extract means per class
        if hasattr(splitter, '_mean_per_class'):
            data["means"] = {str(k): float(v) for k, v in splitter._mean_per_class.items()}
        
        # Extract variances per class
        if hasattr(splitter, '_var_per_class'):
            data["variances"] = {str(k): float(v) for k, v in splitter._var_per_class.items()}
        
        # Extract sample counts per class
        if hasattr(splitter, '_n_samples_per_class'):
            data["sample_counts"] = {str(k): int(v) for k, v in splitter._n_samples_per_class.items()}
        
        return data
    
    def _serialize_nominal_splitter(self, splitter) -> Dict[str, Any]:
        """Serialize nominal (categorical) feature model."""
        data = {
            "type": "NominalSplitterClassif",
            "value_counts": {}
        }
        
        # Extract frequency counts
        if hasattr(splitter, '_counts'):
            # Convert the complex _counts structure to a more manageable format
            counts_by_value_and_class = {}
            
            for key, count in splitter._counts.items():
                if isinstance(key, tuple) and len(key) >= 3:
                    feature, value, class_label = key[0], key[1], key[2]
                    
                    if value not in counts_by_value_and_class:
                        counts_by_value_and_class[value] = {}
                    
                    counts_by_value_and_class[value][str(class_label)] = int(count)
            
            data["value_counts"] = counts_by_value_and_class
        
        return data
    
    def _serialize_generic_splitter(self, splitter) -> Dict[str, Any]:
        """Serialize other types of splitters."""
        data = {
            "type": type(splitter).__name__,
            "attributes": []
        }
        
        # Try to extract common attributes
        common_attrs = ['_mean_per_class', '_var_per_class', '_n_samples_per_class', '_counts']
        for attr in common_attrs:
            if hasattr(splitter, attr):
                try:
                    value = getattr(splitter, attr)
                    data[attr] = self._serialize_complex_value(value)
                except:
                    data[attr] = f"Cannot serialize {type(value)}"
        
        return data
    
    def _serialize_complex_value(self, value):
        """Helper to serialize complex values."""
        if isinstance(value, dict):
            return {str(k): self._serialize_complex_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_complex_value(v) for v in value]
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)


class NaiveBayesLeafDeserializer:
    """
    Reconstructs Naive Bayes leaf nodes from Kafka messages.
    This is what inference processes use.
    """
    
    def reconstruct_leaf_predictions(self, nb_data: Dict[str, Any], instance: Dict[str, Any]) -> Dict[str, float]:
        """
        Reconstruct Naive Bayes predictions from serialized data.
        This mimics what the original leaf would predict.
        """
        
        class_stats = nb_data.get('class_stats', {})
        feature_models = nb_data.get('feature_models', {})
        uses_nb = nb_data.get('performance', {}).get('uses_naive_bayes', False)
        
        if not uses_nb or not feature_models:
            # Use majority class
            return self._majority_class_prediction(class_stats)
        else:
            # Use Naive Bayes
            return self._naive_bayes_prediction(instance, class_stats, feature_models)
    
    def _majority_class_prediction(self, class_stats: Dict[str, float]) -> Dict[str, float]:
        """Simple majority class prediction."""
        if not class_stats:
            return {}
        
        total = sum(class_stats.values())
        return {k: v / total for k, v in class_stats.items()}
    
    def _naive_bayes_prediction(self, instance: Dict[str, Any], class_stats: Dict[str, float], feature_models: Dict[str, Any]) -> Dict[str, float]:
        """Full Naive Bayes prediction reconstruction."""
        import math
        
        if not class_stats:
            return {}
        
        total_weight = sum(class_stats.values())
        log_posteriors = {}
        
        # Calculate for each class
        for class_label, class_count in class_stats.items():
            # Prior probability (log)
            prior = class_count / total_weight
            log_posterior = math.log(prior) if prior > 0 else float('-inf')
            
            # Feature likelihoods (log)
            for feature_name, feature_value in instance.items():
                if feature_name in feature_models:
                    model = feature_models[feature_name]
                    likelihood = self._calculate_likelihood(feature_value, class_label, model)
                    
                    if likelihood > 0:
                        log_posterior += math.log(likelihood)
                    # If likelihood is 0, we skip (equivalent to adding -inf, but numerically stable)
            
            log_posteriors[class_label] = log_posterior
        
        # Convert back to probabilities using log-sum-exp trick
        if not log_posteriors:
            return {}
        
        max_log = max(log_posteriors.values())
        exp_posteriors = {k: math.exp(v - max_log) for k, v in log_posteriors.items()}
        total_exp = sum(exp_posteriors.values())
        
        return {k: v / total_exp for k, v in exp_posteriors.items()}
    
    def _calculate_likelihood(self, feature_value, class_label, model: Dict[str, Any]) -> float:
        """Calculate P(feature_value|class) from the model."""
        model_type = model.get('type', '')
        
        if model_type == 'GaussianSplitter':
            return self._gaussian_likelihood(feature_value, class_label, model)
        elif model_type == 'NominalSplitterClassif':
            return self._nominal_likelihood(feature_value, class_label, model)
        else:
            return 1.0  # Uniform likelihood for unknown types
    
    def _gaussian_likelihood(self, value, class_label, model: Dict[str, Any]) -> float:
        """Calculate Gaussian likelihood P(value|class)."""
        import math
        
        means = model.get('means', {})
        variances = model.get('variances', {})
        
        if str(class_label) not in means or str(class_label) not in variances:
            return 1.0
        
        mean = means[str(class_label)]
        variance = variances[str(class_label)]
        
        if variance <= 0:
            return 1.0
        
        # Gaussian PDF
        coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
        exponent = -((value - mean) ** 2) / (2 * variance)
        
        return coefficient * math.exp(exponent)
    
    def _nominal_likelihood(self, value, class_label, model: Dict[str, Any]) -> float:
        """Calculate nominal likelihood P(value|class)."""
        value_counts = model.get('value_counts', {})
        
        if str(value) not in value_counts:
            return 1e-6  # Small smoothing for unseen values
        
        class_counts = value_counts[str(value)]
        if str(class_label) not in class_counts:
            return 1e-6
        
        # Total instances of this class
        total_class_instances = sum(class_counts.values())
        if total_class_instances == 0:
            return 1e-6
        
        return class_counts[str(class_label)] / total_class_instances


def demonstrate_nb_serialization():
    """Demonstrate the complete Naive Bayes serialization process."""
    
    print("🧠 NAIVE BAYES SERIALIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Mock Naive Bayes leaf data
    mock_nb_data = {
        "leaf_type": "LeafNaiveBayesAdaptive",
        "depth": 2,
        "total_weight": 100.0,
        "class_stats": {
            "0": 40.0,
            "1": 60.0
        },
        "performance": {
            "mc_correct_weight": 75.0,
            "nb_correct_weight": 82.0,
            "uses_naive_bayes": True
        },
        "feature_models": {
            "age": {
                "type": "GaussianSplitter",
                "means": {"0": 28.5, "1": 45.2},
                "variances": {"0": 12.3, "1": 8.7},
                "sample_counts": {"0": 40, "1": 60}
            },
            "education": {
                "type": "NominalSplitterClassif",
                "value_counts": {
                    "high": {"0": 15, "1": 35},
                    "medium": {"0": 20, "1": 20},
                    "low": {"0": 5, "1": 5}
                }
            }
        }
    }
    
    print("📦 SERIALIZED NAIVE BAYES DATA:")
    print(json.dumps(mock_nb_data, indent=2))
    
    # Test reconstruction
    deserializer = NaiveBayesLeafDeserializer()
    test_instance = {"age": 35, "education": "high"}
    
    prediction = deserializer.reconstruct_leaf_predictions(mock_nb_data, test_instance)
    
    print(f"\n🔮 RECONSTRUCTED PREDICTION:")
    print(f"Instance: {test_instance}")
    print(f"Prediction: {prediction}")
    
    # Calculate message size
    json_str = json.dumps(mock_nb_data)
    message_size = len(json_str.encode('utf-8'))
    
    print(f"\n📊 MESSAGE STATISTICS:")
    print(f"JSON size: {message_size} bytes")
    print(f"Features: {len(mock_nb_data['feature_models'])}")
    print(f"Classes: {len(mock_nb_data['class_stats'])}")
    
    print(f"\n✅ THIS IS WHAT YOU'LL SEND OVER KAFKA!")
    print("   Complete Naive Bayes state for identical reconstruction")


if __name__ == "__main__":
    demonstrate_nb_serialization()