"""
Inference Process - Loads model state from captured JSON data

This script demonstrates how an inference node can reconstruct a Hoeffding Tree
by applying split events and leaf updates from JSON files (simulating Kafka messages).

Flow:
1. Initialize empty model
2. Load and apply split event from json_data/split_event/
3. Load and apply leaf updates from json_data/update_leaf/
4. Make predictions on new samples
"""

import json
import os
from pathlib import Path
from river import tree
from river.datasets import synth
from river.tree.nodes.branch import NumericBinaryBranch, NominalBinaryBranch
from river.tree.nodes.branch import NumericMultiwayBranch, NominalMultiwayBranch
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree.splitter import GaussianSplitter, Splitter


class InferenceProcess:
    """
    Inference process that reconstructs model from JSON events
    """
    
    def __init__(self):
        """Initialize inference model"""
        self.model = tree.HoeffdingTreeClassifier(
            grace_period=200,
            delta=1e-5,
            leaf_prediction='mc',
            nb_threshold=10
        )
        print("🔧 Initialized empty Hoeffding Tree for inference")
        print(f"   Initial state: {self.model.n_nodes} nodes, height {self.model.height}")
        print()
    
    def apply_split_event(self, split_data):
        """
        Apply a split event to reconstruct tree structure
        
        Args:
            split_data: Dictionary containing split event information
        """
        print("=" * 70)
        print("📥 APPLYING SPLIT EVENT")
        print("=" * 70)
        
        original_leaf_id = split_data['original_leaf_id']
        split_node_info = split_data['split_node']
        new_leaves_info = split_data['new_leaves']
        
        print(f"Original leaf ID: {original_leaf_id}")
        print(f"Split node ID: {split_node_info['node_id']}")
        print(f"Split node type: {split_node_info['node_type']}")
        print(f"Branch params: {split_node_info['branch_params']}")
        print(f"New leaves: {[leaf['node_id'] for leaf in new_leaves_info]}")
        print()
        
        # Create new leaf children FIRST
        new_leaves = []
        for leaf_info in new_leaves_info:
            leaf = self._create_leaf_from_info(leaf_info)
            new_leaves.append(leaf)
            
        # Create the split node with children
        # Note: The children are attached during construction (left/right params)
        split_node = self._create_split_node(split_node_info, new_leaves)
        
        # Verify children are attached (for debugging)
        print(f"   Split node has {len(split_node.children)} children:")
        for i, child in enumerate(split_node.children):
            child_id = new_leaves_info[i]['node_id']
            print(f"      Child {i}: {type(child).__name__} (will be node_id={child_id})")
        
        # Replace root (for now, assuming root split)
        if self.model._root is None or original_leaf_id == 0:
            self.model._root = split_node
            print(f"✅ Replaced root with split node ID={split_node_info['node_id']}")
        
        # Register nodes in the model's node registry
        # This assigns node_id attributes and adds to registry for O(1) lookup
        if hasattr(self.model, '_register_node'):
            self.model._register_node(split_node, split_node_info['node_id'])
            for leaf, leaf_info in zip(new_leaves, new_leaves_info):
                self.model._register_node(leaf, leaf_info['node_id'])
            print(f"   Registry now contains: {list(self.model._node_registry.keys())}")
        
        print(f"✅ Split event applied successfully")
        print(f"   Model state: {self.model.n_nodes} nodes, height {self.model.height}")
        print()
    
    def _create_split_node(self, split_node_info, children):
        """Create a split node from split event data with children"""
        node_type = split_node_info['node_type']
        branch_params = split_node_info['branch_params']
        stats = split_node_info.get('stats', {})
        depth = split_node_info.get('depth', 0)
        
        # Convert stats from string keys to int
        stats_dict = {int(k): v for k, v in stats.items()} if stats else {}
        
        if node_type == 'NumericBinaryBranch':
            node = NumericBinaryBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                threshold=branch_params['threshold'],
                depth=depth,
                left=children[0],
                right=children[1]
            )
        elif node_type == 'NominalBinaryBranch':
            node = NominalBinaryBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                value=branch_params['value'],
                depth=depth,
                left=children[0],
                right=children[1]
            )
        elif node_type == 'NumericMultiwayBranch':
            node = NumericMultiwayBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                depth=depth,
                *children  # Multiway branches accept variable children
            )
        elif node_type == 'NominalMultiwayBranch':
            node = NominalMultiwayBranch(
                stats=stats_dict,
                feature=branch_params['feature'],
                depth=depth,
                *children
            )
        else:
            raise ValueError(f"Unknown split node type: {node_type}")
        
        return node
    
    def _create_leaf_from_info(self, leaf_info):
        """Create a leaf node from leaf information"""
        stats = {int(k): v for k, v in leaf_info['stats'].items()}
        depth = leaf_info.get('depth', 0)
        
        # Create leaf node (using LeafMajorityClass for now)
        leaf = LeafMajorityClass(stats=stats, depth=depth, splitter=None)
        
        return leaf
    
    def apply_leaf_update(self, update_data):
        """
        Apply a leaf update to synchronize leaf statistics
        
        Args:
            update_data: Dictionary containing leaf update information
        """
        node_id = update_data['node_id']
        leaf_data = update_data['leaf_data']
        
        print(f"📥 Applying leaf update to node {node_id}")
        
        # Find the node in the registry
        if hasattr(self.model, '_node_registry'):
            node = self.model._node_registry.get(node_id)
            if node is None:
                print(f"⚠️  Warning: Node {node_id} not found in registry")
                return
        else:
            print(f"⚠️  Warning: Model doesn't have node registry")
            return
        
        # Update stats
        new_stats = {int(k): v for k, v in leaf_data['stats'].items()}
        node.stats = new_stats
        
        # Update weights
        if hasattr(node, '_mc_correct_weight'):
            node._mc_correct_weight = leaf_data.get('mc_correct_weight', 0)
        if hasattr(node, '_nb_correct_weight'):
            node._nb_correct_weight = leaf_data.get('nb_correct_weight', 0)
        
        # Update splitters (feature distributions)
        if 'splitters' in leaf_data and leaf_data['splitters']:
            self._update_splitters(node, leaf_data['splitters'])
        
        print(f"   ✅ Updated node {node_id}: stats={new_stats}, total_weight={leaf_data['total_weight']}")
    
    def _update_splitters(self, node, splitters_data):
        """Update Gaussian distributions in the node's splitters"""
        if not hasattr(node, 'splitters') or node.splitters is None:
            # Initialize splitters if they don't exist
            node.splitters = {}
        
        for feature_name, splitter_info in splitters_data.items():
            if 'gaussian_data' in splitter_info:
                gaussian_data = splitter_info['gaussian_data']
                
                # Create or update Gaussian splitter
                if feature_name not in node.splitters:
                    node.splitters[feature_name] = GaussianSplitter()
                
                splitter = node.splitters[feature_name]
                
                # Update Gaussian distributions
                if hasattr(splitter, '_dist'):
                    from river.proba import Gaussian
                    
                    distributions = gaussian_data.get('distributions', {})
                    for class_label, dist_params in distributions.items():
                        class_label = int(class_label)
                        
                        # Create Gaussian distribution from state
                        n = dist_params['n_samples']
                        mu = dist_params['mu']
                        sigma = dist_params['sigma']
                        
                        if n > 0:
                            # Use _from_state for deterministic reconstruction
                            splitter._dist[class_label] = Gaussian._from_state(
                                n=n,
                                m=mu,
                                sig=sigma,
                                ddof=1
                            )
                
                # Update min/max per class
                if hasattr(splitter, '_min_per_class'):
                    min_per_class = gaussian_data.get('min_per_class', {})
                    max_per_class = gaussian_data.get('max_per_class', {})
                    
                    splitter._min_per_class = {int(k): v for k, v in min_per_class.items()}
                    splitter._max_per_class = {int(k): v for k, v in max_per_class.items()}
    
    def load_split_events(self, split_dir='json_data/split_event'):
        """Load all split events from directory"""
        split_files = sorted(Path(split_dir).glob('split_*.json'))
        
        print("=" * 70)
        print(f"📂 LOADING SPLIT EVENTS FROM {split_dir}")
        print("=" * 70)
        print(f"Found {len(split_files)} split event(s)")
        print()
        
        for split_file in split_files:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            print(f"Loading: {split_file.name}")
            self.apply_split_event(split_data)
    
    def load_leaf_updates(self, update_dir='json_data/update_leaf', max_updates=None):
        """Load leaf updates from directory"""
        update_files = sorted(Path(update_dir).glob('update_leaf_*.json'))
        
        if max_updates:
            update_files = update_files[:max_updates]
        
        print("=" * 70)
        print(f"📂 LOADING LEAF UPDATES FROM {update_dir}")
        print("=" * 70)
        print(f"Loading {len(update_files)} update(s)")
        print()
        
        for update_file in update_files:
            with open(update_file, 'r') as f:
                update_data = json.load(f)
            
            self.apply_leaf_update(update_data)
        
        print(f"\n✅ Applied {len(update_files)} leaf updates")
        print()
    
    def make_predictions(self, n_samples=5, seed=999):
        """Make predictions on new samples"""
        print("=" * 70)
        print("🔮 MAKING PREDICTIONS")
        print("=" * 70)
        print(f"Model state: {self.model.n_nodes} nodes, height {self.model.height}")
        print()
        
        # Generate test samples
        dataset = synth.Agrawal(classification_function=0, seed=seed)
        test_samples = list(dataset.take(n_samples))
        
        results = []
        
        for i, (x, y_true) in enumerate(test_samples, 1):
            # Get prediction
            y_pred = self.model.predict_one(x)
            y_proba = self.model.predict_proba_one(x)
            
            result = {
                "sample_id": i,
                "features": {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in x.items()},
                "true_label": int(y_true),
                "predicted_label": int(y_pred) if y_pred is not None else None,
                "prediction_probabilities": {
                    str(k): float(v) for k, v in y_proba.items()
                } if y_proba else {},
                "correct": bool(y_pred == y_true) if y_pred is not None else False
            }
            
            results.append(result)
            
            # Print result
            print(f"Sample #{i}:")
            print(f"  Age: {x['age']:.1f}, Salary: {x['salary']:.0f}, Loan: {x['loan']:.0f}")
            print(f"  True: {y_true}, Predicted: {y_pred}, Correct: {y_pred == y_true}")
            print(f"  Probabilities: {y_proba}")
            print()
        
        # Calculate accuracy
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        
        print("=" * 70)
        print("📊 PREDICTION SUMMARY")
        print("=" * 70)
        print(f"Total samples: {len(results)}")
        print(f"Correct predictions: {sum(1 for r in results if r['correct'])}")
        print(f"Accuracy: {accuracy:.2%}")
        print()
        
        return results
    
    def save_predictions(self, results, output_file='json_data/inference_predictions.json'):
        """Save prediction results to file"""
        summary = {
            "model_info": {
                "n_nodes": self.model.n_nodes,
                "height": self.model.height,
                "n_active_leaves": self.model.n_active_leaves,
                "n_inactive_leaves": self.model.n_inactive_leaves,
                "source": "reconstructed_from_json_events"
            },
            "predictions": results,
            "summary": {
                "total_samples": len(results),
                "correct_predictions": sum(1 for r in results if r['correct']),
                "accuracy": sum(1 for r in results if r['correct']) / len(results)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Predictions saved to: {output_file}")
        print()


def main():
    """Main inference process"""
    print("🚀 STARTING INFERENCE PROCESS")
    print("=" * 70)
    print("Loading model state from JSON events...")
    print()
    
    # Create inference process
    inference = InferenceProcess()
    
    # Load split events
    inference.load_split_events('json_data/split_event')
    
    # Load leaf updates (all 10 updates)
    inference.load_leaf_updates('json_data/update_leaf', max_updates=10)
    
    # Make predictions
    results = inference.make_predictions(n_samples=5, seed=999)
    
    # Save results
    inference.save_predictions(results, 'json_data/inference_predictions.json')
    
    print("=" * 70)
    print("✅ INFERENCE PROCESS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
