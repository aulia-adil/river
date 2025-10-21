"""Debug script to understand why class 0 is missing from the right child"""

from river import tree
from river.datasets import synth

# Track what happens in GaussianSplitter
split_occurred = False

def split_callback(split_info):
    global split_occurred
    split_occurred = True
    
    original_leaf = split_info['original_leaf']
    new_split_node = split_info['new_split_node']
    new_leaves = split_info['new_leaves']
    
    print("\n" + "="*80)
    print("SPLIT ANALYSIS")
    print("="*80)
    
    # Check the original leaf's splitters
    if hasattr(original_leaf, 'splitters'):
        print(f"\nOriginal leaf stats: {original_leaf.stats}")
        print(f"Original leaf has {len(original_leaf.splitters)} splitters")
        
        for feature_name, splitter in original_leaf.splitters.items():
            print(f"\n📊 Feature: {feature_name}")
            print(f"   Splitter type: {type(splitter).__name__}")
            
            if hasattr(splitter, '_att_dist_per_class'):
                print(f"   Classes tracked: {list(splitter._att_dist_per_class.keys())}")
                for cls, dist in splitter._att_dist_per_class.items():
                    min_val = splitter._min_per_class[cls]
                    max_val = splitter._max_per_class[cls]
                    print(f"   Class {cls}: n={dist.n_samples:.2f}, min={min_val:.2f}, max={max_val:.2f}")
                    
                # If this is the split feature, show the calculation
                if feature_name == new_split_node.feature:
                    threshold = new_split_node.threshold
                    print(f"\n   ⭐ THIS IS THE SPLIT FEATURE!")
                    print(f"   Split threshold: {threshold:.2f}")
                    print(f"\n   Analyzing split at {threshold:.2f}:")
                    
                    for cls, dist in splitter._att_dist_per_class.items():
                        min_val = splitter._min_per_class[cls]
                        max_val = splitter._max_per_class[cls]
                        
                        print(f"   Class {cls}:")
                        if threshold < min_val:
                            print(f"      All {dist.n_samples:.2f} samples → RIGHT (threshold < min {min_val:.2f})")
                        elif threshold >= max_val:
                            print(f"      All {dist.n_samples:.2f} samples → LEFT (threshold >= max {max_val:.2f})")
                        else:
                            left_count = dist.cdf(threshold) * dist.n_samples
                            right_count = dist.n_samples - left_count
                            print(f"      {left_count:.2f} samples → LEFT (via CDF)")
                            print(f"      {right_count:.2f} samples → RIGHT")
    
    # Check the new leaves
    for idx, leaf in enumerate(new_leaves):
        print(f"\n🍃 New Leaf {idx} (node_id={leaf.node_id}):")
        print(f"   Stats: {leaf.stats}")
        print(f"   Total weight: {leaf.total_weight}")

model = tree.HoeffdingTreeClassifier(
    grace_period=200,
    split_callback=split_callback,
    leaf_prediction='nba'
)

dataset = synth.Agrawal(classification_function=0, seed=42)

instance_count = 0
for x, y in dataset:
    instance_count += 1
    model.learn_one(x, y)
    
    if split_occurred:
        print(f"\n✅ Split occurred after {instance_count} instances")
        break
    
    if instance_count >= 10000:
        break

print("\n" + "="*80)
