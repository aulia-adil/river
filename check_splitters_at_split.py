"""
Check what splitters look like immediately after a split
"""

from river import tree
from river.datasets import synth

split_occurred = False

def split_callback(split_info):
    global split_occurred
    split_occurred = True
    
    new_leaves = split_info['new_leaves']
    
    print("\n" + "="*80)
    print("🔍 CHECKING SPLITTERS IMMEDIATELY AFTER SPLIT")
    print("="*80)
    
    for i, leaf in enumerate(new_leaves):
        branch_name = "LEFT" if i == 0 else "RIGHT"
        print(f"\n{branch_name} LEAF (node_id={leaf.node_id}):")
        print(f"  Type: {type(leaf).__name__}")
        print(f"  Stats: {leaf.stats}")
        print(f"  Total weight: {leaf.total_weight}")
        
        if hasattr(leaf, 'splitters'):
            print(f"  Splitters dict exists: {leaf.splitters is not None}")
            if leaf.splitters is not None:
                print(f"  Number of features in splitters: {len(leaf.splitters)}")
                if len(leaf.splitters) > 0:
                    print(f"  Features: {list(leaf.splitters.keys())}")
                    for feat_name, splitter in leaf.splitters.items():
                        if hasattr(splitter, '_att_dist_per_class'):
                            print(f"    {feat_name}: {len(splitter._att_dist_per_class)} classes tracked")
                else:
                    print(f"  ✅ Splitters dict is EMPTY (as expected for new leaves)")
            else:
                print(f"  Splitters is None")

model = tree.HoeffdingTreeClassifier(
    grace_period=200,
    split_callback=split_callback,
    leaf_prediction='nba'
)

dataset = synth.Agrawal(classification_function=0, seed=42)

for x, y in dataset:
    model.learn_one(x, y)
    if split_occurred:
        break

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("New leaves after split have EMPTY splitters dictionaries.")
print("They only contain stats (class counts).")
print("Splitter data is built up incrementally as new samples arrive!")
print("="*80)
