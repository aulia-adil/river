import csv
import pickle
from river import tree

def test_sizes_csv():
    # Only classification models since AGR dataset has 'class' as target
    models = {
        "HoeffdingTreeClassifier": tree.HoeffdingTreeClassifier(),
        "HoeffdingAdaptiveTreeClassifier": tree.HoeffdingAdaptiveTreeClassifier(),
        "ExtremelyFastDecisionTreeClassifier": tree.ExtremelyFastDecisionTreeClassifier(),
        "LASTClassifier": tree.LASTClassifier(),
        "SGTClassifier": tree.SGTClassifier(),
    }
    
    csv_path = r"c:\Users\muhammad.aulia\Downloads\river\AGR_a_real_test.csv"
    
    # Read samples from CSV
    samples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse features (floats) and target (int/str)
            x = {k: float(v) for k, v in row.items() if k != "class"}
            # The class label can be 0 or 1. Let's make it int.
            y = int(float(row["class"]))
            samples.append((x, y))
            if len(samples) >= 50000:
                break
                
    print(f"{'Model Name':<40} | {'Initial':<12} | {'After 10k':<12} | {'After 50k':<12}")
    print("-" * 85)
    
    for name, model in models.items():
        initial_pkl = len(pickle.dumps(model))
        
        # Train 10k
        for x, y in samples[:10000]:
            model.learn_one(x, y)
        after_10k = len(pickle.dumps(model))
        
        # Train remaining to reach 50k
        for x, y in samples[10000:50000]:
            model.learn_one(x, y)
        after_50k = len(pickle.dumps(model))
        
        print(f"{name:<40} | {initial_pkl:<12,} | {after_10k:<12,} | {after_50k:<12,}")

if __name__ == "__main__":
    test_sizes_csv()
