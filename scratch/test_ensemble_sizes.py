import csv
import pickle
from river import tree
from river.ensemble import ADWINBaggingClassifier, SRPClassifier

def test_ensemble_sizes():
    # Base classifier
    base_model = tree.HoeffdingTreeClassifier()
    
    # Ensembles with 10 estimators (trees)
    ensembles = {
        "HoeffdingTreeClassifier (Baseline)": tree.HoeffdingTreeClassifier(),
        "ADWINBagging (10 x HT)": ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=10,
            seed=42
        ),
        "SRPClassifier (10 x HT)": SRPClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=10,
            seed=42
        ),
    }
    
    csv_path = r"c:\Users\muhammad.aulia\Downloads\river\AGR_a_real_test.csv"
    
    # Read samples
    samples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = {k: float(v) for k, v in row.items() if k != "class"}
            y = int(float(row["class"]))
            samples.append((x, y))
            if len(samples) >= 20000:
                break
                
    print(f"{'Model Name':<40} | {'Initial Size':<15} | {'After 5k':<15} | {'After 20k':<15}")
    print("-" * 90)
    
    for name, model in ensembles.items():
        initial_pkl = len(pickle.dumps(model))
        
        # Train 5k
        for x, y in samples[:5000]:
            model.learn_one(x, y)
        after_5k = len(pickle.dumps(model))
        
        # Train to 20k
        for x, y in samples[5000:20000]:
            model.learn_one(x, y)
        after_20k = len(pickle.dumps(model))
        
        print(f"{name:<40} | {initial_pkl:<15,} | {after_5k:<15,} | {after_20k:<15,}")

if __name__ == "__main__":
    test_ensemble_sizes()
