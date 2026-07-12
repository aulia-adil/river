import csv
import pickle
from river import tree

def test_large_growth():
    csv_path = r"c:\Users\muhammad.aulia\Downloads\river\AGR_a_real_test.csv"
    
    # Initialize classifiers
    efdt = tree.ExtremelyFastDecisionTreeClassifier()
    ht = tree.HoeffdingTreeClassifier()
    
    # Read samples
    samples = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = {k: float(v) for k, v in row.items() if k != "class"}
            y = int(float(row["class"]))
            samples.append((x, y))
            if len(samples) >= 450000:
                break
                
    print(f"{'Samples':<10} | {'HT Size (MB)':<15} | {'EFDT Size (MB)':<15} | {'Ratio (EFDT/HT)':<18}")
    print("-" * 65)
    
    increments = [50000 * i for i in range(1, 10)] # 50k to 450k
    
    # Train and measure
    current_idx = 0
    for limit in increments:
        for x, y in samples[current_idx:limit]:
            efdt.learn_one(x, y)
            ht.learn_one(x, y)
        current_idx = limit
        
        ht_size = len(pickle.dumps(ht)) / (1024 * 1024) # MB
        efdt_size = len(pickle.dumps(efdt)) / (1024 * 1024) # MB
        ratio = efdt_size / ht_size if ht_size > 0 else 0
        
        print(f"{limit:<10,} | {ht_size:<15.4f} | {efdt_size:<15.4f} | {ratio:<18.4f}")

if __name__ == "__main__":
    test_large_growth()
