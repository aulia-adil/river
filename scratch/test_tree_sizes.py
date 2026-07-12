import pickle
from river import datasets, tree

def test_sizes():
    models = {
        "HoeffdingTreeClassifier": tree.HoeffdingTreeClassifier(),
        "HoeffdingAdaptiveTreeClassifier": tree.HoeffdingAdaptiveTreeClassifier(),
        "ExtremelyFastDecisionTreeClassifier": tree.ExtremelyFastDecisionTreeClassifier(),
        "LASTClassifier": tree.LASTClassifier(),
        "SGTClassifier": tree.SGTClassifier(),
        "HoeffdingTreeRegressor": tree.HoeffdingTreeRegressor(),
        "HoeffdingAdaptiveTreeRegressor": tree.HoeffdingAdaptiveTreeRegressor(),
        "SGTRegressor": tree.SGTRegressor(),
        "iSOUPTreeRegressor": tree.iSOUPTreeRegressor(),
    }
    
    # Generate some synthetic data
    # Agrawal generator generates binary classification data with 9 attributes
    dataset = datasets.synth.Agrawal(seed=42)
    
    print(f"{'Model Name':<40} | {'Initial Size':<15} | {'After 2k':<15} | {'After 10k':<15}")
    print("-" * 95)
    
    for name, model in models.items():
        initial_pkl = len(pickle.dumps(model))
        
        # Train on 2,000 samples
        for x, y in dataset.take(2000):
            if "iSOUPTreeRegressor" in name:
                model.learn_one(x, {"target": float(y)})
            elif "Regressor" in name:
                model.learn_one(x, float(y))
            else:
                model.learn_one(x, y)
        
        after_2k = len(pickle.dumps(model))
        
        # Train on 8,000 more samples (total 10,000)
        for x, y in dataset.take(10000):
            if "iSOUPTreeRegressor" in name:
                model.learn_one(x, {"target": float(y)})
            elif "Regressor" in name:
                model.learn_one(x, float(y))
            else:
                model.learn_one(x, y)
                
        after_10k = len(pickle.dumps(model))
        print(f"{name:<40} | {initial_pkl:<15,} | {after_2k:<15,} | {after_10k:<15,}")

if __name__ == "__main__":
    test_sizes()
