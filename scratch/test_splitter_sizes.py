import pickle
from river import datasets, tree

def test_splitters():
    models = {
        "HT Regressor (Default TEBST)": tree.HoeffdingTreeRegressor(),
        "HT Regressor (QOSplitter)": tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.QOSplitter()
        ),
        "HT Regressor (EBSTSplitter)": tree.HoeffdingTreeRegressor(
            splitter=tree.splitter.EBSTSplitter()
        ),
        "HT Classifier (Default Gaussian)": tree.HoeffdingTreeClassifier(),
        "HT Classifier (HistogramSplitter)": tree.HoeffdingTreeClassifier(
            splitter=tree.splitter.HistogramSplitter()
        ),
        "HT Classifier (ExhaustiveSplitter)": tree.HoeffdingTreeClassifier(
            splitter=tree.splitter.ExhaustiveSplitter()
        ),
    }
    
    dataset = datasets.synth.Agrawal(seed=42)
    
    print(f"{'Model + Splitter':<45} | {'Initial':<12} | {'After 10k':<12}")
    print("-" * 75)
    
    for name, model in models.items():
        initial_pkl = len(pickle.dumps(model))
        for x, y in dataset.take(10000):
            if "Regressor" in name:
                model.learn_one(x, float(y))
            else:
                model.learn_one(x, y)
        after_10k = len(pickle.dumps(model))
        print(f"{name:<45} | {initial_pkl:<12,} | {after_10k:<12,}")

if __name__ == "__main__":
    test_splitters()
