"""
Test to show what's inside the obs (splitter) objects during Naive Bayes prediction.
This will reveal the detailed statistics maintained by each feature splitter.
"""

import sys
import os
sys.path.append('/root/river')

# Create some sample data to trigger the debug prints
from river import datasets
from river.tree import HoeffdingTreeClassifier

def test_splitter_objects():
    print("🧪 TESTING SPLITTER OBJECTS (obs) IN NAIVE BAYES PREDICTION")
    print("=" * 60)
    
    # Create a tree
    tree = HoeffdingTreeClassifier(
        grace_period=5,  # Small grace period to force splits quickly
        split_criterion='info_gain',
        leaf_prediction='nb'  # Use Naive Bayes predictions
    )
    
    # Generate some training data
    dataset = datasets.Phishing()
    
    print("📚 Training the tree...")
    # Train with a few samples to build some statistics
    for i, (x, y) in enumerate(dataset):
        if i >= 20:  # Just use first 20 samples
            break
        tree.learn_one(x, y)
        
        # After some training, make a prediction to trigger Naive Bayes
        if i >= 10:  # Start predicting after some training
            print(f"\n🔮 PREDICTION #{i-9} (sample {i+1}):")
            print(f"Input features: {dict(list(x.items())[:3])}...")  # Show first 3 features
            prediction = tree.predict_proba_one(x)
            print(f"Final prediction: {prediction}")
            print("-" * 50)
            
            if i >= 12:  # Only show a few predictions to avoid too much output
                break

if __name__ == "__main__":
    test_splitter_objects()