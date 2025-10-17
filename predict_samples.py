"""
Prediction script - loads the trained model state and makes predictions on 5 samples
"""
import json
from river import tree
from river.datasets import synth

def load_trained_model():
    """
    Train a model to the same state as the captured data
    """
    # Create model with callbacks (we don't need them for prediction)
    model = tree.HoeffdingTreeClassifier(
        grace_period=200,
        delta=1e-5,
        leaf_prediction='mc',
        nb_threshold=10,
        leaf_update_threshold=1
    )
    
    # Train on the same data as the capture script
    dataset = synth.Agrawal(classification_function=0, seed=42)
    dataset = dataset.take(610)  # Same as training in capture script
    
    print("Training model to match captured state...")
    for i, (x, y) in enumerate(dataset):
        model.learn_one(x, y)
        if (i + 1) % 100 == 0:
            print(f"  Trained on {i + 1} instances...")
    
    print(f"✅ Model trained on 610 instances")
    print(f"   Nodes: {model.n_nodes}")
    print(f"   Height: {model.height}")
    print(f"   Active leaves: {model.n_active_leaves}")
    print()
    
    return model

def make_predictions():
    """
    Make predictions on 5 new samples and save results
    """
    # Load the trained model
    print("=" * 60)
    print("LOADING TRAINED MODEL")
    print("=" * 60)
    model = load_trained_model()
    
    print("=" * 60)
    print("MODEL READY FOR PREDICTIONS")
    print("=" * 60)
    
    # Generate 5 new samples for prediction (using different seed)
    dataset = synth.Agrawal(classification_function=0, seed=999)
    test_samples = list(dataset.take(5))
    
    # Make predictions
    results = []
    
    print("=" * 60)
    print("MAKING PREDICTIONS")
    print("=" * 60)
    
    for i, (x, y_true) in enumerate(test_samples, 1):
        # Get prediction and probability
        y_pred = model.predict_one(x)
        y_proba = model.predict_proba_one(x)
        
        # Store result
        result = {
            "sample_id": i,
            "features": {
                "salary": x['salary'],
                "commission": x['commission'],
                "age": x['age'],
                "elevel": x['elevel'],
                "car": x['car'],
                "zipcode": x['zipcode'],
                "hvalue": x['hvalue'],
                "hyears": x['hyears'],
                "loan": x['loan']
            },
            "true_label": int(y_true),
            "predicted_label": int(y_pred) if y_pred is not None else None,
            "prediction_probabilities": {
                str(k): float(v) for k, v in y_proba.items()
            } if y_proba else {},
            "correct": bool(y_pred == y_true) if y_pred is not None else False
        }
        
        results.append(result)
        
        # Print to console
        print(f"\nSample #{i}:")
        print(f"  Age: {x['age']:.1f}, Salary: {x['salary']:.0f}, Loan: {x['loan']:.0f}")
        print(f"  True label: {y_true}")
        print(f"  Predicted label: {y_pred}")
        print(f"  Probabilities: {y_proba}")
        print(f"  Correct: {y_pred == y_true}")
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    
    # Create summary
    summary = {
        "model_info": {
            "n_nodes": model.n_nodes,
            "height": model.height,
            "n_active_leaves": model.n_active_leaves,
            "n_inactive_leaves": model.n_inactive_leaves
        },
        "predictions": results,
        "summary": {
            "total_samples": len(results),
            "correct_predictions": sum(1 for r in results if r['correct']),
            "accuracy": accuracy
        }
    }
    
    # Save to file
    output_file = 'json_data/predictions.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Correct predictions: {sum(1 for r in results if r['correct'])}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nResults saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    make_predictions()
