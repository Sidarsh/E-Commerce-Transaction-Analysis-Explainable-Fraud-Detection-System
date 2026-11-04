# run_evaluation.py
from evaluation import run_complete_evaluation
import pandas as pd
import pickle

def load_model_and_data():
    """Load your trained model and test data"""
    # Load your model (you'll need to save it first)
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    # Load your test data
    test_data = pd.read_csv('test_data.csv')
    
    return model, label_encoders, test_data

if __name__ == "__main__":
    # Load model and data
    model, label_encoders, test_data = load_model_and_data()
    
    # Define feature columns
    feature_columns = ['transaction_amount', 'payment_method', 'location', 'product_name']
    
    # Run evaluation
    evaluator, report = run_complete_evaluation(
        model, label_encoders, test_data, feature_columns
    )
    
    print("✅ Evaluation completed! Check generated files:")
    print("   - confusion_matrix.png")
    print("   - roc_curve.png") 
    print("   - precision_recall_curve.png")
    print("   - model_evaluation_report.json")
    