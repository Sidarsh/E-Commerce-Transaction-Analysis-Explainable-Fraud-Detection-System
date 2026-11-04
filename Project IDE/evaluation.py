# evaluation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import pickle
import json
from datetime import datetime

class FraudDetectionEvaluator:
    """
    Comprehensive evaluation class for E-commerce Fraud Detection Model
    """
    
    def __init__(self, model, label_encoders, test_data, feature_columns):
        self.model = model
        self.label_encoders = label_encoders
        self.test_data = test_data
        self.feature_columns = feature_columns
        self.results = {}
        self.predictions = None
        self.probabilities = None
        
    def create_enhanced_features(self, data):
        """Replicate the feature engineering from the main model"""
        df = data.copy()
        
        # Amount-based risk flags
        df['is_very_low_amount'] = (df['transaction_amount'] <= 100).astype(int)
        df['is_very_high_amount'] = (df['transaction_amount'] > 50000).astype(int)
        df['is_suspicious_amount'] = ((df['transaction_amount'] <= 10) | 
                                     (df['transaction_amount'] % 1 != 0) | 
                                     (df['transaction_amount'] == 0)).astype(int)
        
        # High-risk product categories
        high_value_products = ['Smartphone', 'Laptop', 'Television', 'Camera', 'Gaming Console']
        df['is_high_value_product'] = df['product_name'].isin(high_value_products).astype(int)
        
        # High-risk payment methods
        high_risk_payments = ['Cash on Delivery']
        df['is_high_risk_payment'] = df['payment_method'].isin(high_risk_payments).astype(int)
        
        # High-risk locations
        high_risk_locations = ['Delhi', 'Mumbai', 'Kolkata']
        df['is_high_risk_location'] = df['location'].isin(high_risk_locations).astype(int)
        
        return df
    
    def prepare_features(self, data):
        """Prepare features for model prediction"""
        enhanced_data = self.create_enhanced_features(data)
        
        # Select features
        core_features = ['transaction_amount', 'payment_method', 'location', 'product_name']
        enhanced_features = ['is_very_low_amount', 'is_very_high_amount', 'is_suspicious_amount',
                            'is_high_value_product', 'is_high_risk_payment', 'is_high_risk_location']
        
        features = enhanced_data[core_features + enhanced_features].copy()
        
        # Encode categorical variables
        categorical_cols = ['payment_method', 'location', 'product_name']
        for col in categorical_cols:
            le = self.label_encoders[col]
            # Handle unseen categories
            features[col] = features[col].apply(lambda x: x if x in le.classes_ else 'unknown')
            if 'unknown' not in le.classes_:
                # Add unknown class if not present
                le.classes_ = np.append(le.classes_, 'unknown')
            features[col] = le.transform(features[col].astype(str))
        
        return features
    
    def predict_with_rules(self, features_df, input_data):
        """Replicate the rule-based prediction logic"""
        predictions = self.model.predict(features_df)
        probabilities = self.model.predict_proba(features_df)[:, 1]
        
        # Apply rule-based overrides
        for i, (idx, row) in enumerate(input_data.iterrows()):
            high_value_threshold = 50000
            
            # Rule 1: High-value product + Cash on Delivery + High amount
            if (row['is_high_value_product'] and 
                row['is_high_risk_payment'] and 
                row['transaction_amount'] > high_value_threshold):
                predictions[i] = 1
                probabilities[i] = max(probabilities[i], 0.90)
            
            # Rule 2: High-value + Cash on Delivery + High-risk location
            elif (row['is_high_value_product'] and 
                  row['is_high_risk_payment'] and 
                  row['is_high_risk_location']):
                predictions[i] = 1
                probabilities[i] = max(probabilities[i], 0.85)
            
            # Rule 3: Very low amount for high-value products
            elif (row['transaction_amount'] <= 1 and 
                  row['is_high_value_product']):
                predictions[i] = 1
                probabilities[i] = max(probabilities[i], 0.95)
            
            # Rule 4: High amount + Cash on Delivery
            elif (row['transaction_amount'] > high_value_threshold and 
                  row['is_high_risk_payment']):
                predictions[i] = 1
                probabilities[i] = max(probabilities[i], 0.80)
            
            # Rule 5: Multiple risk factors combined
            risk_factors_count = (row['is_high_value_product'] + 
                                 row['is_high_risk_payment'] + 
                                 row['is_high_risk_location'] +
                                 row['is_very_high_amount'])
            
            if risk_factors_count >= 3:
                predictions[i] = 1
                probabilities[i] = max(probabilities[i], 0.75)
        
        return predictions, probabilities
    
    def evaluate_model(self, threshold=0.5):
        """Comprehensive model evaluation"""
        print("🔍 Starting Comprehensive Model Evaluation...")
        
        # Prepare test features
        X_test = self.prepare_features(self.test_data)
        y_test = self.test_data['is_fraud']
        
        # Get predictions with rule-based overrides
        self.predictions, self.probabilities = self.predict_with_rules(X_test, self.test_data)
        
        # Convert probabilities to binary predictions
        binary_predictions = (self.probabilities >= threshold).astype(int)
        
        # Store results
        self.results = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_size': len(self.test_data),
            'fraud_rate': y_test.mean(),
            'threshold': threshold
        }
        
        # Calculate metrics
        self._calculate_basic_metrics(y_test, binary_predictions)
        self._calculate_advanced_metrics(y_test, self.probabilities)
        self._calculate_business_metrics(y_test, binary_predictions)
        
        return self.results
    
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Calculate basic classification metrics"""
        self.results['accuracy'] = accuracy_score(y_true, y_pred)
        self.results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.results['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        self.results['true_negatives'] = int(tn)
        self.results['false_positives'] = int(fp)
        self.results['false_negatives'] = int(fn)
        self.results['true_positives'] = int(tp)
    
    def _calculate_advanced_metrics(self, y_true, y_prob):
        """Calculate advanced metrics"""
        self.results['roc_auc'] = roc_auc_score(y_true, y_prob)
        self.results['average_precision'] = average_precision_score(y_true, y_prob)
        
        # Precision-Recall curve metrics
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        self.results['precision_recall_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
        
        # ROC curve metrics
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        self.results['roc_curve'] = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
    
    def _calculate_business_metrics(self, y_true, y_pred):
        """Calculate business-oriented metrics"""
        # Financial impact approximation
        avg_transaction_amount = self.test_data['transaction_amount'].mean()
        fraud_caught = self.results['true_positives']
        fraud_missed = self.results['false_negatives']
        false_alarms = self.results['false_positives']
        
        # Assuming average fraud amount is higher than legitimate transactions
        estimated_fraud_amount = avg_transaction_amount * 2  # Conservative estimate
        
        self.results['business_metrics'] = {
            'estimated_fraud_prevented': fraud_caught * estimated_fraud_amount,
            'estimated_fraud_missed': fraud_missed * estimated_fraud_amount,
            'false_alarm_cost': false_alarms * 50,  # Assuming $50 cost per false alarm
            'cost_savings_ratio': (fraud_caught * estimated_fraud_amount) / 
                                 (false_alarms * 50 + 0.001)  # Avoid division by zero
        }
    
    def generate_classification_report(self, y_true, y_pred):
        """Generate detailed classification report"""
        return classification_report(y_true, y_pred, output_dict=True)
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_roc_curve(self, y_true, y_prob, save_path=None):
        """Plot and save ROC curve"""
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_prob, save_path=None):
        """Plot and save Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def feature_importance_analysis(self, feature_names):
        """Analyze and plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.show()
            
            # Return feature importance dictionary
            feature_imp = dict(zip([feature_names[i] for i in indices], importances[indices]))
            self.results['feature_importance'] = feature_imp
            return feature_imp
        else:
            print("Model doesn't have feature_importances_ attribute")
            return None
    
    def evaluate_by_segment(self, y_true, y_pred, segment_data, segment_name):
        """Evaluate model performance by different segments"""
        segment_results = {}
        
        for segment_value in segment_data.unique():
            mask = segment_data == segment_value
            segment_accuracy = accuracy_score(y_true[mask], y_pred[mask])
            segment_precision = precision_score(y_true[mask], y_pred[mask], zero_division=0)
            segment_recall = recall_score(y_true[mask], y_pred[mask], zero_division=0)
            
            segment_results[segment_value] = {
                'accuracy': segment_accuracy,
                'precision': segment_precision,
                'recall': segment_recall,
                'sample_size': mask.sum()
            }
        
        self.results[f'segment_analysis_{segment_name}'] = segment_results
        return segment_results
    
    def generate_report(self, save_path=None):
        """Generate comprehensive evaluation report"""
        report = {
            'model_evaluation': self.results,
            'summary': self._generate_summary()
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _generate_summary(self):
        """Generate executive summary"""
        summary = {
            'overall_performance': 'Excellent' if self.results['roc_auc'] > 0.9 else 
                                  'Good' if self.results['roc_auc'] > 0.8 else 
                                  'Fair' if self.results['roc_auc'] > 0.7 else 'Poor',
            'fraud_detection_capability': 'Strong' if self.results['recall'] > 0.8 else 
                                         'Moderate' if self.results['recall'] > 0.6 else 'Weak',
            'precision_quality': 'High' if self.results['precision'] > 0.8 else 
                                'Moderate' if self.results['precision'] > 0.6 else 'Low',
            'recommendations': self._generate_recommendations()
        }
        return summary
    
    def _generate_recommendations(self):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if self.results['recall'] < 0.7:
            recommendations.append("Improve fraud recall - too many fraud cases are being missed")
        
        if self.results['precision'] < 0.6:
            recommendations.append("Reduce false positives - too many legitimate transactions are being flagged")
        
        if self.results['false_negatives'] > self.results['true_positives']:
            recommendations.append("Critical: Model is missing more fraud than it's catching")
        
        if not recommendations:
            recommendations.append("Model performance is satisfactory. Consider monitoring for data drift.")
        
        return recommendations

# Example usage and test function
def run_complete_evaluation(model, label_encoders, test_data, feature_columns):
    """
    Run complete evaluation pipeline
    """
    print("🚀 Starting Complete Model Evaluation Pipeline...")
    
    # Initialize evaluator
    evaluator = FraudDetectionEvaluator(model, label_encoders, test_data, feature_columns)
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_model(threshold=0.5)
    
    # Generate plots
    y_true = test_data['is_fraud']
    y_pred = (evaluator.probabilities >= 0.5).astype(int)
    
    print("\n📊 Generating Evaluation Visualizations...")
    evaluator.plot_confusion_matrix(y_true, y_pred, 'confusion_matrix.png')
    evaluator.plot_roc_curve(y_true, evaluator.probabilities, 'roc_curve.png')
    evaluator.plot_precision_recall_curve(y_true, evaluator.probabilities, 'precision_recall_curve.png')
    
    # Feature importance
    feature_names = evaluator.prepare_features(test_data).columns.tolist()
    evaluator.feature_importance_analysis(feature_names)
    
    # Segment analysis
    print("\n🔍 Performing Segment Analysis...")
    evaluator.evaluate_by_segment(y_true, y_pred, test_data['payment_method'], 'payment_method')
    evaluator.evaluate_by_segment(y_true, y_pred, test_data['location'], 'location')
    
    # Generate final report
    print("\n📋 Generating Comprehensive Report...")
    final_report = evaluator.generate_report('model_evaluation_report.json')
    
    # Print summary
    print("\n" + "="*50)
    print("🎯 EVALUATION SUMMARY")
    print("="*50)
    print(f"ROC-AUC Score: {results['roc_auc']:.3f}")
    print(f"Precision: {results['precision']:.3f}")
    print(f"Recall: {results['recall']:.3f}")
    print(f"F1-Score: {results['f1_score']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"True Positives: {results['true_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"False Positives: {results['false_positives']}")
    
    # Business impact
    biz_metrics = results['business_metrics']
    print(f"\n💰 BUSINESS IMPACT")
    print(f"Estimated Fraud Prevented: ₹{biz_metrics['estimated_fraud_prevented']:,.2f}")
    print(f"Estimated Fraud Missed: ₹{biz_metrics['estimated_fraud_missed']:,.2f}")
    print(f"False Alarm Costs: ₹{biz_metrics['false_alarm_cost']:,.2f}")
    print(f"Cost Savings Ratio: {biz_metrics['cost_savings_ratio']:.2f}")
    
    return evaluator, final_report

if __name__ == "__main__":
    # This section would be used for testing the evaluation
    print("Evaluation module for E-commerce Fraud Detection")
    print("Import and use FraudDetectionEvaluator class for comprehensive model evaluation")