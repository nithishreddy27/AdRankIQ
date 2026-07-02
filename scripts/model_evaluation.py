import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, auc
)
import json

class ModelEvaluator:
    def __init__(self):
        self.results = {}
        
    def comprehensive_evaluation(self, y_true, y_pred_proba, model_name, threshold=0.5):
        """
        Comprehensive evaluation of binary classification model
        """
        print(f"\n{'='*50}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*50}")
        
        # Basic metrics
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        # Binary predictions
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Store results
        metrics = {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1_score,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred_proba,
            'binary_predictions': y_pred_binary,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_thresholds},
            'pr_curve': {'precision': precision, 'recall': recall, 'thresholds': pr_thresholds}
        }
        
        # Print results
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"AUC-PR: {pr_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_score:.4f}")
        print(f"Recall: {recall_score:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Specificity: {specificity:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn:,}, FP: {fp:,}")
        print(f"FN: {fn:,}, TP: {tp:,}")
        
        return metrics
    
    def plot_roc_curves(self, figsize=(12, 8)):
        """Plot ROC curves for all models"""
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            auc_score = results['metrics']['auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, figsize=(12, 8)):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=figsize)
        
        for model_name, results in self.results.items():
            precision = results['pr_curve']['precision']
            recall = results['pr_curve']['recall']
            pr_auc = results['metrics']['pr_auc']
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})', linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('pr_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, figsize=(15, 5)):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            metrics = results['metrics']
            cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                          [metrics['false_negatives'], metrics['true_positives']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Predicted 0', 'Predicted 1'],
                       yticklabels=['Actual 0', 'Actual 1'],
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.4f}')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, figsize=(12, 8)):
        """Plot metrics comparison bar chart"""
        metrics_to_plot = ['auc', 'pr_auc', 'accuracy', 'precision', 'recall', 'f1_score']
        
        model_names = list(self.results.keys())
        metrics_data = {metric: [self.results[model]['metrics'][metric] for model in model_names] 
                       for metric in metrics_to_plot}
        
        df = pd.DataFrame(metrics_data, index=model_names)
        
        ax = df.plot(kind='bar', figsize=figsize, width=0.8)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_file='model_evaluation_report.json'):
        """Generate comprehensive evaluation report"""
        report = {
            'evaluation_summary': {},
            'model_comparison': {},
            'recommendations': []
        }
        
        # Summary statistics
        all_aucs = [results['metrics']['auc'] for results in self.results.values()]
        report['evaluation_summary'] = {
            'total_models_evaluated': len(self.results),
            'best_auc': max(all_aucs),
            'worst_auc': min(all_aucs),
            'average_auc': np.mean(all_aucs),
            'auc_std': np.std(all_aucs)
        }
        
        # Model comparison
        for model_name, results in self.results.items():
            report['model_comparison'][model_name] = results['metrics']
        
        # Recommendations
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['auc'])
        report['recommendations'].append(f"Best performing model: {best_model}")
        
        if report['evaluation_summary']['best_auc'] >= 0.91:
            report['recommendations'].append("Target AUC of 0.91 achieved!")
        else:
            report['recommendations'].append("Target AUC of 0.91 not achieved. Consider hyperparameter tuning or feature engineering.")
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nEvaluation report saved to {output_file}")
        return report
    
    def create_evaluation_dashboard(self):
        """Create a comprehensive evaluation dashboard"""
        print("Creating evaluation dashboard...")
        
        # Create subplots
        fig = plt.figure(figsize=(20, 15))
        
        # ROC Curves
        plt.subplot(2, 3, 1)
        for model_name, results in self.results.items():
            fpr = results['roc_curve']['fpr']
            tpr = results['roc_curve']['tpr']
            auc_score = results['metrics']['auc']
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # PR Curves
        plt.subplot(2, 3, 2)
        for model_name, results in self.results.items():
            precision = results['pr_curve']['precision']
            recall = results['pr_curve']['recall']
            pr_auc = results['metrics']['pr_auc']
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Metrics Comparison
        plt.subplot(2, 3, 3)
        metrics_to_plot = ['auc', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        for i, model_name in enumerate(model_names):
            values = [self.results[model_name]['metrics'][metric] for metric in metrics_to_plot]
            plt.bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Metrics Comparison')
        plt.xticks(x + width/2, metrics_to_plot, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Feature Importance (if available)
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, 'Feature Importance\n(Model Specific)', 
                ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
        plt.title('Feature Importance')
        
        # Model Performance Summary
        plt.subplot(2, 3, 5)
        summary_text = "Model Performance Summary\n\n"
        for model_name, results in self.results.items():
            metrics = results['metrics']
            summary_text += f"{model_name}:\n"
            summary_text += f"  AUC: {metrics['auc']:.4f}\n"
            summary_text += f"  Precision: {metrics['precision']:.4f}\n"
            summary_text += f"  Recall: {metrics['recall']:.4f}\n\n"
        
        plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        plt.title('Performance Summary')
        
        # Recommendations
        plt.subplot(2, 3, 6)
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['auc'])
        best_auc = self.results[best_model]['metrics']['auc']
        
        recommendations = f"Recommendations:\n\n"
        recommendations += f"• Best Model: {best_model}\n"
        recommendations += f"• Best AUC: {best_auc:.4f}\n"
        recommendations += f"• Target AUC (0.91): {'✓ Achieved' if best_auc >= 0.91 else '✗ Not Achieved'}\n\n"
        
        if best_auc >= 0.91:
            recommendations += "• Model ready for production\n"
            recommendations += "• Consider A/B testing\n"
        else:
            recommendations += "• Consider hyperparameter tuning\n"
            recommendations += "• Add more features\n"
            recommendations += "• Collect more training data\n"
        
        plt.text(0.1, 0.9, recommendations, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top')
        plt.axis('off')
        plt.title('Recommendations')
        
        plt.tight_layout()
        plt.savefig('evaluation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Evaluation dashboard created and saved as 'evaluation_dashboard.png'")

if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator()
    
    # This would normally be called with actual model predictions
    print("Model Evaluator ready for use!")
    print("Usage:")
    print("evaluator.comprehensive_evaluation(y_true, y_pred_proba, 'Model Name')")
    print("evaluator.plot_roc_curves()")
    print("evaluator.create_evaluation_dashboard()")
