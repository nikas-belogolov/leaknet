import numpy as np
import sklearn.metrics as metrics

"""
Bag-Level Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC

False Positive Rate for instances in normal bags.
True Positive Rate for instances in anomalous bags.

For Normal Bags:

    Enforce a zero-tolerance policy for false positives (strict accuracy).

    Use instance FPR to ensure no anomalies are predicted in normal bags.

For Anomalous Bags:

    Prioritize bag recall (minimize false negatives).

    Use instance recall and instance TPR to maximize detection of true anomalies.


"""

def calculate_bag_metrics(predictions, labels):
    """
    Calculate Bag Metrics - Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
    """
    
    classification_report = metrics.classification_report(labels, predictions, output_dict=True)
    return classification_report

def calculate_window_metrics(predictions, labels):
    """
    Calculate Window Metrics - False Positive Rate for normal bags, True Positive Rate for anomalous bags
    """
    
    false_positive_rates = []   # for normal bags
    true_positive_rates = []    # for anomalous bags
    
    for window_pred, bag_label in zip(predictions, labels):
        
        if bag_label == 0:
            pass
        else:
            recall_score = metrics.recall_score(np.full_like(window_pred, bag_label), window_pred)
            true_positive_rates.append(recall_score)
            
    return np.mean(false_positive_rates), np.mean(true_positive_rates)