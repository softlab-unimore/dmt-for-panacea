import os

import numpy as np
import math
from collections import Counter
from sklearn.metrics import silhouette_score, pairwise_distances
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import time
import json

def compute_entropy(labels):
    # y is a list of class labels in the current node
    class_labels = np.unique(labels)
    entropy = 0

    for class_label in class_labels:
        p_i = len(labels[labels == class_label]) / len(labels)  # proportion of class i
        if p_i > 0:  # Only calculate if p_i is greater than 0
            entropy -= p_i * math.log2(p_i)  # entropy formula

    return entropy

# TODO: IN CASE THERE IS A SINGLE CLASS WE GIVE 1 BECAUSE WE NEED TO PRIVILEGE SINGULAR CLASS
def compute_silhouette(data, labels, batch_size=17000):
    if len(np.unique(labels)) == 1:
        # If there is only one class, Silhouette score is 0 BUT FOR OUR PURPOSES WE SET TO 1!!!
        return 1
    elif len(data) == 2 and len(np.unique(labels)) == 2:
        # If there is only two classes and two points, Silhouette score is -1
        return -1
    else:
        print(' --> Compute Silhouette')
        N = len(labels)
        unique_labels = np.unique(labels)
        silhouette_scores = np.zeros(N)

        batch_size = min(batch_size, N)
        for i in tqdm(range(0, N, batch_size)):
            X_batch = data[i:i + batch_size]
            labels_batch = labels[i:i + batch_size]

            distances_batch = pairwise_distances(X_batch, data, metric='euclidean')

            for j in range(len(X_batch)):
                current_label = labels_batch[j]

                # Intra cluster distances
                same_cluster_mask = np.where(labels == current_label, True, False)
                same_cluster_mask[i + j] = False

                if np.any(same_cluster_mask):
                    a_i = np.mean(distances_batch[j, same_cluster_mask])
                else:
                    a_i = 0

                # Inter cluster distances
                b_i = np.inf
                for other_label in unique_labels:
                    if other_label == current_label:
                        continue
                    other_cluster_mask = (labels == other_label)
                    if np.any(other_cluster_mask):
                        b_i = min(b_i, np.mean(distances_batch[j, other_cluster_mask]))

                # Compute s(i)
                silhouette_scores[i + j] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        return np.mean(silhouette_scores)

def compute_homogeneity(labels):
    # y is a list of class labels in the current node
    # Count occurrences of each class
    count = Counter(labels)

    # Get the maximum count of any class
    n_max = max(count.values())

    # Total number of instances
    N = len(labels)

    # Calculate the homogeneity score
    score = n_max / N

    return score


def get_metrics(results, test_time, total_time) -> dict:
    metrics = {
        'f1_score_macro': f1_score(results['Label'], results['Predicted'], average='macro'),
        'f1_score_micro': f1_score(results['Label'], results['Predicted'], average='micro'),
        'acc': accuracy_score(results['Label'], results['Predicted']),
        'precision': precision_score(results['Label'], results['Predicted'], average='macro'),
        'recall': recall_score(results['Label'], results['Predicted'], average='macro'),
        'test_time': test_time,
        'total_time': time.time() - total_time
    }

    print('------------------------------------')
    print('F1 Score Macro: ', metrics['f1_score_macro'])
    print('F1 Score Micro: ', metrics['f1_score_micro'])
    print('Accuracy: ', metrics['acc'])
    print('Precision: ', metrics['precision'])
    print('Recall: ', metrics['recall'])
    print('Test Time: ', metrics['test_time'])
    print('Total Time: ', metrics['total_time'])
    print('------------------------------------')

    return metrics

def save_parameters(args, dir_path):
    with open(os.path.join(dir_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def save_metrics(metrics: dict, dir_path: str, conf: str = ''):
    with open(f'{dir_path}/metrics{"_" + conf if conf != "" else ""}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def get_max_depth(node):
    if node.is_leaf():
        return 1
    return 1 + max(get_max_depth(node.left), get_max_depth(node.right))