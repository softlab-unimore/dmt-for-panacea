import numpy as np
import math
from collections import Counter
from sklearn.metrics import silhouette_score


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
def compute_silhouette(data, labels):
    if len(np.unique(labels)) == 1:
        # If there is only one class, Silhouette score is 0 BUT FOR OUR PURPOSES WE SET TO 1!!!
        return 1
    elif len(data) == 2 and len(np.unique(labels)) == 2:
        # If there is only two classes and two points, Silhouette score is -1
        return -1
    return silhouette_score(data, labels)

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
