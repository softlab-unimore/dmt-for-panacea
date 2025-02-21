from copy import deepcopy
from collections import Counter
import pandas as pd
from sklearn.metrics import silhouette_score
import numpy as np
import math

# Classe Nodo
class TreeNode:
    def __init__(self, data=None, labels=None, depth=0, max_depth=None):
        self.data = data
        self.labels = labels
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_threshold = None
        self.depth = depth
        self.max_depth = max_depth
        self.metrics = self.compute_metrics()

    def is_leaf(self):
        return self.left is None and self.right is None

    def set_feature_threshold(self, feature, threshold):
        self.split_feature = feature
        self.split_threshold = threshold

    def compute_metrics(self):
        # TODO: Compute metrics
        self.metrics = {}
        self.metrics["silhouette"] = self.__compute_silhouette()
        self.metrics["homogeneity"] = self.__homogeneity()
        return self.metrics

    def update_data(self, data, labels):
        if self.data is None:
            self.data = data
            self.labels = labels
        else:
            self.data = pd.concat([self.data, data])
            self.labels = np.concatenate([self.labels, labels])
        self.metrics = self.compute_metrics()

    # TODO: IN CASE THERE IS A SINGLE CLASS WE GIVE 1 BECAUSE WE NEED TO PRIVILEGE SINGULAR CLASS

    def __compute_silhouette(self):
        if len(np.unique(self.labels)) == 1:
            # Silhouette score is 0 BUT FOR OUR PURPOSES WE SET TO 1!!!
            return 1
        elif len(self.data) == 2 and len(np.unique(self.labels)) == 2:
            # If there is only two classes and two points, Silhouette score is -1
            return -1
        return silhouette_score(self.data, self.labels)

    def __entropy(self):
        # y is a list of class labels in the current node
        class_labels = np.unique(self.labels)
        entropy = 0

        for class_label in class_labels:
            p_i = len(self.labels[self.labels == class_label]) / len(self.labels)  # proportion of class i
            if p_i > 0:  # Only calculate if p_i is greater than 0
                entropy -= p_i * math.log2(p_i)  # entropy formula

        return entropy

    def __homogeneity(self):
        # y is a list of class labels in the current node
        # Count occurrences of each class
        count = Counter(self.labels)

        # Get the maximum count of any class
        n_max = max(count.values())

        # Total number of instances
        N = len(self.labels)

        # Calculate the homogeneity score
        score = n_max / N

        return score