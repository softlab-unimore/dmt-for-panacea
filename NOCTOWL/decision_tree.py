import oapackage
import random
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from NOCTOWL.TreeNode import TreeNode
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
import util


class DecisionTree:
    def __init__(self, max_depth=5, min_points_per_leaf=20, number_thresholds=3, dist_threshold=0.5, homogeneity_gain_threshold=0, ordinal_categories=[]):
        self.max_depth = max_depth
        self.min_points_per_leaf = min_points_per_leaf
        self.number_thresholds = number_thresholds
        self.dist_threshold = dist_threshold
        self.ordinal_categories = ordinal_categories
        self.homogeneity_gain_threshold = homogeneity_gain_threshold

    def partial_fit(self, node, data, labels=None):
        if node is None:
            node = TreeNode(data, labels, depth=0)
        elif not node.is_leaf():
            data_left, data_right, labels_left, labels_right = self.apply_split(data, node.split_feature, node.split_threshold, labels)

            node.left = self.partial_fit(node.left, data_left, labels_left)
            node.right = self.partial_fit(node.right, data_right, labels_right)

        else:
            node = self.build_tree(node, data, labels, depth=node.depth + 1)

        return node


    def split_data_side(self, node, data, labels):
        # Create a mask to select values below the threshold
        mask_below = data[node.split_feature] <= node.split_threshold

        # Create a mask to select values above the threshold
        mask_above = data[node.split_feature] > node.split_threshold
        # Divide the DataFrame into two based on the masks
        df_left = data[mask_below]
        labels_left = labels[mask_below]
        df_right = data[mask_above]
        labels_right = labels[mask_above]

        return df_left, df_right, labels_left, labels_right

    def build_tree(self, node, data, labels, depth):

        labels_values = np.unique(node.labels)

        # Calculate the centroid of the current node
        original_centroids = self.find_centroid(node)

        node_app = deepcopy(node)
        node_app.update_data(data, labels)

        if (labels_values.size != 1 or not np.array_equal(labels_values, np.unique(labels))) and depth <= self.max_depth:
            best_split = self.find_best_split(node_app.data, node_app.labels)
            if best_split:
                feature, threshold = best_split
                node.split_feature = feature
                node.split_threshold = threshold

                left_data, right_data, left_labels, right_labels = self.apply_split(node_app.data, feature, threshold, node_app.labels)
                # node.left = self.build_tree(node.left, left_data, left_labels, depth + 1)
                # node.right = self.build_tree(node.right, right_data, right_labels, depth + 1)
                node.left = TreeNode(left_data, left_labels, depth=depth, max_depth=self.max_depth)
                node.right = TreeNode(right_data, right_labels, depth=depth, max_depth=self.max_depth)
                if self.evaluate_split(node_app, node.left, node.right):
                    print('--> Splitting')
                    return node
                else:
                    node = self.check_distance_from_centroids(data, labels, node, node_app, original_centroids)

                    node.left = None
                    node.right = None

                    return node
            else:
                node = self.check_distance_from_centroids(data, labels, node, node_app, original_centroids)

                return node
        else:
            node = self.check_distance_from_centroids(data, labels, node, node_app, original_centroids)

            return node

    def check_distance_from_centroids(self, data, labels, node, node_app, original_centroids):
        update_centroids = self.find_centroid(node_app)

        unique_labels_batch = np.unique(labels)
        unique_labels_node = np.unique(node.labels)
        unique_labels = np.unique(np.concatenate([unique_labels_batch, unique_labels_node]))

        for label in unique_labels:
            dist = np.linalg.norm(update_centroids[update_centroids['Label']==label] - original_centroids[original_centroids['Label']==label])
            if dist > self.dist_threshold or np.isnan(dist):
                all_data = pd.concat([data, labels], axis=1)
                all_data = all_data[all_data['Label'] == label]
                lab = all_data['Label']
                node.update_data(all_data.iloc[:, :-1], lab)

        return node

    def find_best_split(self, data, labels):
        n_thresholds = self.number_thresholds + 1
        results_features = {}

        print('--> Find best split')
        for feature in data.columns:
            if feature in self.ordinal_categories:
                thresholds = self.compute_thresholds(n_thresholds, data[feature], categorical=True)
            else:
                thresholds = self.compute_thresholds(n_thresholds, data[feature])

            for threshold in thresholds:
                left_data, right_data, left_labels, right_labels = self.apply_split(data, feature, threshold, labels)

                if len(left_data) < self.min_points_per_leaf or len(right_data) < self.min_points_per_leaf:
                    continue

                left_results = self.compute_metrics_splitting(left_data, left_labels)
                right_results = self.compute_metrics_splitting(right_data, right_labels)
                val_ret = [x + y for x, y in zip(left_results, right_results)]
                results_features[feature+"___"+str(threshold)] = val_ret

        if results_features == {}:
            return None

        best_split_key = max(results_features, key=lambda k: results_features[k][0])

        # Estrai feature e threshold dalla chiave (come in compute_pareto_optimality)
        value = best_split_key.split('___')
        return (value[0], float(value[1]))

        # best_split = self.compute_pareto_optimality(results_features)
        # return best_split

    def apply_split(self, data, feature, threshold, labels = None):
        if feature in self.ordinal_categories:
            left_mask = data[feature] == threshold
            right_mask = data[feature] != threshold
        else:
            left_mask = data[feature] <= threshold
            right_mask = data[feature] > threshold

        left_data = data[left_mask]
        right_data = data[right_mask]
        if labels is None:
            return left_data, right_data

        left_labels = labels[left_mask]
        right_labels = labels[right_mask]
        return left_data, right_data, left_labels, right_labels

    def compute_thresholds(self, n_thresholds, feature_values, categorical=False):
        thresholds = []
        if categorical:
            # n_thresholds = len(np.unique(feature_values))
            for x in np.unique(feature_values):
                thresholds.append(x)
            # print("For categorical features, the number of thresholds is set to the number of unique values.")
            return thresholds
        kmeans = KMeans(n_clusters=n_thresholds)
        kmeans.fit(feature_values.to_numpy().reshape(-1, 1))
        # thresholds = kmeans.cluster_centers_.reshape(-1)
        # Combine the values and labels, then sort by values
        combined = np.column_stack((list(feature_values), list(kmeans.labels_)))
        sorted_combined = combined[combined[:, 0].argsort()]
        # Separate the sorted values and labels again
        sorted_values = sorted_combined[:, 0]
        sorted_labels = sorted_combined[:, 1]
        # List to store thresholds between clusters
        thresholds = []

        # Iterate over the sorted list and find where the label changes
        for i in range(1, len(sorted_labels)):
            if sorted_labels[i] != sorted_labels[i - 1]:
                # Threshold is the midpoint between two adjacent clusters
                threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
                thresholds.append((sorted_labels[i - 1], sorted_labels[i], threshold))

        # unique_values = np.unique(feature_values)
        # thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        return [t[2] for t in thresholds]

    def compute_metrics_splitting(self, data, labels):
        # sil_score = util.compute_silhouette(data, labels)
        # entropy_score = -util.compute_entropy(labels)
        homogeneity_score = util.compute_homogeneity(labels)
        # return [sil_score, homogeneity_score]
        return [homogeneity_score]

    def compute_pareto_optimality(self, features_results):
        # Create a Pareto object

        datapoints = np.array(list(features_results.values()))
        pareto = oapackage.ParetoDoubleLong()

        # Adding values to the Pareto object
        for ii in range(datapoints.shape[0]):  # Iterate over the rows (points)
            w = oapackage.doubleVector(list(datapoints[ii]))  # Convert to list
            pareto.addvalue(w, ii)

        # Show the Pareto front
        # pareto.show(verbose=1)

        # Get indices of Pareto optimal designs
        lst = pareto.allindices()  # The indices of the Pareto optimal designs

        # Extracting the optimal data points
        optimal_datapoints = datapoints[lst, :]  # Use indices to extract optimal points
        # Step 1: Pick a random value from the array
        chosen_value = optimal_datapoints[random.choice(range(optimal_datapoints.shape[0]))]

        for key in features_results:
            if features_results[key] == list(chosen_value):
                value = key.split('___')
                return (value[0], float(value[1]))

    # Calculate information gain from a split
    def entropy_gain(self, node_parent, node_left, node_right):
        # y is the parent node, y_left and y_right are the child nodes
        parent_entropy = node_parent.metrics["entropy"]
        n = len(node_parent.data)

        # Calculate weighted entropy for child nodes
        weighted_entropy = (len(node_left.data) / n) * node_left.metrics["entropy"] + (len(node_right.data) / n) * node_right.metrics["entropy"]

        # Information gain is the reduction in entropy
        gain = parent_entropy - weighted_entropy
        return gain

    def homogeneity_gain(self, node_parent, node_left, node_right):
        # y is the parent node, y_left and y_right are the chiGainld nodes
        parent_homogeneity = node_parent.metrics["homogeneity"]
        n = len(node_parent.data)

        # Calculate weighted homogeneity for child nodes
        weighted_homogeneity = (len(node_left.data) / n) * node_left.metrics["homogeneity"] + (len(node_right.data) / n) * node_right.metrics["homogeneity"]

        # Information gain is the reduction in homogeneity
        gain = weighted_homogeneity - parent_homogeneity
        return gain

    # Function to calculate the silhouette gain
    def silhouette_gain(self, node_parent, node_left, node_right):
        # Compute the silhouette score for the parent cluster
        parent_silhouette = node_parent.metrics["silhouette"]

        # Calculate the number of points in the dataset
        n = len(node_parent.data)

        # Calculate weighted entropy for child nodes
        weighted_child_silhouette = (len(node_left.data) / n) * node_left.metrics["silhouette"] + (len(node_right.data) / n) * \
                           node_right.metrics["silhouette"]

        # Calculate Silhouette Gain
        gain = weighted_child_silhouette - parent_silhouette

        return gain

    def evaluate_split(self, node_parent, node_left, node_right):
        # entropy_gain = self.entropy_gain(node_parent, node_left, node_right)
        homogeneity_gain = self.homogeneity_gain(node_parent, node_left, node_right)
        # silhouette_gain = self.silhouette_gain(node_parent, node_left, node_right)
        if homogeneity_gain > self.homogeneity_gain_threshold:
            return True
        else:
            return False


    def detect_anomalies_with_mad(self, node, data, label_test, T_mad=3.5):
        if node.is_leaf():
            centroids = self.find_centroid(node)
            median_d_i, MAD = self.compute_mad(node, centroids)

            data_np = data.to_numpy()
            centroids_np = centroids.iloc[:, 1:].values
            dists = np.linalg.norm(data_np[:, np.newaxis, :] - centroids_np[np.newaxis, :, :], axis=2)
            A_inst = np.abs(dists - median_d_i[np.newaxis, :]) / MAD[np.newaxis, :]

            min_A_inst = np.min(A_inst, axis=1)

            mask_drift = min_A_inst > T_mad
            num_drift = (min_A_inst > T_mad).sum()

            pred = centroids.iloc[np.argmin(dists, axis=1), 0].reset_index(drop=True)
            pred.index = label_test.index

            # self.plots_clusters(np.concatenate([node.data, data], axis=0), np.concatenate([node.labels, pred], axis=0), len(pred))

            results = pd.DataFrame({'Label': label_test, 'Predicted': pred})

            if num_drift > 0:
                if len(centroids) == 1:
                    pred[mask_drift] = pred[mask_drift].map(lambda x: 1 if x == 0 else 0 if x == 1 else x)

                _ = self.build_tree(node, data, pred, depth=node.depth + 1)

            return results
        else:
            left_data, right_data, left_labels, right_labels = self.apply_split(data, node.split_feature, node.split_threshold, labels=label_test)

            if len(left_data) != 0:
                left_data = self.detect_anomalies_with_mad(node.left, left_data, left_labels)
            if len(right_data) != 0:
                right_data = self.detect_anomalies_with_mad(node.right, right_data, right_labels)

            return pd.concat([left_data, right_data], axis=0)

    def detect_anomalies_with_mean(self, node, data, label_test, T_mad=3.5):
        if node.is_leaf():
            centroids = self.find_centroid(node)

            data_np = data.to_numpy()
            centroids_np = centroids.iloc[:, 1:].values
            dists = np.linalg.norm(data_np[:, np.newaxis, :] - centroids_np[np.newaxis, :, :], axis=2)

            data_node = node.data.to_numpy()
            distances = {}
            for label in [0, 1]:
                cluster_points = data_node[node.labels == label]
                if len(cluster_points) == 0:
                    distances[label] = np.nan
                    continue
                centroid = centroids[centroids['Label']==label].values
                mean_dists = np.linalg.norm(cluster_points - centroid[:, 1:], axis=1)
                distances[label] = np.mean(mean_dists)

            scores = dists / np.array([distances[c] for c in distances])
            centroids = centroids.sort_values(by=centroids.columns[0])
            pred = centroids.iloc[np.nanargmin(scores, axis=1), 0].reset_index(drop=True)
            scores = np.concatenate((scores, pred.to_numpy().reshape(-1, 1)), axis=1)
            pred.index = label_test.index

            selected_item = np.where(scores[:, -1] == 0, scores[:, 0], scores[:, 1])
            mask = selected_item > 2.0
            pred[mask] = 1

            # self.plots_clusters(np.concatenate([node.data, data], axis=0), np.concatenate([node.labels, pred], axis=0), len(pred))

            results = pd.DataFrame({'Label': label_test, 'Predicted': pred})

            # if num_drift > 0:
            #     if len(centroids) == 1:
            #         pred[mask_drift] = pred[mask_drift].map(lambda x: 1 if x == 0 else 0 if x == 1 else x)
            #
            #     _ = self.build_tree(node, data, pred, depth=node.depth + 1)

            return results
        else:
            left_data, right_data, left_labels, right_labels = self.apply_split(data, node.split_feature, node.split_threshold, labels=label_test)

            if len(left_data) != 0:
                left_data = self.detect_anomalies_with_mean(node.left, left_data, left_labels)
            if len(right_data) != 0:
                right_data = self.detect_anomalies_with_mean(node.right, right_data, right_labels)

            return pd.concat([left_data, right_data], axis=0)

    def detect_anomalies_with_adaptive_clusters(self, node, data, label_test):
        if node.is_leaf():

            list_kmeans = []
            distorsions = []

            for k in range(1, 7):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(node.data)
                list_kmeans.append(kmeans)
                distorsions.append(kmeans.inertia_)

            # Compute the first derivative of the distortion
            first_derivative = np.diff(distorsions)
            optimal_k = np.argmin(first_derivative) + 2
            # Find the best Kmeans
            try:
                kmeans = list_kmeans[optimal_k]
            except IndexError:
                kmeans = list_kmeans[-1]

            kmeans_labels = kmeans.labels_
            centroids = kmeans.cluster_centers_
            labels = pd.DataFrame({'Kmeans': kmeans_labels, 'Label': node.labels})
            cluster_labels = labels.groupby('Kmeans')['Label'].agg(lambda x: x.mode()[0])

            distances = {}
            for label in cluster_labels.index:
                cluster_points = node.data.reset_index(drop=True)[labels['Kmeans'].reset_index(drop=True) == label]
                if len(cluster_points) == 0:
                    distances[label] = np.nan
                    continue
                mean_dists = np.linalg.norm(cluster_points - centroids[label, :], axis=1)
                distances[label] = np.mean(mean_dists)

            data_np = data.to_numpy()
            dists = np.linalg.norm(data_np[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

            scores = dists / np.array([distances[c] for c in distances])
            pred_cluster = np.nanargmin(scores, axis=1)
            pred = cluster_labels[pred_cluster].reset_index(drop=True)

            # self.plots_clusters(np.concatenate([node.data, data], axis=0), np.concatenate([node.labels, pred], axis=0), len(pred))

            results = pd.DataFrame({'Label': label_test.reset_index(drop=True), 'Predicted': pred})

            results = results.reset_index(drop=True)
            data = data.reset_index(drop=True)
            results = pd.concat([results, data], axis=1)

            return results
        else:
            left_data, right_data, left_labels, right_labels = self.apply_split(data, node.split_feature, node.split_threshold, labels=label_test)

            left_res = pd.DataFrame(columns=['Label', 'Predicted'])
            rigth_res = pd.DataFrame(columns=['Label', 'Predicted'])

            if len(left_data) != 0:
                left_res = self.detect_anomalies_with_adaptive_clusters(node.left, left_data, left_labels)
            if len(right_data) != 0:
                rigth_res = self.detect_anomalies_with_adaptive_clusters(node.right, right_data, right_labels)

            return pd.concat([left_res, rigth_res], axis=0)


    def compute_mad(self, node, centroids, b=1.4826):
        labels = node.labels
        centroid_map = {row[0]: row[1:] for row in centroids.to_numpy()} #Mappa tra label e centroide
        dists = np.array([np.linalg.norm(node.data.values[i] - centroid_map[labels[i]]) for i in range(len(labels))])
        median_d_i = pd.Series(dists).groupby(labels).median()
        MAD = b * pd.Series(np.abs(dists - median_d_i[labels].to_numpy())).groupby(labels).median()
        return median_d_i.to_numpy(), MAD.to_numpy()


    def plots_clusters(self, df, y, n_test):
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(df)
        y[-n_test:] = np.where(y[-n_test:] == 0, 3, 4)

        plt.figure(figsize=(8, 6))

        # Classe 0
        plt.scatter(
            X_2d[y == 0, 0], X_2d[y == 0, 1],
            c='dodgerblue', label='Benign', alpha=0.7, edgecolors='k'
        )

        # Classe 1
        plt.scatter(
            X_2d[y == 1, 0], X_2d[y == 1, 1],
            c='crimson', label='Attack - Node', alpha=0.7, edgecolors='k'
        )

        # Classe test
        plt.scatter(
            X_2d[y == 3, 0], X_2d[y == 3, 1],
            c='green', label='test benign', alpha=0.7, edgecolors='k',
        )

        plt.scatter(
            X_2d[y == 4, 0], X_2d[y == 4, 1],
            c='orange', label='Attack', alpha=0.7, edgecolors='k',
        )

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def find_centroid(self, node):
        data_copy = node.data.copy()
        data_copy['Label'] = node.labels
        # Group by the 'Label' column and calculate the mean for each group (centroid)
        data_copy = node.data.copy()
        data_copy['Label'] = node.labels

        # Group by the 'Label' column and calculate the mean for each group (centroid)
        centroids = data_copy.groupby('Label').mean().reset_index()
        return centroids
