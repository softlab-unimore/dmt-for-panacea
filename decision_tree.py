from statistics import median

import oapackage
import random

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from tqdm import tqdm

from TreeNode import TreeNode
import numpy as np
from copy import deepcopy
from sklearn.cluster import KMeans
import util
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
# TODO: RISOLVERE QUANDO NON CI SONO DATI DA METTERE NELLE FOGLIE!!!!
# TODO: CONTROLLARE PERCHé AUMENTANO DI 2 ALLA VOLTA LA PROFONDITA
# Per risolverlo bisognerebbe assegnare un minimo di quante foglie devono esserci ogni volta che splitto i dati
# Se per esempio non ci sono abbastanza dati, allora non posso farne il split

# Classe Albero Decisionale
class DecisionTree:
    def __init__(self, max_depth=5, min_points_per_leaf=20, closest_k_points=0.5, number_thresholds=3, closer_DBSCAN_point=0.1, eps_DBSCAN=0.9, ordinal_categories=[]):
        self.max_depth = max_depth
        self.min_points_per_leaf = min_points_per_leaf
        self.closest_k_points = closest_k_points
        self.number_thresholds = number_thresholds
        self.closer_DBSCAN_point = closer_DBSCAN_point
        self.eps_DBSCAN = eps_DBSCAN
        self.ordinal_categories = ordinal_categories

    def partial_fit(self, node, data, labels=None):
        if node is None:
            node = TreeNode(data, labels, depth=0)
        elif not node.is_leaf():
            data_left, data_right, labels_left, labels_right = self.apply_split(data, node.split_feature, node.split_threshold, labels)

            node.left = self.partial_fit(node.left, data_left, labels_left)
            node.right = self.partial_fit(node.right, data_right, labels_right)

        else:
            node = self.build_tree(node, data, labels, depth=node.depth + 1)
            # TODO: Compute Metrics
            # TODO: Check Optimality
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

        # node = TreeNode(data, labels, depth=depth, max_depth=self.max_depth)
        # Controllare self min points


        if depth >= self.max_depth: # or len(data) <= self.min_points_per_leaf:
            return node.update_data(data, labels)

        # TODO: Find best split
        node_app = deepcopy(node)
        node_app.update_data(data, labels)

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
                return node
            else:
                return node_app
        else:
            return node_app

    def find_best_split(self, data, labels):
        n_thresholds = self.number_thresholds + 1
        results_features = {}
        # TODO: Implementare l'algoritmo di ricerca del miglior split
        for feature in data.columns:
            # TODO: Implementare l'algoritmo di splitting
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
        # Tupla
        best_split = self.compute_pareto_optimality(results_features)
        return best_split

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
        sil_score = util.compute_silhouette(data, labels)
        # entropy_score = -util.compute_entropy(labels)
        homogeneity_score = util.compute_homogeneity(labels)
        return [sil_score, homogeneity_score]

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
        # y is the parent node, y_left and y_right are the child nodes
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

    # TODO: AL MOMENTO ANCHE SE C'è UN PICCOLO GAIN SPLITTA, DOVREMMO GESTIRE MEGLIO QUESTO
    def evaluate_split(self, node_parent, node_left, node_right):
        # entropy_gain = self.entropy_gain(node_parent, node_left, node_right)
        homogeneity_gain = self.homogeneity_gain(node_parent, node_left, node_right)
        silhouette_gain = self.silhouette_gain(node_parent, node_left, node_right)
        if homogeneity_gain > 0 and silhouette_gain > 0:
            return True
        else:
            return False


    def detect_anomalies(self, node, data):
        if node.is_leaf():
            # self.visualize_plot(node, data)
            centroids = self.find_centroid(node)
            closest_class = self.find_closer_class(node, centroids, data)
            # perc_closes = self.compute_closest_class_perc(node, closest_class, data)
            perc_overlapping = self.compute_DBSCAN(node, closest_class, data)
            # TODO: Check if anomaly
            data['lof'] = perc_overlapping['lof']
            # print("Anomaly Value: ", perc_overlapping)
            return data
            # TODO: Check if anomaly
        else:
            left_data, right_data = self.apply_split(data, node.split_feature, node.split_threshold)
            left_data = self.detect_anomalies(node.left, left_data)
            right_data = self.detect_anomalies(node.right, right_data)
            return pd.concat([left_data, right_data], axis=0)


    def detect_anomalies_with_mad(self, node, data, label_test, T_mad=3.5):
        if node.is_leaf():
            centroids = self.find_centroid(node)
            median_d_i, MAD = self.compute_mad(node, centroids)

            data = data.to_numpy()
            dists = np.linalg.norm(data[:, np.newaxis] - centroids.iloc[:, 1:].values, axis=2)
            A_inst = (dists - median_d_i.T) / MAD.T

            min_A_inst = np.min(A_inst, axis=1)

            num_drift = (min_A_inst > T_mad).sum()
            print(num_drift, ' Drift detected')

            pred = centroids.iloc[np.argmin(dists, axis=1), 0].reset_index(drop=True)
            pred.index = label_test.index

            results = pd.DataFrame({'Label': label_test, 'Predicted': pred})
            return results
        else:
            left_data, right_data, left_labels, right_labels = self.apply_split(data, node.split_feature, node.split_threshold, labels=label_test)
            left_data = self.detect_anomalies_with_mad(node.left, left_data, left_labels)
            right_data = self.detect_anomalies_with_mad(node.right, right_data, right_labels)
            return pd.concat([left_data, right_data], axis=0)


    def compute_mad(self, node, centroids, b=1.4826):
        labels = node.labels
        centroids_values = centroids.iloc[:, 1:].values
        dists = np.linalg.norm(node.data.values[:, np.newaxis] - centroids_values, axis=2)
        median_d_i = pd.DataFrame(dists).groupby(labels).median()
        MAD = (b * pd.DataFrame(abs(pd.DataFrame(dists) - pd.DataFrame(dists).groupby(labels).median())).groupby(labels).median())
        return median_d_i.to_numpy(), MAD.to_numpy()

    #TODO: Controllare perché il primo valore del dataframe ritorna 2 volte
    #TODO: Fare in modo che quando arrivano nella funzione anomaly score al posto di perc ci sia il valore di anomalia calcoalto da LOF

    def compute_DBSCAN(self, node,closest_class, data):
        class_dfs = []
        data_node = node.data.copy(deep=True)
        data_node['Label'] = node.labels
        data_node = data_node[data_node['Label'] == closest_class]
        data_label = data
        data_label['Label'] = closest_class
        df_combined = pd.concat([data_node, data_label], axis=0)
        X = df_combined.drop(columns=['Label']).values
        # Normalizza X
        # Ad un certo punto da errore sul MinMaxScaler: Input X contains infinity or a value too large for dtype('float64').
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        # Parametri di DBSCAN
        min_samples = int(len(df_combined)*self.closer_DBSCAN_point)  # numero minimo di punti per formare un cluster
        # Applica DBSCAN
        if min_samples == 0:
            min_samples = 2
        dbscan = DBSCAN(eps=self.eps_DBSCAN, min_samples=min_samples)
        dbscan.fit(X)
        # Etichette di cluster generate da DBSCAN
        labels = dbscan.labels_
        if np.all(labels == -1):
            print('Troppo Rumoroso')
            labels = np.zeros_like(labels)
        # Aggiungi le etichette di cluster al DataFrame combinato
        df_combined['cluster'] = labels
        # Visualizza i risultati
        combine_new = df_combined[len(data_node):]
        if not np.all(combine_new['cluster'].values == 0):
            print('Rumoroso')
            different_cluster = combine_new[combine_new['cluster'] != 0]
            combine_new = combine_new[combine_new['cluster'] == 0]
            combine_new.drop(['cluster','Label'], axis=1, inplace=True)
            # Create a dictionary to hold DataFrames for each class in 'different_cluster'
            class_dfs = [different_cluster[different_cluster['cluster'] == cls] for cls in different_cluster['cluster'].unique()]
            for df_different in class_dfs:
                df_different.drop('cluster', axis=1, inplace=True)
                centroide = df_different.groupby('Label').mean().reset_index()
                app_data = pd.concat([data_node, centroide], axis=0)
                app_data.drop('Label', axis=1, inplace=True)
                lof = LocalOutlierFactor(n_neighbors=min_samples)
                lof.fit_predict(app_data)
                df_different['lof'] = -lof.negative_outlier_factor_[-1] # dovrebbe essere quanto è effettivamente un anomalia, quindi quanto si discosta dal  
            # Iterate over each DataFrame in the dictionary
        if class_dfs != []:
            if combine_new.shape[0] > 0:
                results = self.detect_anomalies(node, combine_new)
                if type(results) == list and len(results) > 0:
                    print('CIAO')
                class_dfs.append(results)
            # class_dfs = pd.concat(class_dfs, axis=0) if class_dfs else class_dfs
            class_dfs = pd.concat(class_dfs, axis=0) if class_dfs else class_dfs[0]
            # class_dfs = pd.concat([class_dfs, df_different], axis=0)

            # class_dfs = pd.concat([class_dfs, df_different], axis=0)
            # df_combined = pd.concat(class_dfs, axis=0)
        else:
            combine_new['lof'] = 0
            class_dfs = combine_new
        return class_dfs


        '''num_negative_ones = (combine_new['cluster'] == -1).sum()
        num_negative_ones = num_negative_ones + (combine_new['cluster'] == 1).sum()
        total_count = len(combine_new['cluster'])
        ratio = num_negative_ones / total_count'''


    def compute_closest_class_perc(self, node, closest_class, data):
        # Extract the data (features) and labels from the node
        data_val = node.data  # DataFrame of N features
        labels_val = node.labels  # Labels associated with the points in the DataFrame

        # Convert the new point to a NumPy array if it's not already
        new_point = np.array(data.mean())

        # Calculate the Euclidean distances between the new point and all points in the data
        distances = np.linalg.norm(data_val.values - new_point, axis=1)

        k_points = int(len(distances) * self.closest_k_points)

        # Get the indices of the k smallest distances
        k_nearest_indices = np.argsort(distances)[:k_points]

        # Retrieve the corresponding labels for the k nearest points
        nearest_labels = labels_val[k_nearest_indices]

        # Determine the most common label among the nearest labels
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        # Count occurrences of each label in nearest_labels
        label_counts = {label: count for label, count in zip(unique_labels, counts)}

        return label_counts[closest_class] / k_points


    def find_centroid(self, node):
        data_copy = node.data.copy()
        data_copy['Label'] = node.labels
        # Group by the 'Label' column and calculate the mean for each group (centroid)
        data_copy = node.data.copy()
        data_copy['Label'] = node.labels

        # Group by the 'Label' column and calculate the mean for each group (centroid)
        centroids = data_copy.groupby('Label').mean().reset_index()
        return centroids

    def find_closer_class(self, node, centroids, data):
        # Separate the 'Label' column and the centroid data (features)
        labels = centroids['Label']
        centroid_features = centroids.drop(columns=['Label'])

        # Convert new_centroid to a NumPy array if it's a list or Pandas Series
        new_centroid = np.array(data.mean())

        # Compute the Euclidean distance between the new centroid and each existing centroid
        distances = np.linalg.norm(centroid_features.values - new_centroid, axis=1)

        # Find the index of the closest centroid
        closest_index = np.argmin(distances)

        # Return the label of the closest centroid
        return labels.iloc[closest_index]

    def visualize_plot(self, node, data):
        import pandas as pd

        # Assuming `result.data` is the DataFrame of labeled data from TreeNode
        labeled_data = deepcopy(node.data)
        data_copy = deepcopy(data)
        # Create a copy of new_data to keep track of which data is labeled vs. unlabeled
        data_copy['Label'] = 'Unlabeled'
        labeled_data['Label'] = node.labels

        # Combine both labeled and unlabeled data for visualization
        combined_data = pd.concat([labeled_data, data_copy], ignore_index=True)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # Separate features from labels (remove 'Label' column temporarily)
        features = combined_data.drop(columns=['Label'])

        # Apply t-SNE to reduce data to 2 dimensions
        tsne = TSNE(n_components=2, random_state=42, perplexity=1)
        reduced_features = tsne.fit_transform(features)

        # Add the t-SNE results back into the DataFrame
        combined_data['TSNE1'] = reduced_features[:, 0]
        combined_data['TSNE2'] = reduced_features[:, 1]

        import seaborn as sns

        # Plot using Seaborn, coloring by 'Label'
        plt.figure(figsize=(10, 7))
        sns.scatterplot(x='TSNE1', y='TSNE2', hue='Label', data=combined_data, palette='Set1')

        plt.title('t-SNE Visualization of Labeled and Unlabeled Data')
        plt.show()



