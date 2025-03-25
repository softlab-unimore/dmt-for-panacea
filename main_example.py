import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from UnsupervisedDMT.decision_tree import DecisionTree
from copy import deepcopy

warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)


if __name__ == "__main__":
    # Generazione di dati di esempio

    # Generate dataset
    n_samples = 1000  # Number of data points
    n_features = 4  # Number of features (dimensions)
    centers = 4  # Number of clusters (blobs)
    cluster_std = 1.0  # Standard deviation of the clusters

    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, cluster_std=cluster_std, random_state=42)
    # Convert X into a pandas DataFrame with column names feature1, feature2, etc.
    column_names = [f'feature{i + 1}' for i in range(n_features)]
    data = pd.DataFrame(data, columns=column_names)
    np.random.seed(42)
    # data = pd.DataFrame(np.random.rand(10,4), columns=['feature1', 'feature2', 'feature3', 'feature4'])
    # labels = np.random.randint(0, 2, 10)  # Due classi


    # TODO: TOGLIERE I DATI SUI NODI INTERNI, CHE STANNO GIA SULLE FOGLIE

    # Inizializzare e addestrare l'albero decisionale
    tree = DecisionTree(max_depth=10, min_points_per_leaf=5, closest_k_points=0.8, closer_DBSCAN_point=0.1, eps_DBSCAN=0.9)
    data1, labels1 = data[:100], labels[:100]
    data2, labels2 = data[100:200], labels[100:200]
    data3, labels3 = data[200:300], labels[200:300]
    data4, labels4 = data[300:400], labels[300:400]
    data5, labels5 = data[400:500], labels[400:500]
    data6, labels6 = data[500:600], labels[500:600]
    data7, labels7 = data[600:700], labels[600:700]
    data8, labels8 = data[700:800], labels[700:800]
    data9, labels9 = data[800:900], labels[800:900]
    data10, labels10 = data[900:1000], labels[900:1000]
    data10.iloc[20:25] += 50
    data10.iloc[70:80] += 1000
    data10.iloc[10:11] += 60

    # data10.iloc[60:80] += 2

    root = tree.partial_fit(None, data1, labels1)

    tree.partial_fit(root, data2, labels2)

    tree.partial_fit(root, data3, labels3)

    tree.partial_fit(root, data4, labels4)

    tree.partial_fit(root, data5, labels5)

    tree.partial_fit(root, data6, labels6)

    tree.partial_fit(root, data7, labels7)

    tree.partial_fit(root, data8, labels8)

    tree.partial_fit(root, data9, labels9)

    get_anomalies = tree.detect_anomalies(root, data10)


    df = deepcopy(get_anomalies)
    df = df.sort_index()

    # Create the figure and first axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Feature1 and Feature2 on the first y-axis
    ax1.plot(df.index, df["feature1"], label="Feature1", color="blue")
    ax1.plot(df.index, df["feature2"], label="Feature2", color="green")
    ax1.plot(df.index, df["feature3"], label="Feature3", color="orange")
    ax1.plot(df.index, df["feature4"], label="Feature4", color="purple")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Feature1, Feature2, Feature3, Feature4", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.legend(loc="upper left")

    # Create the second y-axis and plot Feature3 and Feature4
    ax2 = ax1.twinx()
    ax2.scatter(df.index, df["lof"], label="Feature3", color="red")  # Use scatter instead of plot
    ax2.set_ylabel("lof", color="red")
    ax2.tick_params(axis='y', labelcolor="red")
    ax2.legend(loc="upper right")

    # Show plot
    plt.title("Multivariate Time Series with Dual Y-Axes")
    plt.show()

    # print(get_anomalies)
    # Stampa dell'albero
    # print(tree)

    # Qui si può applicare l'ottimizzazione incrementale se desiderato
