import os
import pandas as pd
from tpot import TPOTClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
import joblib
import warnings

# Configuration
scoring_metric = "poac_sv7"
validation_folder = "datasets/validation_csv"
results_file = f"results/{scoring_metric}/experiment_summary.csv"
output_folder = f"results/{scoring_metric}"
labels_folder = f"results/{scoring_metric}/labels"
population_size = 50
verbosity = 2
random_state = 42

# Ensure directories exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

class SurrogateScorer:
    def __init__(self, model, meta_features, cvi=[silhouette_score, davies_bouldin_score, calinski_harabasz_score]) -> None:
        self.model = model
        self.meta_features = meta_features
        self.cvi = cvi

    def __call__(self, estimator, X):
        try:
            warnings.filterwarnings('ignore')
            cluster_labels = estimator.fit_predict(X)
            mf = self.meta_features.copy()
            mf.extend([score(X,cluster_labels) for score in self.cvi])
            surrogate_score = self.model.predict([mf])[0]
            return surrogate_score if len(set(cluster_labels)) > 1 else -float('inf') 
        except Exception as e:
            raise TypeError(f"{e}")


def plot_pca_comparison(X, y, labels, ari_score, save_path):
    """Generate and save PCA plots comparing original labels and cluster labels."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
    axes[0].set_title('PCA with Original Labels')
    axes[0].legend(*scatter1.legend_elements(), title="Classes")

    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', edgecolor='k')
    axes[1].set_title(f'PCA with Cluster Labels\nARI: {ari_score:.2f}')
    axes[1].legend(*scatter2.legend_elements(), title="Clusters")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def get_processed_datasets(results_file):
    """Load processed datasets from results file."""
    if os.path.exists(results_file):
        df = pd.read_csv(results_file)
        return set(df['Dataset'])
    return set()

# Initialize the results file if it doesn't exist
if not os.path.exists(results_file):
    pd.DataFrame(columns=["Dataset", "Best_Pipeline", "silhouette_score", 
                          "davies_bouldin_score", "calinski_harabasz_score", 
                          "adjusted_rand_score", "Running_Time(s)"]
                ).to_csv(results_file, index=False)

# Get the list of already processed datasets
processed_datasets = get_processed_datasets(results_file)
model = joblib.load("poac/models/poac_sv7.joblib")

# Iterate over datasets
for dataset_name in os.listdir(validation_folder):
    if not dataset_name.endswith(".csv") or dataset_name in processed_datasets:
        print(f"Skipping {dataset_name}, already processed.")
        continue

    try:
        print(f"\nProcessing dataset: {dataset_name}")
        start_time = time.time()

        # Load and preprocess data
        df = pd.read_csv(os.path.join(validation_folder, dataset_name))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Convert categorical labels to numeric if necessary
        if y.dtype == 'object' or y.dtype.name == 'category':
            y = y.astype('category').cat.codes.to_numpy()
        else:
            y = y.to_numpy()

        # Scale numeric features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X.select_dtypes(include=['number']))

        # TPOT clustering optimization
        tpot_clustering = TPOTClustering(
            population_size=population_size,
            verbosity=verbosity,
            random_state=random_state,
            crossover_rate=0.05,
            mutation_rate=0.9,
            max_time_mins=30,
            max_eval_time_mins=1.0,
            scoring="calinski_harabasz_score"
        )

        tpot_clustering.fit(X_scaled)
        labels = tpot_clustering.fitted_pipeline_[-1].labels_

        # Save labels to CSV for comparison
        labels_df = pd.DataFrame(labels, columns=["Cluster_Label"])
        labels_df.to_csv(f"{labels_folder}/{dataset_name.replace('.csv', '_labels.csv')}", index=False)

        # Calculate metrics
        sil = silhouette_score(X_scaled, labels)
        dbs = davies_bouldin_score(X_scaled, labels)
        chs = calinski_harabasz_score(X_scaled, labels)
        ari = adjusted_rand_score(y, labels)
        running_time = round(time.time() - start_time, 2)

        # Save results
        results = pd.DataFrame({
            "Dataset": [dataset_name],
            "Best_Pipeline": [tpot_clustering.fitted_pipeline_.named_steps],
            "silhouette_score": [sil],
            "davies_bouldin_score": [dbs],
            "calinski_harabasz_score": [chs],
            "adjusted_rand_score": [ari],
            "Running_Time(s)": [running_time]
        })

        results.to_csv(results_file, mode="a", header=False, index=False)

        # Save PCA plot
        plot_pca_comparison(X_scaled, y, labels, ari, f"{output_folder}/{dataset_name.replace('.csv', '_pca.png')}")
        print(f"Completed {dataset_name} with ARI: {ari}")

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")
        error_results = pd.DataFrame({
            "Dataset": [dataset_name],
            "Best_Pipeline": ["ERROR"],
            "silhouette_score": ["ERROR"],
            "davies_bouldin_score": ["ERROR"],
            "calinski_harabasz_score": ["ERROR"],
            "adjusted_rand_score": ["ERROR"],
            "Running_Time(s)": [0]
        })
        error_results.to_csv(results_file, mode="a", header=False, index=False)
