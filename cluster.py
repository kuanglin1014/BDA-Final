import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode

def best_map(base_labels, pred_labels):
    unique_base = np.unique(base_labels)
    unique_pred = np.unique(pred_labels)
    cost_matrix = np.zeros((len(unique_base), len(unique_pred)), dtype=int)
    for i, b in enumerate(unique_base):
        for j, p in enumerate(unique_pred):
            cost_matrix[i, j] = -np.sum((base_labels == b) & (pred_labels == p))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    new_labels = np.zeros_like(pred_labels)
    for r, c in zip(row_ind, col_ind):
        new_labels[pred_labels == unique_pred[c]] = unique_base[r]
    return new_labels

def ensemble_kmeans(X, n_clusters, n_runs=21, base_seed=42):
    all_labels = np.zeros((X.shape[0], n_runs), dtype=int)
    kmeans = KMeans(n_clusters=n_clusters, random_state=base_seed, n_init=15)
    base_labels = kmeans.fit_predict(X)
    all_labels[:, 0] = base_labels

    for i in range(1, n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=base_seed + i, n_init=15)
        labels = kmeans.fit_predict(X)
        aligned_labels = best_map(base_labels, labels)
        all_labels[:, i] = aligned_labels

    final_labels, _ = mode(all_labels, axis=1)
    return final_labels.ravel()

def main():
    public_df = pd.read_csv("public_data.csv")
    private_df = pd.read_csv("private_data.csv")

    X_public = public_df.drop(columns=['id']).values
    X_private = private_df.drop(columns=['id']).values

    scaler_public = StandardScaler()
    X_public_scaled = scaler_public.fit_transform(X_public)

    scaler_private = StandardScaler()
    X_private_scaled = scaler_private.fit_transform(X_private)

    n_clusters_public = 15
    n_clusters_private = 23

    public_labels = ensemble_kmeans(X_public_scaled, n_clusters_public, n_runs=21)
    private_labels = ensemble_kmeans(X_private_scaled, n_clusters_private, n_runs=21)

    pd.DataFrame({'id': public_df['id'], 'label': public_labels}).to_csv("public_submission.csv", index=False)
    pd.DataFrame({'id': private_df['id'], 'label': private_labels}).to_csv("private_submission.csv", index=False)

    print("Ensemble K-Means (21 runs) clustering complete.")
    print("- public_submission.csv")
    print("- private_submission.csv")

if __name__ == "__main__":
    main()
