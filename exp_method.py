import scipy.stats.mstats as statm
import pandas as pd
import numpy as np
import time

from sklearn.covariance import GraphicalLasso, LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import precision_score, recall_score, f1_score, mean_squared_error
from datetime import datetime
from itertools import combinations
from typing import List, Tuple


class SSA:
    def __init__(self, time_series: np.ndarray, L: int):
        """
        Parameters:
        -----------
        time_series: numpy.ndarray
            Input time series matrix of shape (n_series, series_length)
        L: int
            Window length (embedded dimension)
        """
        self.time_series = time_series
        self.n_series = time_series.shape[0]
        self.series_length = time_series.shape[1]
        self.L = L
        self.K = self.series_length - self.L + 1

        # Validate inputs
        if not 2 <= L <= self.series_length // 2:
            raise ValueError(f"Window length must be in [2, {self.series_length // 2}]")

        # Initialize results
        self.X = None
        self.U = None
        self.sigma = None
        self.VT = None
        self.elementary_matrices = None

    def embed(self, normalized: bool = True) -> np.ndarray:
        """Create trajectory matrix from time series"""
        if normalized:
            ts = (self.time_series - self.time_series.mean(axis=1, keepdims=True)) / \
                 self.time_series.std(axis=1, keepdims=True)
        else:
            ts = self.time_series

        X_list = []
        for series in ts:
            X = np.column_stack([series[i:i + self.L] for i in range(self.K)])
            X_list.append(X)

        self.X = np.array(X_list)
        return self.X

    def decompose(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform SVD on trajectory matrix"""
        if self.X is None:
            self.embed()

        U_list, sigma_list, VT_list = [], [], []

        for X in self.X:
            U, sigma, VT = np.linalg.svd(X)
            U_list.append(U)
            sigma_list.append(sigma)
            VT_list.append(VT)

        self.U = np.array(U_list)
        self.sigma = np.array(sigma_list)
        self.VT = np.array(VT_list)

        return self.U, self.sigma, self.VT

    def get_elementary_matrices(self) -> np.ndarray:
        """Compute elementary matrices from SVD components"""
        if any(x is None for x in [self.U, self.sigma, self.VT]):
            self.decompose()

        elementary = []
        for i in range(self.n_series):
            U_i = self.U[i]
            sigma_i = self.sigma[i]
            VT_i = self.VT[i]

            elementary_i = []
            for j in range(len(sigma_i)):
                E = sigma_i[j] * np.outer(U_i[:, j], VT_i[j, :])
                elementary_i.append(E)
            elementary.append(elementary_i)

        self.elementary_matrices = np.array(elementary)
        return self.elementary_matrices

    def diagonal_averaging(self, matrix: np.ndarray) -> np.ndarray:
        """
        Perform diagonal averaging on a matrix to convert it back to time series
        """
        L, K = matrix.shape
        N = L + K - 1
        series = np.zeros(N)

        # Diagonal averaging
        for k in range(N):
            if k < L:
                # Upper part
                start = 0
                end = k + 1
            elif k < K:
                # Middle part
                start = 0
                end = L
            else:
                # Lower part
                start = k - K + 1
                end = L

            # Calculate mean of elements on k-th anti-diagonal
            diag_sum = 0
            count = 0
            for i in range(start, end):
                j = k - i
                if j < K:
                    diag_sum += matrix[i, j]
                    count += 1
            series[k] = diag_sum / count

        return series

    def reconstruct(self, indices: List[int]) -> np.ndarray:
        """Reconstruct time series from selected elementary matrices"""
        if self.elementary_matrices is None:
            self.get_elementary_matrices()

        reconstructed = []
        for series_idx in range(self.n_series):
            # Sum selected elementary matrices
            X_reconstructed = np.sum([self.elementary_matrices[series_idx][i]
                                      for i in indices], axis=0)

            # Diagonal averaging
            ts_reconstructed = self.diagonal_averaging(X_reconstructed)
            reconstructed.append(ts_reconstructed)

        return np.array(reconstructed)

    def compute_linear_dependence(self, group_indices: List[int]) -> float:
        """
        Compute linear dependence for reconstructed components in a group
        """
        # Get elementary matrices for group
        group_matrices = self.elementary_matrices[group_indices]

        # Compute correlation matrix between group components
        corr_mat = np.corrcoef([self.diagonal_averaging(X[0]) for X in group_matrices])

        # Get eigenvalues
        eigenvals = np.linalg.eigvals(corr_mat)

        # Linear dependence is 1 - smallest eigenvalue
        linear_dep = 1 - np.min(np.real(eigenvals))

        return linear_dep

    def select_top_components(self, series_idx: int, num_components: int) -> Tuple[List[int], np.ndarray]:
        """
        Select the top `num_components` elementary matrices based on singular values.

        Parameters:
        -----------
        series_idx: int
            Index of the target time series.
        num_components: int
            Number of components to select.

        Returns:
        --------
        top_indices: List[int]
            Indices of the top components.
        reconstructed_series: np.ndarray
            Reconstructed time series using the top components.
        """
        if self.elementary_matrices is None:
            self.get_elementary_matrices()

        # Get the singular values for the target series
        singular_values = self.sigma[series_idx]

        # Rank components by absolute singular values
        ranked_indices = np.argsort(-singular_values)  # Descending order

        # Select the top components
        top_indices = ranked_indices[:num_components]

        # Reconstruct using the top components
        X_reconstructed = np.sum([self.elementary_matrices[series_idx][i] for i in top_indices], axis=0)
        reconstructed_series = self.diagonal_averaging(X_reconstructed)

        return top_indices, reconstructed_series

def linear_dependence (input):

    arr = statm.zscore(input, axis=1)
    x = np.corrcoef(arr.T, rowvar=False)
    x = np.nan_to_num(x)

    eigenvalues, eigenvectors = np.linalg.eig(x)
    min_variance_index = np.argmin(eigenvalues)
    min_variance_eigenvector = eigenvectors[:, min_variance_index]

    linear_dependence = np.var(np.dot(arr.T, min_variance_eigenvector))

    return linear_dependence

def SVD(input, X):
    """
    Perform SVD-based linear regression and evaluate the model performance.

    Parameters:
    - input (array-like): Target variable (response vector).
    - X (array-like): Design matrix (features).

    Returns:
    - beta_svd (ndarray): Coefficients estimated using SVD.
    - mse (float): Mean squared error of the predictions.
    """
    try:
        # Ensure X is a 2D matrix
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Compute pseudo-inverse of Σ (diagonal matrix)
        S_inv = np.diag(1 / S)

        # Solve for regression coefficients
        beta_svd = Vt.T @ S_inv @ U.T @ input

        # Make predictions
        y_pred = X @ beta_svd

        # Compute mean squared error
        mse = mean_squared_error(input, y_pred)

        return beta_svd, mse

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Error in SVD computation: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

def sparse_SVD(input, X, alpha=1.0):
    """
    Perform sparse SVD-based linear regression and evaluate model performance.

    Parameters:
    - input (array-like): Target variable (response vector), length n.
    - X (array-like): Design matrix (features), shape (n, p).
    - alpha (float): Regularization strength for L1 regularization (Lasso).

    Returns:
    - beta_sparse (ndarray): Sparse coefficients estimated using LASSO.
    - mse (float): Mean squared error of the predictions.
    """
    try:
        # Ensure X is a 2D matrix
        X = np.atleast_2d(X)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")

        # Perform SVD decomposition
        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        # Compute pseudo-inverse of Σ (diagonal matrix)
        S_inv = np.diag(1 / S)

        # Solve for regression coefficients using the pseudo-inverse
        beta_svd = Vt.T @ S_inv @ U.T @ input

        # Apply LASSO for sparsity control on the coefficients
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, input)

        # Extract sparse coefficients
        beta_sparse = lasso.coef_

        # Make predictions
        y_pred = X @ beta_sparse

        # Compute mean squared error
        mse = mean_squared_error(input, y_pred)

        return beta_sparse, mse

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Error in SVD computation: {e}")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")

def test_SVD(input_matrix, sparse_opt=False, sparse_level=0.1):
    """
    Tests SVD or sparse SVD on a given input matrix.

    Parameters:
        input_matrix (np.ndarray): 2D array of input data.
        sparse_opt (bool): If True, use sparse SVD.
        sparse_level (float, optional): Sparsity level for sparse SVD.

    Returns:
        predict (list): List of predicted beta values.
        result (list): List of mean squared errors (MSEs).
    """
    # Select the SVD function dynamically
    svd = lambda input, X: sparse_SVD(input, X, alpha=sparse_level) if sparse_opt else SVD(input, X)

    predict = []
    result = []

    # Loop through each row in the input matrix
    for i in range(len(input_matrix)):
        ts = input_matrix[i]  # Current time series (row)

        # Remove the i-th row to create the decomposition matrix
        decompose_ts = np.concatenate((input_matrix[:i], input_matrix[i + 1:]), axis=0)

        # Apply the selected SVD method
        beta_svd, mse = svd(ts, decompose_ts.T)

        # Collect results
        predict.append(beta_svd)
        result.append(mse)

    return predict, result

def graphical_LASSO_markov_net(time_series_data, sparse_level):
    """
    Fit a Graphical Lasso model and infer the Markov network structure for time series data,
    with edge weights in the Markov network representing the absolute values of the
    corresponding entries in the precision matrix.

    Parameters:
    - time_series_data (array-like): Input data matrix where rows are time series (e.g., 171)
      and columns are time points or features (e.g., 108).
    - sparse_level (float): Regularization parameter controlling sparsity.

    Returns:
    - model: The fitted GraphicalLasso model.
    - precision_matrix: Estimated sparse precision (inverse covariance) matrix.
    - markov_net: Weighted adjacency matrix representing the inferred Markov network.
    """
    # Standardize across time points (axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(time_series_data.T).T

    # Transpose so samples are rows, features are columns
    data_T = data.T  # shape: (n_samples, n_features)

    # Fit Graphical Lasso directly to data (not to covariance matrix)
    model = GraphicalLasso(alpha=sparse_level)
    model.fit(data_T)

    # Get the precision matrix
    precision_matrix = model.precision_

    # Absolute weights for adjacency
    markov_net = np.abs(precision_matrix)
    np.fill_diagonal(markov_net, 0)

    return model, precision_matrix, markov_net

def markov_blanket_all_sorted(markov_net):
    """
    Extract and sort the Markov Blankets for all nodes in a Markov network by edge weight.

    Parameters:
    - markov_net (array-like): Weighted adjacency matrix of the Markov network (shape [n_nodes, n_nodes]).
      markov_net[i, j] represents the weight of the edge between nodes i and j.
      A weight of 0 means no edge exists.

    Returns:
    - blankets (dict): A dictionary where each key is a node index, and the value is:
        - List of neighbors sorted by edge weight (Markov Blanket) if the node is valid.
        - Empty list if the node fails validation.
    """
    # Validate input adjacency matrix
    if not isinstance(markov_net, np.ndarray):
        raise ValueError("Markov network must be a NumPy array.")
    if markov_net.shape[0] != markov_net.shape[1]:
        raise ValueError("Markov network adjacency matrix must be square.")

    # Initialize the blanket dictionary
    n_nodes = markov_net.shape[0]
    blankets = {}

    for node in range(n_nodes):
        # Validate the node index
        if not (0 <= node < n_nodes):
            # Invalid node index
            blankets[node] = []
        else:
            # Extract neighbors and their edge weights
            neighbors = np.where(markov_net[node] != 0)[0]
            weights = markov_net[node, neighbors]

            # Pair neighbors with their weights and sort by weight (descending)
            sorted_neighbors = sorted(zip(neighbors, weights), key=lambda x: x[1], reverse=True)

            # Store sorted neighbors
            blankets[node] = [neighbor for neighbor, weight in sorted_neighbors]

    return blankets

def eval_SVD(data, k, top, tag=''):

    predics = data['predict']
    mlp = []
    st = []
    for i in range(len(predics)):
        predic = abs(predics[i])
        preds = np.argsort(predic)[::-1]

        tmp_mp = []
        tmp_st = []
        for pred in combinations(preds, k-1):
            pred = np.array(pred)
            pred[pred >= i] += 1
            arr = np.concatenate(([data[i]], data[pred]), axis=0)
            ld = linear_dependence(arr)
            ml = np.zeros((k), dtype=int)
            ml[0] = i
            ml[-(k - 1):] = [x for x in pred]
            tmp_mp.append(ml)
            tmp_st.append(ld)
            if len(tmp_mp) == top:
                break
        mlp.append(tmp_mp)

        st.append(np.average(tmp_st))

    df = pd.DataFrame({
        'Multipoles': mlp,
        'Strength': st,
    })
    df.to_excel(f'results/test_{tag}_SVD_{k}_pole_top{top}.xlsx', index=False)

def eval_LASSO(data, k, top, tag, save_opt = True):
    """
    Evaluate linear dependence of k-node combinations based on Markov blankets.

    Parameters:
    - data: dictionary with keys:
        - 'markov_blanket': dict of node -> list of neighbors
        - 'time_series': 2D NumPy array of shape [n_nodes, n_samples]
    - k: size of each multipole (e.g., 3 means triplets)
    - top: maximum number of combinations to keep per node
    """
    mb = data['markov_blanket']
    ts_data = data['time_series']

    mlp = []  # List of multipole node indices
    st = []   # List of strengths (linear dependence)

    for i in range(len(mb)):
        tmp_mp = []
        tmp_st = []

        predicts = np.array(mb[i])
        predict = predicts[predicts != i]

        if len(predict) < k - 1:
            mlp.append([])
            st.append(0)
            continue

        for pred in combinations(predict, k - 1):
            pred = np.array(pred)
            arr = np.concatenate(([ts_data[i]], ts_data[pred]), axis=0)
            ld = linear_dependence(arr)

            ml = np.zeros((k,), dtype=int)
            ml[0] = i
            ml[1:] = pred
            tmp_mp.append(ml)
            tmp_st.append(ld)

            if len(tmp_mp) == top:
                break

        mlp.append(tmp_mp)
        st.append(np.average(tmp_st) if tmp_st else 0)

    if save_opt:
        df = pd.DataFrame({
            'Multipoles': mlp,
            'Strength': st,
        })
        df.to_excel(f'results/test_{tag}_LASSO_{k}_pole_top{top}.xlsx', index=False)

    return {'multipoles': mlp, 'strenght': st}


def find_related_series(ssa: SSA, target_series_idx: int, metric="correlation") -> List[int]:
    """
    Find indices of time series in X most similar to the target series.

    Parameters:
    -----------
    ssa: SSA
        An instance of the SSA class.
    target_series_idx: int
        Index of the target time series in X.
    metric: str
        Similarity metric to use ("correlation" or "mse").

    Returns:
    --------
    ranked_indices: List[int]
        Indices of time series in X ranked by similarity to the target.
    """
    target_series = ssa.time_series[target_series_idx]
    similarities = []

    for i in range(ssa.n_series):
        reconstructed = np.sum(ssa.elementary_matrices[i], axis=0)  # Reconstruct full series
        reconstructed_series = ssa.diagonal_averaging(reconstructed)

        if metric == "correlation":
            similarity = np.corrcoef(target_series, reconstructed_series)[0, 1]
        elif metric == "mse":
            similarity = -np.mean((target_series - reconstructed_series) ** 2)  # Negate MSE for ranking
        else:
            raise ValueError("Unsupported metric. Use 'correlation' or 'mse'.")

        similarities.append(similarity)

    # Rank series by similarity
    ranked_indices = np.argsort(-np.array(similarities))  # Descending order
    return ranked_indices

def k_pole_of_ts(datapath, tag, index, k, threshold):
    print(f'Begin finding {k}_pole of {index} in {tag}!')
    start_time = time.time()

    data = np.load(datapath, allow_pickle=True)
    list_index = range(len(data))
    list_index = np.delete(list_index, index)

    results = []
    for pred in combinations(list_index, k - 1):
        pred = np.array(pred)
        arr = np.concatenate(([data[index]], data[pred]), axis=0)

        ld = linear_dependence(arr)
        if ld > threshold:
            results.append({
                'Multipoles': [index] + pred.tolist(),
                'Strength': ld
            })

    output_file = f'result_{k}_pole_of_ts{index}_{tag}.csv'
    pd.DataFrame(results).to_csv(output_file, index=False)

    elapsed_time = (time.time() - start_time) / 60
    print(f'Finished finding {k}_pole after {elapsed_time:.2f} minutes')


def evaluate_predictions(predicted_lists, true_label_lists, top_k_list=[1, 2, 3]):
    """
    predicted_lists: List[List[List[int]]] - each inner list is a group of predicted labels
    true_label_lists: List[List[int]] - true labels per sample
    top_k_list: List[int] - evaluate using the first k prediction groups per sample

    Returns a dict with @k precision, recall, f1, and accuracy
    """
    assert len(predicted_lists) == len(true_label_lists), "Length mismatch between predictions and true labels"

    results = {}

    for k in top_k_list:
        all_precision = []
        all_recall = []
        all_f1 = []
        acc_hits = 0

        for preds, trues in zip(predicted_lists, true_label_lists):
            # Flatten the first k prediction groups
            top_k_preds = set(label for group in preds[:k] for label in group)
            true_set = set(trues)

            match = top_k_preds & true_set
            num_hits = len(match)

            precision = num_hits / len(top_k_preds) if top_k_preds else 0
            recall = num_hits / len(true_set) if true_set else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

            if num_hits > 0:
                acc_hits += 1

        results[f'@{k}_precision'] = np.mean(all_precision)
        results[f'@{k}_recall'] = np.mean(all_recall)
        results[f'@{k}_f1'] = np.mean(all_f1)
        results[f'@{k}_acc'] = acc_hits / len(predicted_lists)

    return results


def unsup_LASSO_test(data, k_query, top_ks = [1,5,10], tag='', save_opt = true):
    _, pm, mn = graphical_LASSO_markov_net(data, 0.5)
    output = {
        'markov_blanket': markov_blanket_all_sorted(mn),
        'time_series': syn_data
    }

    result = []
    for top_k in top_ks:
        result.append(eval_LASSO(output, k_query, top=top_k, tag=tag, save_opt=save_opt))

    return result


def unsup_SVD_test(data, k_query, sparse_level=0, top_ks=[1, 5, 10], tag=''):
    if sparse_level > 0:
        pred, result = test_SVD(data, sparse_opt=True, sparse_level=sparse_level)
    else:
        pred, result = test_SVD(data)

    result = []
    for top_k in top_ks:
        result.append(eval_SVD(pred, k=k_query, top=top_k, tag=tag))

    return result

def unsup_SSA_test(
    data: np.ndarray,
    k_query: int,
    top_ks: List[int] = [1, 5, 10],
    tag: str = "",
    save_opt: bool = True,
) -> Dict[str, Any]:
    """
    Unsupervised SSA pass:
      - choose a safe window length L,
      - run SSA on each series,
      - select top components by singular values,
      - reconstruct for multiple top-k values (top_ks),
      - return indices, reconstructions, and energy fractions.

    Args:
        data: (n_series, series_length) array (raw or unnormalized)
        k_query: default number of components to keep (also included in top_ks if missing)
        top_ks: list of k values for reconstructions (e.g., [1, 5, 10])
        tag: label for filenames
        save_opt: whether to save outputs to disk

    Returns:
        dict with:
          - 'L': chosen window length
          - 'sigma': list of singular values per series
          - 'top_indices': dict {k: list[series -> indices]}
          - 'reconstructed': dict {k: (n_series, series_length) np.ndarray}
          - 'energy': dict {k: list[series -> energy_fraction]}
    """
    assert data.ndim == 2, "data must be (n_series, series_length)"
    n_series, series_length = data.shape

    # --- pick a safe window length L (heuristic) ---
    # Must satisfy 2 ≤ L ≤ series_length//2
    # Heuristic: N/5 clipped to [2, N//2]
    L = int(np.clip(series_length // 5, 2, series_length // 2))

    # Ensure k_query is considered in top_ks
    if k_query not in top_ks:
        top_ks = sorted(set(top_ks + [k_query]))

    # --- run SSA ---
    ssa = SSA(time_series=data, L=L)
    ssa.embed(normalized=True)   # disable if data already z-scored
    ssa.decompose()
    ssa.get_elementary_matrices()

    # Collect singular values per series (for energy computation)
    # sigma[series] is a 1-D array of singular values
    sigma_list = [np.asarray(ssa.sigma[i]) for i in range(n_series)]

    # Helper: pick top-k component indices by singular value for a series
    def top_k_indices_for_series(series_idx: int, k: int) -> np.ndarray:
        sig = sigma_list[series_idx]
        k_eff = min(k, sig.shape[0])
        ranked = np.argsort(-sig)  # descending by singular value
        return ranked[:k_eff]

    # --- build outputs for each k in top_ks ---
    top_indices: Dict[int, List[np.ndarray]] = {k: [] for k in top_ks}
    reconstructed: Dict[int, np.ndarray] = {}
    energy: Dict[int, List[float]] = {k: [] for k in top_ks}

    # Pre-allocate reconstructed arrays per k
    for k in top_ks:
        reconstructed[k] = np.zeros((n_series, series_length), dtype=float)

    for s in range(n_series):
        sig = sigma_list[s]
        total_energy = float((sig ** 2).sum()) if sig.size else 0.0

        for k in top_ks:
            idxs = top_k_indices_for_series(s, k)
            top_indices[k].append(idxs)

            # reconstruct series s from selected components
            X_rec = np.sum([ssa.elementary_matrices[s][c] for c in idxs], axis=0) if idxs.size > 0 else np.zeros((L, series_length - L + 1))
            recon = ssa.diagonal_averaging(X_rec)
            reconstructed[k][s] = recon

            # energy fraction explained by top-k
            top_energy = float((sig[idxs] ** 2).sum()) if idxs.size > 0 else 0.0
            frac = (top_energy / total_energy) if total_energy > 0 else 0.0
            energy[k].append(frac)

    result = {
        "L": L,
        "sigma": sigma_list,
        "top_indices": top_indices,          # {k: [array_of_indices_per_series]}
        "reconstructed": reconstructed,      # {k: (n_series, series_length)}
        "energy": energy,                    # {k: [float per series]}
    }

    if save_opt:
        os.makedirs("results", exist_ok=True)
        tag_sfx = f"_{tag}" if tag else ""
        # Save a compact npz with per-k recon + metadata
        np.savez_compressed(
            f"results/ssa_unsup{tag_sfx}_L{L}.npz",
            L=L,
            top_ks=np.array(top_ks, dtype=int),
            # store singular values lengths & flattened values for portability
            sigma_lens=np.array([len(s) for s in sigma_list], dtype=int),
            sigma_flat=np.concatenate(sigma_list) if sigma_list else np.array([], dtype=float),
            # reconstructions per k
            **{f"recon_k{k}": reconstructed[k] for k in top_ks},
        )
        # Save indices and energy in a pickle-friendly npz
        # (convert lists-of-arrays to object dtype)
        np.savez_compressed(
            f"results/ssa_unsup_meta{tag_sfx}_L{L}.npz",
            top_indices=np.array([[top_indices[k][s] for s in range(n_series)] for k in top_ks], dtype=object),
            energy=np.array([[energy[k][s] for s in range(n_series)] for k in top_ks], dtype=float),
            top_ks=np.array(top_ks, dtype=int),
        )

    return result



def unsup_methods_test(data, k_query, sparse_level=0, top_ks=[1, 5, 10], tag='', save_opt=true):

    unsup_SVD_test(data, k_query, sparse_level, top_ks, tag)

    unsup_LASSO_test(data, k_query, top_ks, tag, save_opt)

    unsup_SSA_test(data, k_query, sparse_level, top_ks, tag)
