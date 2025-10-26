import torch
import numpy as np
from itertools import combinations
from heapq import nlargest


def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim=1), dim=0)


def normalize(input, dim):
    min = torch.min(input, dim=dim)[0]
    max = torch.max(input, dim=dim)[0]
    x = (input.view(-1, 1024) - min) / (max - min)
    return x


def k_largest_exclude_index(x: np.ndarray, k: int, exclude_index):
    """
    Return indices of the k largest elements of x, excluding the given index (or indices).
    Uses argpartition for O(n) selection.
    """
    x = np.asarray(x)
    n = x.shape[0]

    # Normalize exclude_index to a boolean mask
    mask = np.ones(n, dtype=bool)
    if exclude_index is not None:
        if np.isscalar(exclude_index):
            mask[int(exclude_index)] = False
        else:
            mask[np.asarray(exclude_index, dtype=int)] = False

    # Candidates after exclusion
    cand_idx = np.nonzero(mask)[0]
    if cand_idx.size == 0 or k <= 0:
        return np.array([], dtype=int)

    k_eff = min(k, cand_idx.size)

    # Select top-k using argpartition (no full sort)
    cand_vals = x[cand_idx]
    part = np.argpartition(cand_vals, -k_eff)[-k_eff:]
    top_unsorted = cand_idx[part]

    # Order descending by value
    order = np.argsort(x[top_unsorted])[::-1]
    return top_unsorted[order]


def top_m_combinations(x: np.ndarray, k: int, exclude_index=None, m: int = 10, widen: int = 8):
    """
    Return the top-m k-combinations of indices (excluding exclude_index)
    ranked by the SUM of their values in x.

    Strategy:
      1) Exclude indices.
      2) Preselect a smaller candidate pool of size L = min(len, k + widen)
         using argpartition (keeps quality high while avoiding O(n choose k)).
      3) Enumerate all C(L, k) combos within the pool, rank by sum, take top-m.

    Params:
      x: 1D array of scores
      k: size of each combination
      exclude_index: int or iterable of ints to exclude
      m: number of top combos to return
      widen: controls pool size L ~= k + widen (increase if you want better accuracy)

    Returns:
      List[Tuple[np.ndarray, float]] where each item is (indices_array, total_score)
    """
    x = np.asarray(x)
    n = x.shape[0]

    # Build candidate mask
    mask = np.ones(n, dtype=bool)
    if exclude_index is not None:
        if np.isscalar(exclude_index):
            mask[int(exclude_index)] = False
        else:
            mask[np.asarray(exclude_index, dtype=int)] = False

    cand_idx = np.nonzero(mask)[0]
    if k <= 0 or cand_idx.size < k or m <= 0:
        return []

    # Preselect a manageable pool
    L = min(cand_idx.size, k + max(0, widen))
    # If L == cand_idx.size we just take all; else take top-L by value
    if L < cand_idx.size:
        part = np.argpartition(x[cand_idx], -L)[-L:]
        pool = cand_idx[part]
    else:
        pool = cand_idx

    # Enumerate combinations inside the pool and rank by sum
    # If C(L, k) is big, you can guard with a cap or increase 'widen' carefully.
    combos = combinations(pool, k)

    # Use heap to keep only m best by sum
    # nlargest will call the key only once per item; sum via x.take is fast.
    top = nlargest(m, combos, key=lambda idxs: float(np.sum(x[np.fromiter(idxs, dtype=int)])))

    # Build results with scores
    results = []
    for idxs in top:
        idxs = np.fromiter(idxs, dtype=int)
        results.append((idxs, float(x[idxs].sum())))

    return results