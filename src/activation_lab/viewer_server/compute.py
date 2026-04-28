"""Numpy kernels for per-npz and pair-compare computations."""
from __future__ import annotations

import numpy as np

SOURCES = ("hidden_in", "hidden_out", "attn_out", "mlp_down_out", "qkv_last")


def _discover_layers(keys: list[str]) -> list[int]:
    idx: set[int] = set()
    for k in keys:
        if k.startswith("layer_") and "/" in k:
            head = k.split("/", 1)[0]
            try:
                idx.add(int(head.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(idx)


def layers_available(tensors: dict[str, np.ndarray]) -> list[int]:
    layers = _discover_layers(list(tensors.keys()))
    if layers:
        return layers
    # Snapshot/reference NPZs may only carry qkv_last (L, D); derive layer count from it.
    if "qkv_last" in tensors:
        arr = np.asarray(tensors["qkv_last"])
        if arr.ndim == 2:
            return list(range(arr.shape[0]))
    return []


def has_attention(tensors: dict[str, np.ndarray]) -> bool:
    return any(k.endswith("/attn_weights") for k in tensors)


def has_qk(tensors: dict[str, np.ndarray]) -> bool:
    return any(k.endswith("/q") for k in tensors) and any(k.endswith("/k") for k in tensors)


def has_qkv_last(tensors: dict[str, np.ndarray]) -> bool:
    return "qkv_last" in tensors


def qkv_last_matrix(tensors: dict[str, np.ndarray]) -> np.ndarray:
    """Return the per-layer ⟨q,k⟩·v row stack — shape (num_layers, D)."""
    arr = np.asarray(tensors["qkv_last"], dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"qkv_last expected 2D, got {arr.shape}")
    return arr


# ---------------------------------------------------------------- attention

def attention_matrix(tensors: dict[str, np.ndarray], layer: int, head_mean: bool = True) -> np.ndarray:
    """Return softmax attention for a layer.

    Handles three storage shapes:
      (B, H, T_q, T_kv) — full step capture          → drop batch → (H, T_q, T_kv)
      (H, T_kv)         — pooled reference (1 query)  → expand     → (H, 1, T_kv)
      (T_q, T_kv)       — already head-averaged        → as-is
    Returns (T_q, T_kv) when head_mean=True, else (H, T_q, T_kv).
    """
    key = f"layer_{layer:02d}/attn_weights"
    arr = np.asarray(tensors[key], dtype=np.float32)
    if arr.ndim == 4:           # (B, H, T_q, T_kv)
        arr = arr[0]            # (H, T_q, T_kv)
    elif arr.ndim == 2:         # (H, T_kv) — pooled reference
        arr = arr[:, np.newaxis, :]  # (H, 1, T_kv)
    if head_mean:
        return arr.mean(axis=0)  # (T_q, T_kv)
    return arr


def qk_scores(tensors: dict[str, np.ndarray], layer: int) -> np.ndarray:
    """Compute raw q·k/sqrt(dk) averaged over heads — shape (T_q, T_kv).

    Handles:
      (B, T, D) — full per-token capture  → drop batch → (T, D)
      (D,)      — pooled reference         → reshape   → (1, D)
    """
    q = np.asarray(tensors[f"layer_{layer:02d}/q"], dtype=np.float32)
    k = np.asarray(tensors[f"layer_{layer:02d}/k"], dtype=np.float32)
    if q.ndim == 3:
        q = q[0]
        k = k[0]
    elif q.ndim == 1:           # pooled reference: (D,) → (1, D)
        q = q[np.newaxis, :]
        k = k[np.newaxis, :]
    # Attempt head split using q as reference. Assume head_dim is inferable when Hq*D == q.shape[-1].
    # Use 64 as a common head_dim; fall back to a single-head score over the flat features.
    T, qdim = q.shape
    Tk, kdim = k.shape
    head_dim = _infer_head_dim(qdim, kdim)
    if head_dim is None:
        scores = q @ k.T / np.sqrt(qdim)
        return scores
    num_q_heads = qdim // head_dim
    num_k_heads = kdim // head_dim
    qh = q.reshape(T, num_q_heads, head_dim)
    kh = k.reshape(Tk, num_k_heads, head_dim)
    # Average over q-heads (and broadcast kv-heads if grouped)
    # groups per kv-head:
    group = max(1, num_q_heads // num_k_heads)
    # (T, Hq, D) → group kv: for each q-head i, use kv-head i // group
    out = np.zeros((T, Tk), dtype=np.float32)
    for h in range(num_q_heads):
        kh_idx = h // group if num_k_heads > 0 else 0
        out += qh[:, h, :] @ kh[:, kh_idx, :].T
    out /= max(num_q_heads, 1)
    return out / np.sqrt(head_dim)


def _infer_head_dim(qdim: int, kdim: int) -> int | None:
    for d in (128, 64, 96, 80, 48, 32):
        if qdim % d == 0 and kdim % d == 0:
            return d
    return None


def attention_entropy(tensors: dict[str, np.ndarray]) -> list[float]:
    """Mean attention entropy per layer (nats). Higher = more diffuse. Returns list indexed by layer."""
    layers = layers_available(tensors)
    out: list[float] = []
    for layer in layers:
        key = f"layer_{layer:02d}/attn_weights"
        if key not in tensors:
            out.append(float("nan"))
            continue
        arr = np.asarray(tensors[key], dtype=np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        # arr: (H, T, T); entropy per query token then mean
        eps = 1e-9
        ent = -(arr * np.log(arr + eps)).sum(axis=-1)  # (H, T)
        out.append(float(ent.mean()))
    return out


def stack_all_layers(extract) -> np.ndarray:
    """Stack per-layer (T, T) matrices vertically (list-style), returning (L*T, T) image.

    `extract` is a callable layer_idx -> (T, T).
    """
    mats: list[np.ndarray] = []
    for m in extract:
        mats.append(m)
    return np.vstack(mats)


# ---------------------------------------------------------------- pair metrics

def _per_layer_vectors(tensors: dict[str, np.ndarray], source: str) -> dict[int, np.ndarray]:
    """Collect per-layer tensors for a source, reduced to (T, D) by dropping batch dim."""
    # qkv_last is stored as one stacked (L, D) matrix — expose each row as a (1, D) layer vector.
    if source == "qkv_last":
        if "qkv_last" not in tensors:
            return {}
        arr = qkv_last_matrix(tensors)
        return {layer: arr[layer][None, :] for layer in range(arr.shape[0])}

    out: dict[int, np.ndarray] = {}
    for layer in layers_available(tensors):
        key = f"layer_{layer:02d}/{source}"
        if key not in tensors:
            continue
        arr = np.asarray(tensors[key], dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr[None, :]  # (1, D) — pooled snapshot fallback
        out[layer] = arr
    return out


def _cosine_per_token(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a, b: (T, D). returns (T,)
    na = np.linalg.norm(a, axis=-1) + 1e-12
    nb = np.linalg.norm(b, axis=-1) + 1e-12
    return (a * b).sum(axis=-1) / (na * nb)


def pair_metrics(
    a: dict[str, np.ndarray],
    b: dict[str, np.ndarray],
    sources: list[str],
) -> dict[str, dict[str, list[float]]]:
    """For each source, compute per-layer:
        mean_cos, min_cos, mean_mae, mean_overlap (top-1% Jaccard), std_diff.
    Returns dict[source] = { metric: [value per layer] }. Layers are intersected.
    """
    layers = sorted(set(layers_available(a)) & set(layers_available(b)))
    result: dict[str, dict[str, list[float]]] = {}
    for src in sources:
        a_by = _per_layer_vectors(a, src)
        b_by = _per_layer_vectors(b, src)
        metrics = {k: [] for k in ("mean_cos", "min_cos", "mean_mae", "mean_overlap", "std_diff", "layers")}
        for layer in layers:
            if layer not in a_by or layer not in b_by:
                continue
            av = a_by[layer]
            bv = b_by[layer]
            # Align on min T dim (in case different lengths)
            T = min(av.shape[0], bv.shape[0])
            av = av[:T]
            bv = bv[:T]
            cos_vec = _cosine_per_token(av, bv)
            diff = av - bv
            metrics["layers"].append(layer)
            metrics["mean_cos"].append(float(cos_vec.mean()))
            metrics["min_cos"].append(float(cos_vec.min()))
            metrics["mean_mae"].append(float(np.mean(np.abs(diff))))
            metrics["std_diff"].append(float(np.std(diff)))
            metrics["mean_overlap"].append(_top1_jaccard(av, bv))
        result[src] = metrics
    return result


def multi_metrics(
    tensors_list: list[dict[str, np.ndarray]],
    sources: list[str],
) -> dict:
    """All-pairs metrics for >=2 references. Returns:
        { 'pairs': [{ 'i': int, 'j': int, 'metrics': pair_metrics_result }, ...] }
    """
    pairs: list[dict] = []
    n = len(tensors_list)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "i": i,
                "j": j,
                "metrics": pair_metrics(tensors_list[i], tensors_list[j], sources),
            })
    return {"pairs": pairs}


def _top1_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Average-over-token Jaccard of top-1% channel indices (by |value|)."""
    if a.ndim == 1:
        return _top1_jaccard_vec(a, b)
    T, D = a.shape
    k = max(1, D // 100)
    jac_vals = []
    for t in range(T):
        idx_a = set(np.argpartition(np.abs(a[t]), -k)[-k:].tolist())
        idx_b = set(np.argpartition(np.abs(b[t]), -k)[-k:].tolist())
        inter = len(idx_a & idx_b)
        union = len(idx_a | idx_b)
        jac_vals.append(inter / union if union else 0.0)
    return float(np.mean(jac_vals))


def _top1_jaccard_vec(a: np.ndarray, b: np.ndarray) -> float:
    D = a.shape[0]
    k = max(1, D // 100)
    idx_a = set(np.argpartition(np.abs(a), -k)[-k:].tolist())
    idx_b = set(np.argpartition(np.abs(b), -k)[-k:].tolist())
    inter = len(idx_a & idx_b)
    union = len(idx_a | idx_b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------- advanced multi-ref analysis


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float | None:
    """Efficient linear CKA via kernel matrices: O(T²D) not O(D²T).

    Returns None for T≤1 (uncentered CKA is trivially 1 for rank-1 captures).
    """
    T = min(X.shape[0], Y.shape[0])
    if T <= 1:
        return None
    X = X[:T].astype(np.float64) - X[:T].mean(axis=0)
    Y = Y[:T].astype(np.float64) - Y[:T].mean(axis=0)
    K_xx = X @ X.T   # (T, T)
    K_yy = Y @ Y.T
    num = float(np.trace(K_xx @ K_yy))
    denom = float(np.linalg.norm(K_xx, "fro") * np.linalg.norm(K_yy, "fro"))
    return num / denom if denom > 1e-12 else None


def svd_analysis(
    tensors_list: list[dict[str, np.ndarray]],
    sources: list[str],
    top_k: int = 32,
) -> dict:
    """Per-ref per-layer SVD: top-K singular values, spectral norm, nuclear norm, effective rank.

    For T=1 captures: spectral/nuclear norm = ‖v‖₂; effective_rank=null (trivially 1).
    """
    result: dict = {}
    for src in sources:
        by_ref = [_per_layer_vectors(t, src) for t in tensors_list]
        layers = sorted(set().union(*[set(d.keys()) for d in by_ref]))
        ref_results = []
        for rd in by_ref:
            spectra_r: list = []
            er_r: list = []
            nn_r: list = []
            sn_r: list = []
            for layer in layers:
                if layer not in rd:
                    spectra_r.append(None)
                    er_r.append(None)
                    nn_r.append(None)
                    sn_r.append(None)
                    continue
                arr = np.asarray(rd[layer], dtype=np.float64)
                T = arr.shape[0]
                if T > 1:
                    arr_c = arr - arr.mean(axis=0)
                    s = np.linalg.svd(arr_c, compute_uv=False)
                    s_pos = s[s > 1e-12]
                    spectra_r.append(s[:top_k].tolist())
                    if s_pos.size > 0:
                        p = s_pos / s_pos.sum()
                        er_r.append(float(np.exp(-np.sum(p * np.log(p + 1e-12)))))
                    else:
                        er_r.append(None)
                    nn_r.append(float(s.sum()))
                    sn_r.append(float(s[0]) if s.size > 0 else None)
                else:
                    norm = float(np.linalg.norm(arr[0]))
                    spectra_r.append([norm])
                    er_r.append(None)
                    nn_r.append(norm)
                    sn_r.append(norm)
            ref_results.append({
                "spectra": spectra_r,
                "effective_rank": er_r,
                "nuclear_norm": nn_r,
                "spectral_norm": sn_r,
            })
        result[src] = {"layers": [int(la) for la in layers], "refs": ref_results}
    return result


def cka_analysis(
    tensors_list: list[dict[str, np.ndarray]],
    sources: list[str],
) -> dict:
    """Per-layer linear CKA matrix (N×N). None where T≤1 or data missing."""
    n = len(tensors_list)
    result: dict = {}
    for src in sources:
        by_ref = [_per_layer_vectors(t, src) for t in tensors_list]
        layers = sorted(set().union(*[set(d.keys()) for d in by_ref]))
        matrices = []
        for layer in layers:
            mat = []
            for i in range(n):
                row = []
                for j in range(n):
                    if layer in by_ref[i] and layer in by_ref[j]:
                        row.append(_linear_cka(by_ref[i][layer], by_ref[j][layer]))
                    else:
                        row.append(None)
                mat.append(row)
            matrices.append(mat)
        result[src] = {"layers": [int(la) for la in layers], "matrices": matrices}
    return result


def pca_analysis(
    tensors_list: list[dict[str, np.ndarray]],
    sources: list[str],
) -> dict:
    """PCA on stacked per-ref per-layer mean vectors → 2D layer-trajectory coordinates."""
    result: dict = {}
    for src in sources:
        by_ref = [_per_layer_vectors(t, src) for t in tensors_list]
        layers = sorted(set().union(*[set(d.keys()) for d in by_ref]))

        points: list[np.ndarray] = []
        ref_idxs: list[int] = []
        layer_idxs: list[int] = []
        for li, layer in enumerate(layers):
            for ri, rd in enumerate(by_ref):
                if layer in rd:
                    points.append(rd[layer].mean(axis=0).astype(np.float64))
                    ref_idxs.append(ri)
                    layer_idxs.append(li)

        if len(points) < 2:
            result[src] = {
                "coords": [], "ref_indices": [], "layer_indices": [],
                "layers": [int(la) for la in layers], "explained_variance": [0.0, 0.0],
            }
            continue

        X = np.stack(points)
        X_c = X - X.mean(axis=0)
        _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
        total_var = float((s ** 2).sum()) or 1.0
        n_comp = min(2, Vt.shape[0])
        proj_raw = X_c @ Vt[:n_comp].T
        # Guarantee 2-column output even when rank < 2
        if n_comp < 2:
            zeros = np.zeros((proj_raw.shape[0], 2 - n_comp))
            proj_raw = np.concatenate([proj_raw, zeros], axis=1)
        ev = [
            float(s[0] ** 2 / total_var),
            float(s[1] ** 2 / total_var) if len(s) > 1 else 0.0,
        ]
        result[src] = {
            "coords": proj_raw.tolist(),
            "ref_indices": ref_idxs,
            "layer_indices": layer_idxs,
            "layers": [int(la) for la in layers],
            "explained_variance": ev,
        }
    return result


# ---------------------------------------------------------------- multi-snapshot group analysis


def _layer_mean_vector(arr: np.ndarray) -> np.ndarray:
    """Reduce a per-source layer tensor (T, D) or (D,) to a (D,) mean vector."""
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=0)


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u)) + 1e-12
    nv = float(np.linalg.norm(v)) + 1e-12
    return float(np.dot(u, v) / (nu * nv))


def _topk_indices(v: np.ndarray, k: int) -> set[int]:
    if k >= v.size:
        return set(range(v.size))
    return set(np.argpartition(np.abs(v), -k)[-k:].tolist())


def _average_linkage(dist: np.ndarray) -> list[list[float]]:
    """Lance-Williams average-linkage clustering. dist: symmetric (N, N).

    Returns SciPy-style linkage rows: [cluster_i, cluster_j, distance, member_count].
    Cluster ids 0..N-1 are leaves; new clusters get ids N, N+1, ...
    """
    n = dist.shape[0]
    if n < 2:
        return []
    D = dist.astype(np.float64).copy()
    np.fill_diagonal(D, np.inf)
    sizes = [1] * n
    ids = list(range(n))
    rows: list[list[float]] = []
    next_id = n
    active = list(range(n))
    while len(active) > 1:
        # Find min pair among active rows/cols
        sub = D[np.ix_(active, active)]
        flat = np.argmin(sub)
        ai, aj = divmod(flat, len(active))
        if ai == aj:
            break
        if ai > aj:
            ai, aj = aj, ai
        i, j = active[ai], active[aj]
        d_ij = float(D[i, j])
        ni, nj = sizes[i], sizes[j]
        rows.append([float(ids[i]), float(ids[j]), d_ij, float(ni + nj)])
        # Merge j into i with weighted average
        for m in active:
            if m == i or m == j:
                continue
            new_d = (ni * D[i, m] + nj * D[j, m]) / (ni + nj)
            D[i, m] = new_d
            D[m, i] = new_d
        sizes[i] = ni + nj
        ids[i] = next_id
        next_id += 1
        # Drop j
        D[j, :] = np.inf
        D[:, j] = np.inf
        active.remove(j)
    return rows


def group_analysis(
    tensors_list: list[dict[str, np.ndarray]],
    sources: list[str],
    topk_pct: float = 0.01,
) -> dict:
    """Per-source N-way comparison metrics for identifying matching snapshot patterns.

    Returns, per source:
      layers          : [int]
      cos_to_centroid : [[float per snapshot] per layer]  — alignment of each snapshot with the group mean
      divergence      : [float per layer]                   — 1 − mean pairwise cosine of layer means
      pairwise_cos    : [[float] (N,N)]                     — layer-averaged pairwise cosine of layer means
      topk_jaccard    : [[float] (N,N)]                     — layer-averaged Jaccard of top-k channel indices
      linkage         : [[i, j, dist, count]]               — average-link clustering on (1 − pairwise_cos)
    """
    n = len(tensors_list)
    result: dict = {}
    for src in sources:
        by_ref = [_per_layer_vectors(t, src) for t in tensors_list]
        layers = sorted(set().union(*[set(d.keys()) for d in by_ref])) if by_ref else []

        # Per-layer per-snapshot mean vectors. None where layer missing for that snapshot.
        means_per_layer: list[list[np.ndarray | None]] = []
        for layer in layers:
            row = []
            for rd in by_ref:
                row.append(_layer_mean_vector(rd[layer]) if layer in rd else None)
            means_per_layer.append(row)

        # cos to centroid + divergence per layer
        cos_centroid: list[list[float | None]] = []
        divergence: list[float | None] = []
        for row in means_per_layer:
            present = [v for v in row if v is not None]
            if len(present) < 2:
                cos_centroid.append([None] * n)
                divergence.append(None)
                continue
            centroid = np.mean(np.stack(present), axis=0)
            cs = [_cos(v, centroid) if v is not None else None for v in row]
            cos_centroid.append(cs)
            pair_cs: list[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if row[i] is not None and row[j] is not None:
                        pair_cs.append(_cos(row[i], row[j]))
            divergence.append(1.0 - float(np.mean(pair_cs)) if pair_cs else None)

        # Layer-averaged pairwise cosine and top-k Jaccard matrices
        cos_sum = np.zeros((n, n), dtype=np.float64)
        cos_cnt = np.zeros((n, n), dtype=np.int64)
        jac_sum = np.zeros((n, n), dtype=np.float64)
        jac_cnt = np.zeros((n, n), dtype=np.int64)
        for row in means_per_layer:
            # determine top-k size from any present vector
            ref_v = next((v for v in row if v is not None), None)
            if ref_v is None:
                continue
            D = ref_v.size
            k = max(1, int(round(D * topk_pct)))
            topk_sets = [None if v is None else _topk_indices(v, k) for v in row]
            for i in range(n):
                for j in range(n):
                    if row[i] is None or row[j] is None:
                        continue
                    cos_sum[i, j] += _cos(row[i], row[j]) if i != j else 1.0
                    cos_cnt[i, j] += 1
                    si, sj = topk_sets[i], topk_sets[j]
                    if si is not None and sj is not None:
                        union = len(si | sj)
                        jac_sum[i, j] += (len(si & sj) / union) if union else 0.0
                        jac_cnt[i, j] += 1

        def _avg_mat(s: np.ndarray, c: np.ndarray) -> list[list[float | None]]:
            out = [[None] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if c[i, j] > 0:
                        out[i][j] = float(s[i, j] / c[i, j])
            return out

        cos_mat = _avg_mat(cos_sum, cos_cnt)
        jac_mat = _avg_mat(jac_sum, jac_cnt)

        # Distance matrix for clustering: 1 - cos, fall back to 1.0 where missing.
        dist = np.ones((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                v = cos_mat[i][j]
                if v is not None:
                    dist[i, j] = max(0.0, 1.0 - v)
        np.fill_diagonal(dist, 0.0)
        # symmetrize defensively
        dist = 0.5 * (dist + dist.T)
        linkage = _average_linkage(dist)

        result[src] = {
            "layers": [int(la) for la in layers],
            "cos_to_centroid": cos_centroid,
            "divergence": divergence,
            "pairwise_cos": cos_mat,
            "topk_jaccard": jac_mat,
            "linkage": linkage,
        }
    return result


# ---------------------------------------------------------------- residual convergence

def residual_convergence(
    tensors: dict[str, np.ndarray],
    source: str = "hidden_out",
) -> dict:
    """Adjacent-layer cosine similarity and relative update norm of the residual stream.

    Returns both last-token and mean-over-sequence variants so callers can compare
    whether the stream settles uniformly or only at the final token position.

    Effective depth = first layer index where cosine similarity stays ≥ 0.99 for all
    subsequent layers (i.e. the stream has effectively stopped changing).
    """
    by_layer = _per_layer_vectors(tensors, source)
    layers = sorted(by_layer.keys())

    if len(layers) < 2:
        return {
            "layers": [int(l) for l in layers],
            "adj_layers": [],
            "adj_cosine_last": [],
            "adj_cosine_mean": [],
            "delta_norm_last": [],
            "delta_norm_mean": [],
            "effective_depth_last": None,
            "effective_depth_mean": None,
        }

    adj_layers: list[int] = []
    adj_cosine_last: list[float] = []
    adj_cosine_mean: list[float] = []
    delta_norm_last: list[float] = []
    delta_norm_mean: list[float] = []

    for i in range(len(layers) - 1):
        l0, l1 = layers[i], layers[i + 1]
        h0 = np.asarray(by_layer[l0], dtype=np.float32)
        h1 = np.asarray(by_layer[l1], dtype=np.float32)
        T = min(h0.shape[0], h1.shape[0])
        h0, h1 = h0[:T], h1[:T]

        cos_vec = _cosine_per_token(h0, h1)  # (T,)
        adj_cosine_mean.append(float(cos_vec.mean()))
        adj_cosine_last.append(float(cos_vec[-1]))

        delta = h1 - h0
        rel_norms = np.linalg.norm(delta, axis=-1) / (np.linalg.norm(h0, axis=-1) + 1e-12)
        delta_norm_mean.append(float(rel_norms.mean()))
        delta_norm_last.append(float(rel_norms[-1]))

        adj_layers.append(int(l0))

    THRESHOLD = 0.99

    def _effective_depth(cos_list: list[float]) -> int | None:
        for i in range(len(cos_list)):
            if all(c >= THRESHOLD for c in cos_list[i:]):
                return adj_layers[i]
        return None

    return {
        "layers": [int(l) for l in layers],
        "adj_layers": adj_layers,
        "adj_cosine_last": adj_cosine_last,
        "adj_cosine_mean": adj_cosine_mean,
        "delta_norm_last": delta_norm_last,
        "delta_norm_mean": delta_norm_mean,
        "effective_depth_last": _effective_depth(adj_cosine_last),
        "effective_depth_mean": _effective_depth(adj_cosine_mean),
    }


# ---------------------------------------------------------------- pair heatmap derivatives

def diff_matrices(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    """Given two (T, T) matrices, produce the derivative heatmaps."""
    abs_diff = np.abs(a - b)
    sq_err = (a - b) ** 2
    hadamard = a * b
    # Normalized ratio: a / (b + eps), clipped for visualization sanity
    eps = 1e-9
    ratio = a / (np.abs(b) + eps)
    ratio = np.clip(ratio, -50.0, 50.0)
    return {
        "abs_diff": abs_diff,
        "sq_err": sq_err,
        "hadamard": hadamard,
        "ratio": ratio,
    }
