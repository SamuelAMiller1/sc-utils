# src/cellstate_aligner/align.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Tuple, Literal

import scanpy as sc
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

# ---------------- low-level helpers ----------------

def _ensure_1d(a): return np.asarray(a).ravel()
def _to_float_array(df: pd.DataFrame) -> np.ndarray: return np.asarray(df, dtype=np.float64)

def _rank_columns(A: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(rankdata, 0, A)

def _pearson_matrix(A: np.ndarray, B: np.ndarray):
    # A,B: genes × groups
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    A_std = A.std(axis=0, ddof=1, keepdims=True)
    B_std = B.std(axis=0, ddof=1, keepdims=True)
    nzA = (A_std > 0).ravel()
    nzB = (B_std > 0).ravel()
    A = A[:, nzA] / A_std[:, nzA]
    B = B[:, nzB] / B_std[:, nzB]
    M = (A.T @ B) / (A.shape[0] - 1)
    return M, nzA, nzB

def _spearman_matrix(A: np.ndarray, B: np.ndarray):
    Ar = _rank_columns(A)
    Br = _rank_columns(B)
    return _pearson_matrix(Ar, Br)

def _ccc(x: np.ndarray, y: np.ndarray) -> float:
    x = _ensure_1d(x); y = _ensure_1d(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    r = pearsonr(x, y)[0]
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return np.nan
    return (2 * r * np.sqrt(vx * vy)) / denom

def _maybe_import_dcor():
    try:
        from dcor import distance_correlation
        return distance_correlation
    except Exception as e:
        raise ImportError("Install `dcor` to use method='distance_correlation' (pip install dcor).") from e

# ---------------- feature selection & data prep ----------------

def _extract_features(adata1, adata2,
    features_use: Optional[Iterable[str]],
    genes_blacklist: Optional[Iterable[str]],
    highly_variable_only: bool,
    hv_key: str,
) -> list:
    v1, v2 = pd.Index(adata1.var_names), pd.Index(adata2.var_names)
    feats = pd.Index(features_use) if features_use is not None else v1.intersection(v2)
    feats = feats.intersection(v1).intersection(v2)
    if highly_variable_only:
        hv1 = adata1.var.index[adata1.var.get(hv_key, False)]
        hv2 = adata2.var.index[adata2.var.get(hv_key, False)]
        feats = feats.intersection(hv1).intersection(hv2)
    if genes_blacklist is not None:
        feats = feats.difference(pd.Index(genes_blacklist))
    feats = [g for g in adata1.var_names if g in feats]  # deterministic order
    if len(feats) == 0:
        raise ValueError("No overlapping features between adata1 and adata2 after filtering.")
    return feats

def _subset_to_features_and_layer(adata, features: list, layer: Optional[str]):
    A = adata[:, features].copy()
    if layer is not None:
        if layer not in A.layers:
            raise KeyError(f"Layer '{layer}' not found in AnnData.layers.")
        A.X = A.layers[layer].copy()
    return A

def _scale_per_dataset(A, zero_center: bool, max_value: Optional[float]):
    sc.pp.scale(A, max_value=max_value, zero_center=zero_center, copy=False)
    return A

def _group_pseudobulk(A, groupby: str):
    obs_cat = A.obs[groupby].astype('category')
    cats = list(obs_cat.cat.categories)
    genes = list(A.var_names)
    avg = pd.DataFrame(0.0, index=genes, columns=cats)
    sizes = obs_cat.value_counts()
    for cl in cats:
        Xg = A[obs_cat == cl, :].X
        avg.loc[:, cl] = _ensure_1d(np.asarray(Xg.mean(0)))
    return avg, cats, sizes

def _concat_pseudobulk_scale(avg1: pd.DataFrame, avg2: pd.DataFrame):
    C = pd.concat([avg1, avg2], axis=1)
    M = C.mean(axis=1).values[:, None]
    S = C.std(axis=1, ddof=1).values[:, None]
    S[S == 0] = 1.0
    Z = (C.values - M) / S
    a1z = pd.DataFrame(Z[:, :avg1.shape[1]], index=avg1.index, columns=avg1.columns)
    a2z = pd.DataFrame(Z[:, avg1.shape[1]:], index=avg2.index, columns=avg2.columns)
    return a1z, a2z

# ---------------- similarity calculators (rows=g1, cols=g2) ----------------

def _cosine_vectorized(avg1, avg2):
    A = _to_float_array(avg1).T  # groups × genes
    B = _to_float_array(avg2).T
    M = cosine_similarity(A, B)
    return pd.DataFrame(M, index=avg1.columns, columns=avg2.columns)

def _pearson_vectorized(avg1, avg2):
    A = _to_float_array(avg1); B = _to_float_array(avg2)
    M, nzA, nzB = _pearson_matrix(A, B)
    out = pd.DataFrame(np.nan, index=avg1.columns, columns=avg2.columns)
    out.iloc[np.where(nzA)[0], np.where(nzB)[0]] = M
    return out

def _spearman_vectorized(avg1, avg2):
    A = _to_float_array(avg1); B = _to_float_array(avg2)
    M, nzA, nzB = _spearman_matrix(A, B)
    out = pd.DataFrame(np.nan, index=avg1.columns, columns=avg2.columns)
    out.iloc[np.where(nzA)[0], np.where(nzB)[0]] = M
    return out

def _kendall_loop(avg1, avg2, pbar=None):
    out = pd.DataFrame(np.nan, index=avg1.columns, columns=avg2.columns)
    for g1 in avg1.columns:
        x = avg1[g1].values
        for g2 in avg2.columns:
            out.loc[g1, g2] = kendalltau(x, avg2[g2].values, nan_policy='omit')[0]
            if pbar is not None: pbar.update()
    return out

def _ccc_loop(avg1, avg2, pbar=None):
    out = pd.DataFrame(np.nan, index=avg1.columns, columns=avg2.columns)
    for g1 in avg1.columns:
        x = avg1[g1].values
        for g2 in avg2.columns:
            out.loc[g1, g2] = _ccc(x, avg2[g2].values)
            if pbar is not None: pbar.update()
    return out

def _dcorr_loop(avg1, avg2, pbar=None):
    distance_correlation = _maybe_import_dcor()
    out = pd.DataFrame(np.nan, index=avg1.columns, columns=avg2.columns)
    for g1 in avg1.columns:
        x = avg1[g1].values
        for g2 in avg2.columns:
            out.loc[g1, g2] = distance_correlation(x, avg2[g2].values)
            if pbar is not None: pbar.update()
    return out

def _hungarian_match(sim_df: pd.DataFrame) -> pd.DataFrame:
    cost = -sim_df.values
    r_ind, c_ind = linear_sum_assignment(cost)
    rows = sim_df.index.to_numpy()[r_ind]
    cols = sim_df.columns.to_numpy()[c_ind]
    scores = sim_df.values[r_ind, c_ind]
    return pd.DataFrame({'group1': rows, 'group2': cols, 'score': scores})

# ---------------- public API: function ----------------

def cell_state_correlation(
    adata1, adata2, groupby1: str, groupby2: str,
    method: Literal['pearson','spearman','kendall','cosine','distance_correlation','ccc']='cosine',
    *,
    features_use: Optional[Iterable[str]] = None,
    genes_blacklist: Optional[Iterable[str]] = None,
    highly_variable_only: bool = False,
    hv_key: str = 'highly_variable',
    layer: Optional[str] = None,
    scale: bool = True,
    scale_mode: Literal['per_dataset','concat_pseudobulk','none'] = 'concat_pseudobulk',
    zero_center: bool = False,
    max_value: Optional[float] = 6.0,
    min_cells_per_group: int = 1,
    verbose: bool = True,
    return_matching: bool = False,
):
    valid = {'pearson','spearman','kendall','cosine','distance_correlation','ccc'}
    if method not in valid:
        raise ValueError(f"Invalid method '{method}'. Choose from {sorted(valid)}")
    if scale_mode not in {'per_dataset','concat_pseudobulk','none'}:
        raise ValueError("scale_mode must be one of {'per_dataset','concat_pseudobulk','none'}")

    feats = _extract_features(
        adata1, adata2,
        features_use=features_use,
        genes_blacklist=genes_blacklist,
        highly_variable_only=highly_variable_only,
        hv_key=hv_key,
    )
    if verbose:
        print(f"Using {len(feats)} features for {method} (scale_mode={scale_mode})")

    A1 = _subset_to_features_and_layer(adata1, feats, layer)
    A2 = _subset_to_features_and_layer(adata2, feats, layer)
    if scale and scale_mode == 'per_dataset':
        _scale_per_dataset(A1, zero_center=zero_center, max_value=max_value)
        _scale_per_dataset(A2, zero_center=zero_center, max_value=max_value)

    avg1, g1_levels, g1_sizes = _group_pseudobulk(A1, groupby1)
    avg2, g2_levels, g2_sizes = _group_pseudobulk(A2, groupby2)

    if min_cells_per_group > 1:
        keep1 = [g for g in g1_levels if int(g1_sizes.get(g,0)) >= min_cells_per_group]
        keep2 = [g for g in g2_levels if int(g2_sizes.get(g,0)) >= min_cells_per_group]
        avg1 = avg1.loc[:, keep1]; g1_levels = keep1
        avg2 = avg2.loc[:, keep2]; g2_levels = keep2
        if verbose:
            print(f"Kept {len(g1_levels)} groups in adata1 and {len(g2_levels)} in adata2 after min_cells filter.")

    if scale and scale_mode == 'concat_pseudobulk':
        avg1, avg2 = _concat_pseudobulk_scale(avg1, avg2)

    pbar = None
    if verbose and method in {'kendall','distance_correlation','ccc'}:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(g1_levels)*len(g2_levels), position=0, leave=True)
        except Exception:
            pbar = None

    if method == 'cosine':
        sim_df = _cosine_vectorized(avg1, avg2)
    elif method == 'pearson':
        sim_df = _pearson_vectorized(avg1, avg2)
    elif method == 'spearman':
        sim_df = _spearman_vectorized(avg1, avg2)
    elif method == 'kendall':
        sim_df = _kendall_loop(avg1, avg2, pbar=pbar)
    elif method == 'ccc':
        sim_df = _ccc_loop(avg1, avg2, pbar=pbar)
    elif method == 'distance_correlation':
        sim_df = _dcorr_loop(avg1, avg2, pbar=pbar)

    if pbar is not None: pbar.close()
    sim_df = sim_df.reindex(index=g1_levels, columns=g2_levels)

    if return_matching:
        return sim_df, _hungarian_match(sim_df)
    return sim_df

# ---------------- public API: OO wrapper ----------------

try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

class CellStateAligner:
    """Stateful wrapper: fit once, then plot/assign."""

    def __init__(self):
        self.sim_df: Optional[pd.DataFrame] = None
        self.method: Optional[str] = None
        self.scale_mode: Optional[str] = None
        self.features: Optional[list] = None
        self.groups_ref: Optional[list] = None
        self.groups_query: Optional[list] = None
        self.group_sizes_ref: Optional[pd.Series] = None
        self.group_sizes_query: Optional[pd.Series] = None
        self.meta = {}

    def fit(self, adata1, adata2, groupby1: str, groupby2: str,
            method: Literal['pearson','spearman','kendall','cosine','distance_correlation','ccc']='cosine',
            *,
            features_use: Optional[Iterable[str]] = None,
            genes_blacklist: Optional[Iterable[str]] = None,
            highly_variable_only: bool = False,
            hv_key: str = 'highly_variable',
            layer: Optional[str] = None,
            scale: bool = True,
            scale_mode: Literal['per_dataset','concat_pseudobulk','none'] = 'concat_pseudobulk',
            zero_center: bool = False,
            max_value: Optional[float] = 6.0,
            min_cells_per_group: int = 1,
            verbose: bool = True):
        self.method = method
        self.scale_mode = scale_mode

        sim_df = cell_state_correlation(
            adata1, adata2, groupby1, groupby2, method=method,
            features_use=features_use, genes_blacklist=genes_blacklist,
            highly_variable_only=highly_variable_only, hv_key=hv_key,
            layer=layer, scale=scale, scale_mode=scale_mode,
            zero_center=zero_center, max_value=max_value,
            min_cells_per_group=min_cells_per_group, verbose=verbose,
            return_matching=False
        )
        self.sim_df = sim_df
        self.features = [f for f in adata1.var_names if f in sim_df.index]
        self.groups_ref = list(sim_df.index)
        self.groups_query = list(sim_df.columns)
        self.group_sizes_ref = adata1.obs[groupby1].value_counts().reindex(self.groups_ref).fillna(0).astype(int)
        self.group_sizes_query = adata2.obs[groupby2].value_counts().reindex(self.groups_query).fillna(0).astype(int)
        self.meta = dict(groupby1=groupby1, groupby2=groupby2, method=method,
                         scale_mode=scale_mode, layer=layer, hv_key=hv_key,
                         features_use_provided=(features_use is not None),
                         genes_blacklist_count=len(genes_blacklist) if genes_blacklist is not None else 0,
                         min_cells_per_group=min_cells_per_group)
        return self

    # -------- assignments --------

    @staticmethod
    def _normalize_rows(df: pd.DataFrame, normalize: Literal['none','row_z','row_minmax']):
        if normalize == 'none': return df
        A = df.values.astype(float).copy()
        if normalize == 'row_z':
            m = A.mean(axis=1, keepdims=True)
            s = A.std(axis=1, ddof=1, keepdims=True); s[s==0]=1
            A = (A - m) / s
        elif normalize == 'row_minmax':
            mn = A.min(axis=1, keepdims=True); mx = A.max(axis=1, keepdims=True)
            d = mx - mn; d[d==0]=1
            A = (A - mn) / d
        else:
            raise ValueError("normalize must be 'none','row_z','row_minmax'")
        return pd.DataFrame(A, index=df.index, columns=df.columns)

    def assign(self, strategy: Literal['argmax','hungarian']='argmax', *,
               normalize: Literal['none','row_z','row_minmax']='none',
               min_score: Optional[float]=None, min_margin: Optional[float]=None,
               top_k: int = 3, pred_col: Optional[str]=None, adata_query=None) -> pd.DataFrame:
        if self.sim_df is None: raise RuntimeError("Call .fit(...) first.")
        S = self._normalize_rows(self.sim_df, normalize=normalize)

        if strategy == 'argmax':
            pred = S.idxmax(axis=0)
            best = S.max(axis=0)
            if S.shape[0] >= 2:
                part_sorted = np.sort(S.values, axis=0)
                second = pd.Series(part_sorted[-2, :], index=S.columns)
            else:
                second = pd.Series(0.0, index=S.columns)
            margin = best - second
            df = pd.DataFrame({
                'query_group': S.columns,
                'pred_ref': pred.values,
                'score': best.values,
                'margin': margin.values,
                'method': self.method,
                'strategy': 'argmax',
            })
        elif strategy == 'hungarian':
            cost = -S.values
            r_ind, c_ind = linear_sum_assignment(cost)
            matched = pd.DataFrame({
                'query_group': S.columns[c_ind],
                'pred_ref': S.index[r_ind],
                'score': S.values[r_ind, c_ind],
                'method': self.method,
                'strategy': 'hungarian',
            }).set_index('query_group').reindex(S.columns)
            margins = []
            for q in S.columns:
                col = S[q]
                if pd.notna(matched.loc[q, 'pred_ref']):
                    best_val = matched.loc[q, 'score']; best_row = matched.loc[q, 'pred_ref']
                    second_best = col.drop(best_row).max() if len(col) > 1 else 0.0
                    margins.append(best_val - second_best)
                else:
                    margins.append(np.nan)
            matched['margin'] = margins
            df = matched.reset_index()
        else:
            raise ValueError("strategy must be 'argmax' or 'hungarian'")

        # top-k diagnostics
        K = min(top_k, S.shape[0])
        order = np.argsort(-S.values, axis=0)
        cols = {}
        for k in range(K):
            cols[f'top{k+1}_ref'] = S.index.values[order[k, :]]
            cols[f'top{k+1}_score'] = S.values[order[k, :], range(S.shape[1])]
        topk_df = pd.DataFrame(cols, index=S.columns).reset_index().rename(columns={'index':'query_group'})
        df = df.merge(topk_df, on='query_group', how='left')

        # thresholds
        assigned = df['pred_ref'].copy()
        if min_score is not None: assigned[df['score'] < float(min_score)] = 'unassigned'
        if min_margin is not None: assigned[df['margin'] < float(min_margin)] = 'unassigned'
        df['final_label'] = assigned

        if adata_query is not None and pred_col is not None:
            import warnings
            if 'unassigned' in set(df['final_label']): warnings.warn("Some predictions are 'unassigned'.")
            mapping = dict(zip(df['query_group'], df['final_label']))
            q_col = self.meta['groupby2']
            adata_query.obs[pred_col] = adata_query.obs[q_col].map(mapping).astype('category')
        return df

    # -------- plotting --------

    def plot_heatmap(self, figsize: Tuple[float,float]=(8,6), cmap: str='viridis',
                     vmin: Optional[float]=None, vmax: Optional[float]=None,
                     normalize: Literal['none','row_z','row_minmax']='none',
                     annotate: bool=False, annot_fmt: str=".2f",
                     title: Optional[str]=None, ax: Optional[plt.Axes]=None,
                     colorbar: bool=True):
        if self.sim_df is None: raise RuntimeError("Call .fit(...) first.")
        S = self._normalize_rows(self.sim_df, normalize=normalize)
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(S.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(S.shape[1])); ax.set_xticklabels(S.columns, rotation=90)
        ax.set_yticks(np.arange(S.shape[0])); ax.set_yticklabels(S.index)
        ax.set_title(title or f"{self.method} similarity ({normalize})")
        if colorbar: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if annotate:
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    ax.text(j, i, format(S.iat[i, j], annot_fmt), ha="center", va="center", fontsize=8)
        return ax

    def plot_clustermap(self, method: str='average', metric: str='correlation',
                        cmap: str='viridis', normalize: Literal['none','row_z','row_minmax']='none',
                        z_score: Optional[int]=None, standard_scale: Optional[int]=None, **kwargs):
        if self.sim_df is None: raise RuntimeError("Call .fit(...) first.")
        S = self._normalize_rows(self.sim_df, normalize=normalize)
        try:
            import seaborn as sns
        except Exception:
            import warnings
            warnings.warn("seaborn not installed; using plot_heatmap() instead.")
            return self.plot_heatmap()
        g = sns.clustermap(S, method=method, metric=metric, cmap=cmap,
                           z_score=z_score, standard_scale=standard_scale, **kwargs)
        g.ax_heatmap.set_title(f"{self.method} similarity clustered ({normalize})")
        return g
