import pandas as pd
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import numpy as np
from cell_eval import MetricsEvaluator

import logging
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout


ALL_METRICS  = [
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    "pearson_edistance",
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
    "overlap_at_200",
    "overlap_at_500",
    "precision_at_N",
    "precision_at_50",
    "precision_at_100",
    "precision_at_200",
    "precision_at_500",
    "de_spearman_sig",
    "de_direction_match",
    "de_spearman_lfc_sig",
    "de_sig_genes_recall",
    "de_nsig_counts",
    "pr_auc",
    "roc_auc",
    "clustering_agreement"
]
METRICS_TO_TRACK = [
    'overlap_at_N', 
    'overlap_at_50', 
    'de_direction_match', 
    'pr_auc', 
    'pearson_delta', 
    'mae', 
    'discrimination_score_l1'
]


@contextmanager
def suppress_all_output():
    """Gags both standard output and the logging module."""
    # 1. Silence Logging
    logging.getLogger('cell_eval').setLevel(logging.ERROR)
    logging.getLogger('pdex').setLevel(logging.ERROR)
    
    # 2. Silence C-level streams (tqdm usually writes to stderr)
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield

def filter_pred_by_true_combinations(adata_true, adata_pred, context_col='celltype', pert_col='genotype', ctrl_value='WT', min_cells=30):
    """
    Filters adata_true based on experimental viability, then filters adata_pred 
    to match the resulting valid (context, pert) combinations.
    """
    
    # --- STEP 1: Filter adata_true for viable perturbations ---
    # Count cells for every combination
    counts = adata_true.obs.groupby([context_col, pert_col]).size().reset_index(name='count')
    
    # Requirement: Non-control perturbations must have >= min_cells
    # Note: We usually keep the control regardless of its count to allow for comparisons, 
    # but the logic below ensures the context is dropped if ONLY the control remains.
    valid_perts = counts[
        (counts[pert_col] == ctrl_value) | (counts['count'] >= min_cells)
    ]
    
    # --- STEP 2: Filter adata_true for viable contexts ---
    # A context is valid ONLY if it contains the ctrl_value AND at least one other perturbation
    def has_experiment_structure(group):
        has_ctrl = ctrl_value in group[pert_col].values
        has_treated = any(group[pert_col] != ctrl_value)
        return has_ctrl and has_treated

    valid_contexts = valid_perts.groupby(context_col).filter(has_experiment_structure)
    
    # Apply these filters to adata_true
    # Create a helper key to match back to our valid list
    def get_key(df): return df[context_col].astype(str) + "_" + df[pert_col].astype(str)
    
    valid_keys_set = set(get_key(valid_contexts))
    
    true_mask = get_key(adata_true.obs).isin(valid_keys_set)
    adata_true_filtered = adata_true[true_mask].copy()
    
    # --- STEP 3: Match adata_pred to the cleaned adata_true ---
    true_combos = set(get_key(adata_true_filtered.obs).unique())
    pred_combos_series = get_key(adata_pred.obs)
    
    pred_mask = pred_combos_series.isin(true_combos)
    adata_pred_filtered = adata_pred[pred_mask].copy()

    # Logging
    #print(f"Original true contexts: {adata_true.obs[context_col].nunique()}")
    #print(f"Filtered true contexts: {adata_true_filtered.obs[context_col].nunique()}")
    #print(f"Removed {adata_pred.n_obs - adata_pred_filtered.n_obs} cells from adata_pred.")
    
    return adata_true_filtered, adata_pred_filtered




def create_perturb_anndata_pair_for_cell_eval(adata_true, 
                                              adata_pred, 
                                              pert_col = 'genotype', 
                                              context_col = 'celltype', 
                                              downsample = False, 
                                              sample = False,
                                              top_k = None,
                                              ctrl_value = 'WT',
                                              min_eval_cells = 30):
    adata_true_ctrl_expr = adata_true[adata_true.obs[pert_col] == 'WT'].X.toarray()
    ctrl_expr_context = adata_true[adata_true.obs[pert_col] == 'WT',:].obs[context_col].values

    #new_sf = 1/(np.sum(adata_pred.obsm['mvc_expr'], 1)/np.median(np.sum(adata_pred.obsm['mvc_expr'], 1))).reshape(-1,1)
    #adata_true_ctrl_expr = np.log(adata_pred.obsm['mvc_expr']*new_sf+1)
    #ctrl_expr_context  = adata_pred.obs[context_col].values

    pert_index = np.where(adata_true.obs.genotype.isin(adata_pred.obs.genotype_next.unique()))[0]
    ctrl_inds = np.where(adata_true.obs.genotype == 'WT')[0]
    pert_true_adata = adata_true[np.concatenate([pert_index, ctrl_inds]),:].copy()
    cell_names = [f"Cell_{i}" for i in range(adata_true_ctrl_expr.shape[0])]
    
    obs_A = pd.DataFrame({
        context_col: ctrl_expr_context,
        pert_col: 'WT'
    }, index=cell_names)
    adata_A = ad.AnnData(X=adata_true_ctrl_expr, obs=obs_A)
    adata_A.var_names = adata_true.var.index
    if sample:
        mask = (np.random.rand(*adata_pred.obsm['mvc_next_expr_zero'].shape) < adata_pred.obsm['mvc_next_expr_zero']).astype(int)
    else:
        mask = 1.0
    pert_pred_expr = adata_pred.obsm['mvc_next_expr']*mask

    perturbation_label = adata_pred.obs.genotype_next.values
    pred_ctrl_expr_context = adata_pred.obs[context_col].values
    # Step B: Create the object for Matrix B (The 'Model' data)
    # We use the same vector C, but map vector D to genotype
    cell_names = [f"Cell_{i}" for i in range(pert_pred_expr.shape[0])]

    obs_B = pd.DataFrame({
        context_col: pred_ctrl_expr_context,
        pert_col: perturbation_label
    }, index=cell_names)

    
    #pert_pred_expr = log_discretize_np(pert_pred_expr)
    pert_pred_expr[pert_pred_expr < 0 ] = 0

    adata_B = ad.AnnData(X=pert_pred_expr, obs=obs_B)
    adata_B.var_names = adata_true.var.index

    # Step C: Concatenate them into a single object
    # index_unique='-' ensures cell IDs remain distinct (e.g., Cell_0-raw vs Cell_0-model)
    pert_pred_adata = ad.concat(
        {'raw': adata_A, 'model': adata_B}, 
        label='data_source', 
        index_unique='-'
    )

    # final alignment between pred and real to filter out predictions that can't be validated due to celltype_genotype combinations
    pert_true_adata, pert_pred_adata = filter_pred_by_true_combinations(pert_true_adata, 
                                                       pert_pred_adata,
                                                       context_col=context_col,
                                                       pert_col=pert_col,
                                                       ctrl_value = ctrl_value,
                                                       min_cells = min_eval_cells
                                                       )
    
    if top_k is not None:
        assert type(top_k) == int, 'top_k is integer or None'
        if isinstance(adata_true.X, np.ndarray):
            means = adata_true.X.mean(axis=0)
        else:
            # Use .A1 to flatten the matrix result into a 1D array
            means = np.array(adata_true.X.mean(axis=0)).flatten()

        # Create a Series and sort
        mean_series = pd.Series(means, index=adata_true.var_names)
        top_genes = mean_series.sort_values(ascending=False).head(top_k)
        return  pert_true_adata[:,top_genes.index].copy(), pert_pred_adata [:,top_genes.index].copy()
    return pert_true_adata, pert_pred_adata 


def run_cell_eval(adata_real, adata_pert, outdir, context_col = 'celltype', pert_col = 'genotype', control_pert = 'WT', min_eval_cells = 30):
    results = {}
    real, pred = create_perturb_anndata_pair_for_cell_eval(adata_real, 
                                                           adata_pert, 
                                                           context_col=context_col,
                                                           pert_col=pert_col,
                                                           min_eval_cells = min_eval_cells)

    for c in real.obs[context_col].unique():
        pred_sub = pred[pred.obs[context_col] == c,].copy()
        real_sub = real[real.obs[context_col] == c,].copy()
        if type(c) == int:
            c = 'celltype_'+str(c)
        with suppress_all_output():
            evaluator = MetricsEvaluator(
                adata_pred=pred_sub,
                adata_real=real_sub,
                control_pert=control_pert,
                pert_col=pert_col,
                outdir = outdir,
                num_threads=8,
                prefix = c,
            )
            results[c] = evaluator.compute(skip_metrics = [m for m in ALL_METRICS if m not in METRICS_TO_TRACK])
    return results

def cell_eval_to_wandb(true, pred, outdir, epoch, key = 'test', min_eval_cells =30, distribution = 'nb', context_col = 'celltype', pert_col = 'genotype', ctrl = 'WT'):
    if distribution in ['nb', 'zinb', 'hnb', 'pois', 'zipois']:
        new_sf = 1/(np.sum(pred.obsm['mvc_next_expr'], 1)/np.median(np.sum(pred.obsm['mvc_next_expr'], 1))).reshape(-1,1)
        pred.obsm['mvc_next_expr'] = np.log(pred.obsm['mvc_next_expr']*new_sf + 1)

    results = run_cell_eval(
        true, 
        pred, 
        outdir / f'cellEval_{key}_e{epoch}',
        context_col = context_col, 
        pert_col = pert_col, 
        control_pert = ctrl,
        min_eval_cells = min_eval_cells
    )
    k_l = {m:[]for m in METRICS_TO_TRACK}
    for k in results:
        for m in METRICS_TO_TRACK:
            k_l[m].append(results[k][0][m])
    
    k_l = {f'{key}/'+m:np.round(np.concatenate(k_l[m]).mean(),4) for m in METRICS_TO_TRACK}
    k_l['epoch'] = epoch
    return {'metrics':k_l, 'images':{}, 'eval_dict_key': key, 'epoch': epoch}