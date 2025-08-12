"""
Runs a suite of probing tasks (fitting simple ML models to zero-shot embeddings) with a given model
"""

import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
from calm.pretrained import CaLM
import torch
from cdslm.utils import (
    convert_to_codons,
    load_model_and_tokenizer,
    get_sequence_embeddings,
    process_and_tokenize,
)
from torch.utils.data import DataLoader
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, PredefinedSplit
from sklearn.decomposition import PCA
from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
from typing import Optional, Tuple, Iterator

torch.set_float32_matmul_precision('high')

def trunc_seqs(seq):
    if len(seq) % 3 == 1:
        seq = seq[:-1]
    elif len(seq) % 3 == 2:
        seq = seq[:-2]
    return seq

def get_calm_embeddings(model, data):
    data = [seq[:3072] if len(seq) > 3072 else seq for seq in data]
    for seq in data:
        seq = trunc_seqs(seq)
    embs = model.embed_sequences(data)
    return embs.cpu().numpy()

def get_split_iterator(X: np.array, k: int = 5, predefined_split_idxs: Optional[pd.Series] = None, random_state: int = 42) -> Iterator:
    """
    Returns an iterator for cross validation splitting
    """
    if predefined_split_idxs is not None:
        split = PredefinedSplit(predefined_split_idxs)
        iterator = enumerate(split.split())
    else:
        indices = np.arange(len(X))
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        iterator = enumerate(kf.split(indices))
    return iterator

def fit_regression(
    X: np.array,
    y: np.array,
    task: str,
    n_splits: int = 5,
    random_state: int = 42,
    elastic_alpha: float = 0.001,
    predefined_split_idxs: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Splits dataset and fits an elastic net to the PCA transformed embeddings.
    The split will be a random K-fold split (n_splits), unless predefined_split_idxs is specified.
    """
    R_list = []
    rho_list = []
    all_preds = []
    all_folds = []
    iterator = get_split_iterator(X, k = n_splits, predefined_split_idxs=predefined_split_idxs, random_state=random_state)
    for fold, (train_index, test_index) in iterator:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        enet = ElasticNet(alpha=elastic_alpha, random_state=random_state)
        enet.fit(X_train, y_train)
        preds = enet.predict(X_test)
        all_preds.extend(preds)
        r, p_val = pearsonr(y_test, preds)
        R_list.append(r)
        rho, p_val = spearmanr(y_test, preds)
        rho_list.append(rho)
        all_folds.extend([fold] * len(preds))

    res_df = pd.DataFrame({"pearson_r": R_list, "r2": [*map(lambda x: x**2, R_list)], "spearman_r": rho_list})
    pred_df = pd.DataFrame({'prediction': all_preds, 'target': y, "fold": all_folds})
    pred_df['task'] = task
    print(f"Performance in {task} task:")
    print(f"Spearman rho: {res_df['spearman_r'].mean():.4f}")
    print(f"Pearson R\u00b2: {res_df['r2'].mean():.4f}")

    return res_df, pred_df
    

def benchmark_task(
    model,
    tokenizer,
    data_path: str,
    task: str,
    sequence_col: str,
    target_col: str,
    emb_option: str = "mean",
    emb_layer: int = -1,
    embs_dir: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
    elastic_alpha: float = 0.001,
    emb_batch_size: int = 32,
    max_length: int = 1280,
    predefined_split: bool = False,
) -> pd.DataFrame:
    """
    Runs full probing for a given dataset.
    """
    print(f"Running task {task}")
    task_data = pd.read_csv(data_path, index_col=0)
    
    if predefined_split:
        predefined_split_idxs = task_data['split']
    else:
        predefined_split_idxs = None
    
    # Read embs csv if it exists
    embs_path = Path(f'{embs_dir}/{task}.csv')
    if embs_path.exists():
        embs = pd.read_csv(embs_path, index_col=0)
    else:
        embs = get_calm_embeddings(model, task_data[sequence_col])
        if embs_dir:
            # Save embeddings if embs_dir specified
            pd.DataFrame(embs).to_csv(f"{embs_dir}/{task}.csv")

    pca = PCA(n_components=320)
    X = pca.fit_transform(embs)
    y = np.array(task_data[target_col].values)

    res_df, pred_df = fit_regression(X, y, task, n_splits, random_state, elastic_alpha, predefined_split_idxs)

    return res_df, pred_df


def run_benchmarks(
    model,
    tokenizer,
    data_dir: str,
    res_dir: str,
    save_preds: bool = True,
    emb_option: str = "mean",
    emb_layer: int = -1,
    embs_dir: str = None,
    n_splits: int = 5,
    random_state: int = 42,
    elastic_alpha: float = 0.001,
    emb_batch_size: int = 32,
    max_length: int = 1280
) -> None:
    """
    Runs the benchmarking tasks - edit this to add/remove tasks as necessary
    """
    
    path = Path(res_dir)
    path.mkdir(parents=True, exist_ok=True)

    if embs_dir:
        path = Path(embs_dir)
        path.mkdir(parents=True, exist_ok=True)

    pred_list = []
    # Melting point
    data_path = f"{data_dir}/melting_temperature/melting_temperature.csv"
    meltome_res, preds = benchmark_task(
        model,
        tokenizer,
        data_path,
        "melting_temperature",
        "sequence",
        "melting_temperature",
        emb_option,
        emb_layer,
        embs_dir,
        n_splits,
        random_state,
        elastic_alpha,
        emb_batch_size,
        max_length,
    )
    meltome_res.to_csv(f"{res_dir}/melting_temperature_res.csv")
    pred_list.append(preds)
    
    # Transcript abundance
    spec_list = [
        "athaliana",
        "dmelanogaster",
        "ecoli",
        "hsapiens",
        "hvolcanii",
        "ppastoris",
        "scerevisiae",
    ]
    data_path = f"{data_dir}/transcript_abundance"
    res_list = []
    for species in spec_list:
        species_path = f"{data_path}/{species}.csv"
        spec_res, preds = benchmark_task(
            model,
            tokenizer,
            species_path,
            f"transcript_abundance_{species}",
            "sequence",
            "logtpm",
            emb_option,
            emb_layer,
            embs_dir,
            n_splits,
            random_state,
            elastic_alpha,
            emb_batch_size,
            max_length,
        )
        spec_res["species"] = species
        res_list.append(spec_res)
        pred_list.append(preds)
    pd.concat(res_list).to_csv(f"{res_dir}/transcript_abundance.csv")

    cell_lines = {
        'human': ['hlf', 'htert1'],
        'yeast': ['glucose', 'ethanol'],
    }
    data_path = f"{data_dir}/cell_size_protein_abundance"
    res_list = []
    for species, conditions in cell_lines.items():
        for condition in conditions:
            cond_path = f"{data_path}/{species}_{condition}_size_abundance.csv"
            cell_res, preds = benchmark_task(
                model,
                tokenizer,
                cond_path,
                f"cell_size_protein_abundance_{species}_{condition}",
                "Sequence",
                "Mean Protein Slope",
                emb_option,
                emb_layer,
                embs_dir,
                n_splits,
                random_state,
                elastic_alpha,
                emb_batch_size,
                max_length,
            )
            cell_res["cell_line"] = condition
            res_list.append(cell_res)
            pred_list.append(preds)
    pd.concat(res_list).to_csv(f"{res_dir}/cell_size_protein_abundance.csv")

    # Saluki transcript stability
    data_path = f"{data_dir}/saluki/human_sss_reprocessed.csv"
    stability_res, preds = benchmark_task(
        model,
        tokenizer,
        data_path,
        "saluki_stability",
        "CDS",
        "y",
        emb_option,
        emb_layer,
        embs_dir,
        n_splits,
        random_state,
        elastic_alpha,
        emb_batch_size,
        max_length,
        predefined_split = True,
    )
    stability_res.to_csv(f"{res_dir}/stability_res.csv")
    pred_list.append(preds)

    data_path = f"{data_dir}/saluki/mouse_sss_reprocessed.csv"
    stability_res, preds = benchmark_task(
        model,
        tokenizer,
        data_path,
        "saluki_mouse_stability",
        "CDS",
        "y",
        emb_option,
        emb_layer,
        embs_dir,
        n_splits,
        random_state,
        elastic_alpha,
        emb_batch_size,
        max_length,
        predefined_split = True,
    )
    stability_res.to_csv(f"{res_dir}/saluki_mouse.csv")
    pred_list.append(preds)
    
    data_path = f"{data_dir}/translation_rate.csv"
    translation_rate_res, preds = benchmark_task(
        model,
        tokenizer,
        data_path,
        "translation_rate",
        "CDS",
        "y",
        emb_option,
        emb_layer,
        embs_dir,
        n_splits,
        random_state,
        elastic_alpha,
        emb_batch_size,
        max_length,
        predefined_split=True,
    )
    translation_rate_res.to_csv(f'{res_dir}/translation_rate.csv')
    pred_list.append(preds)

    data_path = f"{data_dir}/icodon/icodon_zebrafish.csv"
    translation_rate_res, preds = benchmark_task(
        model,
        tokenizer,
        data_path,
        "icodon_zebrafish",
        "CDS",
        "y",
        emb_option,
        emb_layer,
        embs_dir,
        n_splits,
        random_state,
        elastic_alpha,
        emb_batch_size,
        max_length,
        predefined_split=True,
    )
    translation_rate_res.to_csv(f'{res_dir}/icodon_zebrafish.csv')
    pred_list.append(preds)

    if save_preds:
        pd.concat(pred_list).to_csv(f'{res_dir}/predictions.csv')
    
    return None

subset_size = "100perc_orths"
model_type = "albert"
size = "large"
save_preds = True
emb_option = "mean"
seed = 42
n_splits = 5
emb_batch_size = 32
emb_layer = -1
elastic_alpha = 0.001

data_base_path = "/home/jovyan/workspace/CDS-LM/data/finetuning"

'''
model_path = f"./assets/saved_models/subset_{subset_size}/{model_type}_{size}_mlm"
res_dir = f"./results/subset_{subset_size}/{model_type}_{size}_mlm/layer_{emb_layer}"
embs_dir = f"./embs/subset_{subset_size}/{model_type}_{size}_mlm/layer_{emb_layer}"
'''
'''
model_path = "./assets/saved_models/CDSLM_roformer/CDSLM_roformer_orths"
res_dir = f"./results/CDS-LM-roformer/CDS-LM-roformer-orths/layer_{emb_layer}"
embs_dir = f"./embs/CDS-LM-roformer/CDS-LM-roformer-orths/layer_{emb_layer}"
'''

name = "CaLM"
model_path = "/home/jovyan/workspace/CaLM/calm/calm_weights/calm_weights.ckpt"
# model_path = '/home/jovyan/shared/toby/cds-lm/assets/checkpoints/modern-BERT/modern-BERT-large-eukaryotes-sss/checkpoint-247050'
res_dir = f"./results/{name}/layer_{emb_layer}"
embs_dir = f"/home/jovyan/shared/toby/cds-lm/embs/CaLM/{name}/layer_{emb_layer}"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")


model = CaLM(weights_file=model_path, device='cuda')
model.model.eval()

run_benchmarks(
    model,
    tokenizer,
    data_base_path,
    res_dir,
    save_preds,
    emb_option=emb_option,
    emb_layer=emb_layer,
    embs_dir=embs_dir,
    n_splits=n_splits,
    random_state=seed,
    elastic_alpha=elastic_alpha,
    emb_batch_size=emb_batch_size,
    max_length=model.config.max_position_embeddings
)
