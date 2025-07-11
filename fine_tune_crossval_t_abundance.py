import math
import argparse
import pickle
import pandas as pd
import numpy as np
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef
from sklearn.model_selection import PredefinedSplit, KFold, train_test_split
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from scipy.stats import pearsonr, spearmanr
from transformers import BertConfig

from data_module import CodonDataModule
from checkpointing import PeriodicCheckpoint
from calm.sequence import CodonSequence
from calm.alphabet import Alphabet
from calm.model import ProteinBertRegressor


class TrainValTestSplit:
    """
    Adapter to extend cross-validators for train/val/test splits.
    Mimics scikit-learn's cross-validator interface.
    If test_fold is specified, use a predefined split like PredefinedSplit in scikit-learn
    """
    
    def __init__(self, n_folds=5, test_fold=None):
        """
        Parameters:
        test_fold: array-like, fold assignments (0-9 for 10-fold CV)
        """
        if test_fold:
            self.test_fold = np.array(test_fold)
            self.unique_folds = np.unique(self.test_fold)
            self.n_folds=None
        elif n_folds:
            self.n_folds=n_folds
        else:
            raise ValueError("Need to specify one of n_folds or a predefined test fold")
    
    def split(self, data=None):
        """
        Generate train/val/test splits.
        Returns (train_idx, val_idx, test_idx) tuples.
        """
        if self.n_folds:
            if data is None:
                raise ValueError("Need to provide data to split")
                
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            for train_idx, val_test_idx in kf.split(data):
                test_idx, val_idx = train_test_split(val_test_idx, test_size=0.5, shuffle=True, random_state=42)
                yield train_idx, val_idx, test_idx
        else:        
            for test_fold_val in self.unique_folds:
                val_fold_val = (test_fold_val + 1) % len(self.unique_folds)
                
                test_idx = np.where(self.test_fold == test_fold_val)[0]
                val_idx = np.where(self.test_fold == val_fold_val)[0]
                train_idx = np.where(~np.isin(self.test_fold, [test_fold_val, val_fold_val]))[0]
                
                yield train_idx, val_idx, test_idx
                

class PEFTModelWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.config = BertConfig()

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        This wrapper translates Hugging Face-style inputs into what your model expects.
        - input_ids → tokens
        - kwargs like `repr_layers`, `need_head_weights` can be forwarded if needed
        """
        return self.base_model(tokens=input_ids)
        

class PLProteinBertRegressor(pl.LightningModule):
    def __init__(self, model, args, test_out_file=None, peft_config=None, checkpoint_path=None, lr=1e-4):
        super().__init__()
        self.model = model
        self.args = args
        self.loss_fn = nn.HuberLoss()  # Loss for regression
        self.lr = args.learning_rate if args else lr
        self.test_preds = []
        self.test_labels = []
        self.test_out_file = test_out_file
        
        if checkpoint_path:
            self.model = self.load_pretrained_model(self.model, checkpoint_path)

        if peft_config:
            self.model = get_peft_model(PEFTModelWrapper(model), peft_config)

    def load_pretrained_model(self, model, state_dict_path):
        """Load the pretrained state dict and replace classification head."""
        # Load the .pkl state dict
        with open(state_dict_path, "rb") as f:
            state_dict = pickle.load(f)

        # Load weights into model
        self.model.load_state_dict(state_dict, strict=False)  # strict=False allows head replacement
        print("Loaded pretrained weights (excluding classifier head).")
        
        self.model.regressor = nn.Sequential(
            nn.Linear(model.args.embed_dim, model.args.embed_dim // 2),
            nn.GELU(),
            nn.Linear(model.args.embed_dim // 2, 1)  # Regression output
        )
        return model    
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        data, labels = train_batch['input'].to(), train_batch['labels'].float()
        preds = self.model(data)['logits']
        loss = self.loss_fn(preds.squeeze(-1), labels)
        
        if batch_idx % self.args.accumulate_gradients == 0:
            self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch['input'].to(), val_batch['labels'].float()
        preds = self.model(data)['logits']
        loss = self.loss_fn(preds.squeeze(-1), labels)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch["input"], batch["labels"]
        outputs = self.model(inputs)
        
        preds = outputs['logits'].squeeze(-1)

        # Detach and store
        self.test_preds.append(preds.detach().cpu())
        self.test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        preds = preds = torch.cat(self.test_preds).to(torch.float32).numpy()
        labels = torch.cat(self.test_labels).to(torch.float32).numpy()

        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]

        self.log("test_pearson", pearson_corr, prog_bar=True)
        self.log("test_spearman", spearman_corr, prog_bar=True)

        df = pd.DataFrame([{
            "pearsonr": pearson_corr,
            "spearmanr": spearman_corr
        }])
        if self.test_out_file:
            df.to_csv(self.test_out_file, index=False)


        # Optional: reset state
        self.test_preds.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

            if self.args.lr_scheduler == 'none':
                return optimizer
            elif self.args.lr_scheduler == 'warmup_sqrt':
                def schedule(global_step):
                    if global_step < self.args.warmup_steps:
                        return (global_step + 1) / self.args.warmup_steps
                    else:
                        return np.sqrt(self.args.warmup_steps / global_step)
            elif self.args.lr_scheduler == 'warmup_cosine':
                def schedule(global_step):
                    if global_step < self.args.warmup_steps:
                        return (global_step + 1) / self.args.warmup_steps
                    else:
                        progress = (global_step - self.args.warmup_steps) / self.args.num_steps
                        return max(0., .5 * (1. + math.cos(math.pi * progress)))
            else:
                raise ValueError('Unrecognized learning rate scheduler')

            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [scheduler]




if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--ffn_embed_dim", type=int, default=3072)
    parser.add_argument("--attention_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    '''
    parser.add_argument('--max_positions', type=int, default=1024)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--accumulate_gradients', type=int, default=1)
    parser.add_argument('--version', type=int, default=50)

    ProteinBertRegressor.add_args(parser)
    args = parser.parse_args()

    data_path = '../CDS-LM/data/finetuning/transcript_abundance'
    task = "transcript_abundance"
    sequence_column = "sequence"
    target_column = "logtpm"
    name = 'CaLM'
    alphabet = Alphabet.from_architecture('CodonModel')

    res_dir = f"/home/jovyan/shared/toby/cds-lm/results/finetuning/{task}/"
    overall_df_list = []
    for species in ['athaliana', 'dmelanogaster', 'hsapiens', 'ppastoris', 'scerevisiae']:
        
        file_name = f"{species}.csv"
        data = pd.read_csv(f"{data_path}/{file_name}", index_col=0)
        iterator = enumerate(TrainValTestSplit(n_folds=5).split(data))        
    
        df_list = []        

        checkpoint_dir = f"/home/jovyan/shared/toby/cds-lm/assets/checkpoints/finetuning/{name}/{task}_{species}"
        log_dir = f"./logs/finetuning/{task}/{name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for fold, idxs in iterator:
    
            datamodule = CodonDataModule(args, alphabet, f"{data_path}/{file_name}", args.batch_size,
                                         fine_tune=True, sequence_column = sequence_column,
                                         target_column = target_column, split_idxs = idxs)
    
            model = ProteinBertRegressor(args, alphabet)
    
            peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS, # TOKEN_CLS
            r=8,
            lora_alpha=16,  # 2x sqrt(hidden size)
            lora_dropout=0.5,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            modules_to_save=["regressor"],
            use_rslora=True,
        )
    
            out_file = f'fold_{fold+1}.csv'
            pl_model = PLProteinBertRegressor(model, args, out_file, peft_config, checkpoint_path='/home/jovyan/shared/toby/cds-lm/assets/saved_models/calm_weights.pkl')
    
    
            fast_dev_run = False # set to True to run a single batch for testing.
    
            logger = TensorBoardLogger(save_dir = './lightning_logs/crossval/', version = fold)
            
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f'fold_{fold}+1',
                save_top_k=1,               
                monitor="val_loss",        
                mode="min"                  
            )
            
            early_stop_callback = EarlyStopping(
                monitor="val_loss",   # Metric to monitor
                patience=3,           # Number of epochs with no improvement before stopping
                verbose=True,         # Prints logs when stopping
                mode="min"
            )
        
            trainer = pl.Trainer(max_epochs=15, accelerator='gpu', precision="bf16-mixed",
                                 val_check_interval=0.5, fast_dev_run = fast_dev_run, logger = logger,
                                 log_every_n_steps = 1,
                                 callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), early_stop_callback])  
            trainer.fit(pl_model, datamodule=datamodule)
            trainer.test(pl_model, dataloaders=datamodule.test_dataloader())
            df_list.append(pd.read_csv(out_file))
    
        species_res = pd.concat(df_list)
        species_res['Species'] = species
        overall_df_list.append(species_res)
        print(f'Species: {species}')
        print(f'R: {species_res['pearsonr'].mean()}')
        print(f'Rho: {species_res['spearmanr'].mean()}')

    pd.concat(overall_df_list).to_csv(f'{res_dir}/{name}_test.csv')