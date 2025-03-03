import math
import argparse
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
from torchmetrics.regression import PearsonCorrCoef, SpearmanCorrCoef

from data_module import CodonDataModule
from checkpointing import PeriodicCheckpoint
from calm.sequence import CodonSequence
from calm.alphabet import Alphabet
from calm.model import ProteinBertRegressor

class PLProteinBertRegressor(pl.LightningModule):
    def __init__(self, model, args, checkpoint_path=None, lr=1e-4):
        super().__init__()
        self.model = model
        self.args = args
        self.loss_fn = nn.MSELoss()  # Loss for regression
        self.lr = lr
        self.pearsonr = PearsonCorrCoef()
        self.spearmanr = SpearmanCorrCoef()
        if checkpoint_path:
            self.model = self.load_pretrained_model(self.model, checkpoint_path)

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
        r = self.pearsonr(preds.squeeze(-1), labels)
        rho = self.spearmanr(preds.squeeze(-1), labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_pearsonr", r, prog_bar=True)
        self.log("val_spearmanr", rho, prog_bar=True)
        return loss
    
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
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--ffn_embed_dim", type=int, default=3072)
    parser.add_argument("--attention_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    '''
    parser.add_argument('--max_positions', type=int, default=1024)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine')
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_gradients', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=15000)
    parser.add_argument('--version', type=int, default=50)

    ProteinBertRegressor.add_args(parser)
    args = parser.parse_args()

    data_path = '/Users/clark04/toby/CDS-LM/data/finetuning/mrna_half-life.csv'
    sequence_column = 'CDS'
    target_column = 'y'
    name = 'saluki_stability'

    # Initialize model
    alphabet = Alphabet.from_architecture('CodonModel')

    datamodule = CodonDataModule(args, alphabet, data_path, args.batch_size,
                                 fine_tune=True, target_column = target_column,
                                 sequence_column = sequence_column)

    model = ProteinBertRegressor(args, alphabet)
    pl_model = PLProteinBertRegressor(model, args, checkpoint_path='calm_weights.pkl')

    checkpoint_callback = ModelCheckpoint(
        dirpath="assets/",
        filename=name,
        save_top_k=1,               
        monitor="val_loss",        
        mode="min"                  
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,    
        mode="min"      
    )

    fast_dev_run = False # set to True to run a single batch for testing.
    
    logger = TensorBoardLogger(save_dir = '.', version = args.version)
    
    trainer = pl.Trainer(max_epochs=10, precision="16-mixed", accelerator='gpu', accumulate_grad_batches=args.accumulate_gradients,
                         check_val_every_n_epoch=1, fast_dev_run = fast_dev_run, gradient_clip_val=1.0, logger = logger,
                         log_every_n_steps = 1,
                         callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step'), early_stop_callback])  
    trainer.fit(pl_model, datamodule=datamodule)