"""PyTorch Lightning module for standard training."""

import math
import argparse

import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import CodonDataModule, CodonDataModuleHF
from checkpointing import PeriodicCheckpoint
from calm.sequence import CodonSequence
from calm.alphabet import Alphabet
from calm.model import ProteinBertModel


class CodonModel(pl.LightningModule):
    """PyTorch Lightning module for standard training."""

    def __init__(self, args, alphabet):
        super().__init__()
        self.args = args
        self.alphabet = alphabet
        self.model = ProteinBertModel(args, alphabet)

        def init_weights(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                torch.nn.init.normal_(module.weight, std=.02)

            if isinstance(module, (torch.nn.Linear)):
                module.bias.data.fill_(0)
        self.model.apply(init_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay)

        if self.args.lr_scheduler == 'none':
            return optimizer
        elif self.args.lr_scheduler == 'warmup_sqrt':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    return np.sqrt(self.args.warmup_steps / global_step)
        elif self.args.lr_scheduler == 'warmup_cosine':
            def schedule(global_step):
                if global_step < self.args.warmup_steps:
                    return (global_step+1) / self.args.warmup_steps
                else:
                    progress = (global_step - self.args.warmup_steps) / self.args.num_steps
                    return max(0., .5 * (1. + math.cos(math.pi * progress)))
        else:
            raise ValueError('Unrecognised learning rate scheduler')

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, schedule),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        data, labels = \
            train_batch['input'].to(), \
            train_batch['labels'].to(dtype=torch.int64)

        output = self.model(data)
        likelihoods = output['logits']
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )

        if batch_idx % self.args.accumulate_gradients == 0:
            self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, labels = \
            val_batch['input'].to(), \
            val_batch['labels'].to(dtype=torch.int64)

        output = self.model(data)
        likelihoods = output['logits']
        loss = self.loss_fn(
            likelihoods.view(-1, len(self.alphabet.all_toks)),
            labels.view(-1)
        )
        self.log('val_loss', loss)
        return loss


if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_positions', type=int, default=1280)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate_gradients', type=int, default=8) # With 32 GPUs 8 * 8 * 32 gives an effective batch size of 2048 - c.f. 1840 for CaLM.
    parser.add_argument('--mask_proportion', type=float, default=.25)
    parser.add_argument('--leave_percent', type=float, default=.1)
    parser.add_argument('--mask_percent', type=float, default=.8)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='warmup_cosine')
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--num_steps', type=int, default=121000)
    ProteinBertModel.add_args(parser)
    args = parser.parse_args()

    # data
    alphabet = Alphabet.from_architecture('CodonModel')
    
    datamodule = CodonDataModuleHF(args, alphabet, '/Users/clark04/toby/CDS-LM/processed_data/subset_100perc_orths/clean/',
                                   args.batch_size, train_path='train_dataset_clean', val_path='eval_dataset_clean')
    
    # model
    model = CodonModel(args, alphabet)
    

    
    # training
    name = 'calm-orths'
    
    fast_dev_run = False
    
    best_checkpoint_callback = ModelCheckpoint(
        dirpath="assets/",
        filename=name,
        save_top_k=1,               
        monitor="val_loss",        
        mode="min"                  
    )
    
    epoch_checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints/CaLM_orths",
        filename="epoch-{epoch:02d}",
        save_top_k=-1,
        every_n_epochs=1,
    )

    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True,    
        mode="min"      
    )
    
    logger = TensorBoardLogger(save_dir = './lightning_logs/orths/')
    
    trainer = pl.Trainer(accelerator='gpu', precision="16-mixed", devices=8, num_nodes=4,
        max_steps=args.num_steps, logger=logger, log_every_n_steps=1, gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_gradients, strategy='ddp',
        check_val_every_n_epoch=1, fast_dev_run = fast_dev_run, max_epochs=20,
        callbacks=[best_checkpoint_callback, LearningRateMonitor(logging_interval='step'), early_stop_callback, epoch_checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)
