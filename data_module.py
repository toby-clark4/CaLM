"""Data modules for PyTorch Lightning."""

from argparse import Namespace
from typing import Optional, Tuple

import torch
from torch.utils.data import Subset
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split

from calm.alphabet import Alphabet
from calm.dataset import SequenceDataset, SequenceDatasetFromHF, SequenceClassificationDataset
from calm.ft_pipeline import (
    FTPipeline,
    FTDataCollator,
    FTDataTrimmer,
    FTDataPadder,
    FTDataPreprocessor,
)
from calm.pipeline import (
    Pipeline,
    DataCollator,
    DataTrimmer,
    DataPadder,
    DataPreprocessor,
)


class CodonDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for manipulating FASTA files (pretraining) or csv files (fine-tuning)
    containing sequences of codons."""

    def __init__(self, args: Namespace, alphabet: Alphabet, data_path: str,
            batch_size: int, random_seed: int = 42, test_size : float = 0.01,
            fine_tune: bool = False, sequence_column: str = 'sequence',
            target_column: Optional[str] = None, split_idxs: Optional[Tuple[np.array]] = None):
        super().__init__()
        self.data_path = data_path
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.fine_tune = fine_tune
        if self.fine_tune:
             self.pipeline = FTPipeline([
                FTDataCollator(args, alphabet),
                FTDataTrimmer(args, alphabet),
                FTDataPadder(args, alphabet),
                FTDataPreprocessor(args, alphabet)
            ])
        else:
            self.pipeline =  Pipeline([
                DataCollator(args, alphabet),
                DataTrimmer(args, alphabet),
                DataPadder(args, alphabet),
                DataPreprocessor(args, alphabet)
            ])

        self.train_data = None
        self.val_data = None
        self.target_column = target_column
        self.sequence_column = sequence_column
        self.split_idxs = split_idxs

    def setup(self, stage: Optional[str] = None):
        if not self.fine_tune:
            dataset = SequenceDataset(self.data_path, codon_sequence=True)
        else:
            if self.target_column:
                dataset = SequenceClassificationDataset(self.data_path, target_column=self.target_column, sequence_column=self.sequence_column)
            else:
                raise KeyError('Cannot load sequence classification dataset when target_column is None')
        
        if self.split_idxs:
            '''
            split_idxs contains a tuple of train and split indices for a given fold.
            Use this when running cross-validation (or if a custom val split is desired).
            '''
            train_idx, val_idx = self.split_idxs
            self.train_data = Subset(dataset, train_idx)
            self.val_data = Subset(dataset, val_idx)
        else:
            # Perform standard splitting
            self.train_data, self.val_data = train_test_split(dataset,
                test_size=self.test_size, shuffle=True, random_state=self.random_seed)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, num_workers=4,
            batch_size=self.batch_size, collate_fn=self.pipeline)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, num_workers=1,
            batch_size=self.batch_size, collate_fn=self.pipeline)

class CodonDataModuleHF(CodonDataModule):
    """
    Pytorch Lightning DataModule for reading pre-split Hugging Face datasets.
    Inherits dataloader logic from CodonDataModule
    Accepts either a dataset directory or a folder with train and eval datasets at the specified file paths.
    If train path and eval path are not present, splits the data into train and eval sets.
    """
    def __init__(self, args: Namespace, alphabet: Alphabet, data_path: str, batch_size: int, random_seed: int = 42, 
                 test_size: float = 0.01, fine_tune: bool = False, sequence_column: str = 'codon_sequence', 
                 target_column: bool=None, split_idxs=None, train_path=None, val_path=None):
        # Call the parent constructor with required arguments
        super().__init__(args, alphabet, data_path, batch_size, random_seed, 
                         test_size, fine_tune, sequence_column, target_column, split_idxs)
        self.train_path = train_path
        self.val_path = val_path
        
    def setup(self, stage: Optional[str] = None):
        if self.train_path and self.val_path:
            train_path = f'{self.data_path}/{self.train_path}'
            val_path = f'{self.data_path}/{self.val_path}'
            self.train_data = SequenceDatasetFromHF(train_path, self.sequence_column)
            self.val_data = SequenceDatasetFromHF(val_path, self.sequence_column)
        else:
            dataset = SequenceDatasetFromHF(self.data_path, self.sequence_column)
            self.train_data, self.val_data = train_test_split(dataset,
                test_size=self.test_size, shuffle=True, random_state=self.random_seed)
                                                    