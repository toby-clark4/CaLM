"""Common class for sequence datasets."""

from typing import Tuple

import pandas as pd
import torch
from Bio import SeqIO
from datasets import load_from_disk

from .sequence import CodonSequence, Sequence


class SequenceDataset(torch.utils.data.Dataset):
    """Common class for sequence datasets."""

    def __init__(self, fasta_file: str):
        self.fasta_file = fasta_file
        self._sequences, self._titles = [], []

        for record in SeqIO.parse(fasta_file, "fasta"):
            self._titles.append(record.id)
            self._sequences.append(CodonSequence(record.seq))

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx) -> Sequence:
        return self._sequences[idx]


class SequenceDatasetFromHF(torch.utils.data.Dataset):
    """Common class for sequence Hugging Face datasets"""

    def __init__(self, dataset_path: str, sequence_column: str = "codon_sequence"):
        self.dataset_path = dataset_path
        self.sequence_column = sequence_column

        self._data = load_from_disk(dataset_path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx) -> CodonSequence:
        return CodonSequence(self._data[idx][self.sequence_column])


class SequenceClassificationDataset(torch.utils.data.Dataset):
    """Common class for fine-tuning datasets."""

    def __init__(
        self, csv_file: str, target_column: str, sequence_column: str = "sequence"
    ):
        self.csv_file = csv_file
        self.target_column = target_column
        self.sequence_column = sequence_column

        data = pd.read_csv(csv_file)

        self._sequences = data[sequence_column].apply(CodonSequence).values

        self._labels = data[target_column].values

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx) -> Tuple[CodonSequence, float]:
        return (self._sequences[idx], self._labels[idx])
