"""Utilities to preprocess data for fine-tuning."""

import abc
import itertools
from copy import deepcopy
from typing import List, Tuple
from collections import namedtuple

import torch
import numpy as np
from Bio.Data.CodonTable import standard_dna_table

from .sequence import Sequence
from .alphabet import Alphabet


def _split_array(array: np.ndarray, chunks: List[int]):
    """Split an array into N chunks of defined size."""
    assert np.sum(chunks) == len(array)
    acc = 0
    arrays = []
    for chunk in chunks:
        arrays.append(array[acc:acc+chunk])
        acc += chunk
    return arrays


PipelineInput = namedtuple('PipelineInput', ['sequence', 'labels'])
PipelineOutput = namedtuple('PipelineOutput', ['input', 'labels'])
_PipelineData = namedtuple('PipelineData',
    ['sequence', 'labels'])

class PipelineData(_PipelineData):
    """Data structure for inner pipeline data."""

    @property
    def size(self):
        """Number of sequences in the data, equivalent
        to batch size."""
        return len(self.sequence)

    def iterate(self):
        """Iterate over the data."""
        for i in range(self.size):
            yield self.sequence[i], self.labels[i]


class PipelineBlock(abc.ABC):
    """Base class for data preprocessing pipeline blocks."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEntrypoint(PipelineBlock):
    """Starting point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineInput) -> PipelineData:
        """Apply the block to a sequence."""
        raise NotImplementedError


class PipelineEndpoint(PipelineBlock):
    """Final point for a pipeline."""

    @abc.abstractmethod
    def __call__(self, input_: PipelineData) -> PipelineOutput:
        """Apply the block to the data."""
        raise NotImplementedError


class FTPipeline:
    """Class to preprocess data for training.

    This class is used to preprocess data for training. It is a pipeline of
    transformations that are applied to the data. The pipeline is defined by a
    list of callables that are applied in order.
    """

    def __init__(self, pipeline: List[PipelineBlock]):
        """Initialize the pipeline.

        Args:
            pipeline: List of callables that are applied in order.
        """

        if not issubclass(type(pipeline[0]), PipelineEntrypoint):
            raise ValueError('First block in a pipeline must be PipelineEntrypoint.')
        for block in pipeline[1:-1]:
            if issubclass(type(block), PipelineEntrypoint) or issubclass(type(block), PipelineEndpoint):
                raise ValueError('Intermediate blocks cannot be PipelineEntrypoint or PipelineEndpoint.')
        self.pipeline = pipeline

    def __call__(self, data_: List[Tuple[Sequence, float]]) -> PipelineEndpoint:
        """Apply the pipeline to the data.

        Args:
            data: Data to apply the pipeline to.

        Returns:
            Data after the pipeline has been applied.
        """
        sequence, labels = zip(*data_) 
        data = PipelineInput(sequence=sequence, labels=labels)
        for transform in self.pipeline:
            data = transform(data)
        return data._asdict()


class FTDataCollator(PipelineEntrypoint):
    """Class to process sequences. The output
    of a call to DataCollator are strings of tokens, separated by spaces,
    and arrays with the labels for the task."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

        if self.alphabet.use_codons:
            self.coding_toks = [''.join(letters)
                for letters in itertools.product(['A', 'U', 'C', 'G'], repeat=3)]
        else:
            self.coding_toks = list('ARNDCQEGHILKMFPSTWYV')

    def __call__(self, input_: PipelineInput) -> PipelineData:
        output = PipelineData(sequence=[], labels=[])
        
        for seq in input_.sequence:
            output.sequence.append(' '.join(seq.tokens))
        for label in input_.labels:
            output.labels.append(label)

        return output


class FTDataTrimmer(PipelineBlock):
    """Class to trim sequences. Returns sequences and masks that have
    been trimmed to the maximum number of positions of the model."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(sequence=[], labels=[])

        for sequence, label in input_.iterate():
            sequence = self._trim_seq(sequence)
            output.sequence.append(sequence)
            output.labels.append(label)

        return output

    def _trim_seq(
        self,
        original_seq: str,
    ) -> str:

        original_tokens = original_seq.split()
        n_tokens = len(original_tokens)
        if n_tokens <= self.params.max_positions:
            return original_seq
        else:
            start = 0 # np.random.randint(0, n_tokens-self.params.max_positions) if dynamically subsampling - can enable or disable this.
            end = start+self.params.max_positions
            new_original_seq = ' '.join(original_tokens[start:end])
            return new_original_seq


class FTDataPadder(PipelineBlock):
    """Class to pad sequences."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet

    def __call__(self, input_: PipelineData) -> PipelineData:
        output = PipelineData(sequence=[], labels=[])

        max_positions = max(len(seq.split()) for seq in input_.sequence)

        for sequence, label in input_.iterate():
            sequence = self._pad_seq(
                sequence,
                max_positions = max_positions)
            output.sequence.append(sequence)
            output.labels.append(label)

        return output

    def _pad_seq(
        self,
        original_seq: str,
        max_positions: int,
    ) -> str:
        n_tokens = len(original_seq.split())
        if len(original_seq.split()) < max_positions:
            original_seq_ = ' '.join(original_seq.split() \
                + ['<pad>'] * (max_positions - n_tokens))
            return original_seq_
        else:
            return original_seq


class FTDataPreprocessor(PipelineEndpoint):
    """Class to transform tokens into PyTorch Tensors."""

    def __init__(self, params, alphabet):
        self.params = params
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()

    def __call__(self, input_: PipelineData) -> PipelineOutput:
        new_input = self._compute_input(input_.sequence)
        labels = self._compute_labels(input_.labels)
        return PipelineOutput(input=new_input, labels=labels)

    def _compute_input(self, seq_list: List[str]) -> torch.Tensor:
        _, _, input_ = self.batch_converter([
            ('', seq) for seq in seq_list])
        return input_.to(dtype=torch.int32)
    
    def _compute_labels(self, labels: List[float]) -> torch.Tensor:
        return torch.tensor(labels)

