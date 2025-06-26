from collections import namedtuple

from calm.alphabet import Alphabet
from calm.sequence import CodonSequence
from calm.ft_pipeline import (
    FTPipeline,
    PipelineInput,
    FTDataCollator,
    FTDataTrimmer,
    FTDataPadder,
    FTDataPreprocessor,
)


def fake_args():
    Args = namedtuple('args', [
        'mask_proportion',
        'max_positions',
        'mask_percent',
        'leave_percent'
    ])
    return Args(mask_proportion=.25, max_positions=10,
        mask_percent=.8, leave_percent=.1)

def test_FTDataCollator_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_collator = FTDataCollator(args, alphabet)

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    input_ = PipelineInput(sequence=[seq1, seq2], labels=[10.6, 10.8])
    output = data_collator(input_)
    assert output.sequence[0] == seq1.seq

def test_FTDataTrimmer_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_trimmer = FTPipeline([
        FTDataCollator(args, alphabet),
        FTDataTrimmer(args, alphabet)
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_trimmer([(seq1, 10.6), (seq2, 10.8)])
    assert len(output.sequence[0].split(' ')) == args.max_positions

def test_FTDataPadder_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_padder = FTPipeline([
        FTDataCollator(args, alphabet),
        FTDataTrimmer(args, alphabet),
        FTDataPadder(args, alphabet),
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_padder([(seq1, 10.6), (seq2, 10.8)])
    assert len(output.sequence[1].split(' ')) == args.max_positions
    
def test_FTDataPreprocessor_codon():
    args = fake_args()
    alphabet = Alphabet.from_architecture('CodonModel')
    data_preprocessor = FTPipeline([
        FTDataCollator(args, alphabet),
        FTDataTrimmer(args, alphabet),
        FTDataPadder(args, alphabet),
        FTDataPreprocessor(args, alphabet)
    ])

    seq1 = CodonSequence('AUG GGA CGC UUU UAC CAA AUG GGA CGC UUU UAC CAA UAA ' * 10)
    seq2 = CodonSequence('AUG GGA CGC UAA')
    output = data_preprocessor([(seq1, 10.6), (seq2, 10.8)])

