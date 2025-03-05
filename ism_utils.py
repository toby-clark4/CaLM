# Common functions for ISM

import pandas as pd

def convert_to_codons(sequence: str) -> str:
    """
    Splits a nucleotide sequence into codons (3-mers) with a stride of 3.
    Pads the sequence with "N" if its length is not a multiple of 3.
    """
    sequence = sequence.upper().replace(" ", "")  # Ensure uppercase and remove spaces
    remainder = len(sequence) % 3
    if remainder != 0:
        sequence += "N" * (3 - remainder)  # Pad with "N"s
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]
    return " ".join(codons)


def mutate_sequence(sequence: str, mode: str = 'nucleotide') -> pd.DataFrame:
    '''
    Performs saturation ISM for either nucleotides or synonymous codons.
    
    Returns a DataFrame with columns position, original_base/codon, mutated_base/codon and sequence.
    '''
    aa_codon_dict = {'D':['GAC','GAT'], 'E': ['GAA','GAG'], 'A':['GCG','GCT','GCA','GCC'],'S': ['TCC','TCT','TCG','TCA','AGT','AGC'],'T':['ACA','ACG','ACT','ACC'],'N':['AAT','AAC'],
                     'W':['TGG'],'Y':['TAT','TAC'],'F':['TTT', 'TTC'],'C':['TGT','TGC'], 'L':['CTC','CTT','CTA','TTA','TTG','CTG'], 'I': ['ATA','ATT','ATC'], 'M':['ATG'], 'V':['GTC','GTT','GTG','GTA'],
                     'Q':['CAG', 'CAA'], 'K': ['AAA','AAG'], 'R': ['AGA','AGG','CGG','CGT','CGC','CGA'], 'H': ['CAT','CAC'], 'P': ['CCG','CCT','CCC','CCA'],' ': ['TAA','TGA','TAG'],
                     'G':['GGA','GGC','GGG','GGT']}
    genetic_code = {'GAC':'D','GAT':'D','GAA':'E','GAG':'E','GCG':'A','GCT':'A','GCA':'A','GCC':'A','TCC':'S','TCT':'S','TCG':'S','TCA':'S','AGT':'S','AGC':'S','ACA':'T','ACG':'T','ACT':'T','ACC':'T','AAT':'N','AAC':'N',
                 'TGG':'W','TAT':'Y','TAC':'Y','TTT':'F','TTC':'F','TGT':'C','TGC':'C','CTC':'L','CTT':'L','CTA':'L','TTA':'L','TTG':'L','CTG':'L','ATA':'I','ATT':'I','ATC':'I','ATG':'M','GTC':'V','GTT':'V','GTG':'V',
                 'GTA':'V','CAG':'Q','CAA':'Q','AAA':'K','AAG':'K','AGA':'R','AGG':'R','CGG':'R','CGT':'R','CGC':'R','CGA':'R','CAT':'H','CAC':'H','CCG':'P','CCT':'P','CCC':'P','CCA':'P','TGA':' ','TAA':' ','TAG':' ',
                'GGA':'G','GGC':'G','GGG':'G','GGT':'G'}
    nucleotides = ['A','C','G','T']
    
    mutations = []
    
    if mode == 'nucleotide':
        for i, original_base in enumerate(sequence):
            for mut_base in nucleotides:
                if original_base == mut_base:
                    continue
                mutated_sequence = sequence[:i] + mut_base + sequence[i+1:]

                mutations.append({
                    "position": i,
                    "original_base": original_base,
                    "mutated_base": mut_base,
                    "sequence": mutated_sequence
                })
    else: # codon ISM
        codon_sequence = convert_to_codons(sequence).split(' ')
        for i, codon in enumerate(codon_sequence):
            try:
                aa = genetic_code[codon]
            except KeyError:
                continue
            for mut_codon in aa_codon_dict[aa]:
                if mut_codon == codon:
                    continue
                mutated_sequence = codon_sequence.copy()
                mutated_sequence[i] = mut_codon

                mutations.append({'position': i,
                                 'original_codon': codon,
                                 'mutated_codon': mut_codon,
                                 'sequence': ''.join(mutated_sequence)
                            })
    
    mutation_df = pd.DataFrame(mutations)
    
    return mutation_df

def optimize_csc(sequence: str, csc_df: pd.DataFrame, experiment: str = 'mean_endo_csc', mode: str = 'optimize'):
    '''
    Optimizes or deoptimizes a codon sequence to the most/least stable by codon stabilisation coefficient
    '''
    if mode == 'optimize':
        # Get highest CSC codon for each amino acid
        opt_df = csc_df.groupby('Name').idxmax()
    elif mode == 'deoptimize':
        # Get the lowest CSC codon for each amino acid
        opt_df = csc_df.groupby('Name').idxmin()
    else:
        print('Mode should be one of "optimize" or "deoptimize", returning the original sequence')
        return sequence
    
    
    codon_sequence = convert_to_codons(sequence).split(' ')
    
    aa_sequence = [csc_df['Name'].get(codon, codon) for codon in codon_sequence]
    
    opt_codon_sequence = [opt_df[experiment].get(aa, aa) for aa in aa_sequence]
    
    return ''.join(opt_codon_sequence)
