import numpy as np
import pandas as pd 
from itertools import product
import pathlib
import os 
import sys
import glob

#https://github.com/kathylambchops/cs123a_HMM-based-Prokaryotic-Gene-Prediction/blob/main/Final_CS123A.py


#function from original code, 
#pi - initial states vector 
#a - transitions matrix (nStates x nStates)
#b - emission matrix (nStates x nObs)
#obs - observation sequence of size T
def viterbi(pi, a, b, obs):

    nStates = np.shape(b)[0]  #number of states, first matrix size
    T = np.shape(obs)[0]    #length of observation sequence 
    
    #auxiliary matrix
    path = np.zeros(T)  #final path, most P(x) sequence of states 
    delta = np.zeros((nStates,T)) #matrix of highest P(x) for each state in given time 
    phi = np.zeros((nStates,T)) #points to previous states 
    
    delta[:,0] = pi * b[:,obs[0]] #P(x) of start in state and emission of fist obervation
    phi[:,0] = 0   #only for t=0 where there is no previous state

    #recursion, for each time form 1 to end of sequence 
    for t in range(1,T):

        #for each state 
        for s in range(nStates):

            #calculates best path to reach state s in time t
            #np.max(max value form matrix): product of P(x) of previous states x P(x) of transistion to this state 
            delta[s,t] = np.max(delta[:,t-1]*a[:,s])*b[s,obs[t]]
            #np.argmax(max value indices), which of the previous states gave the best P(x)
            phi[s,t] = np.argmax(delta[:,t-1]*a[:,s])
    
    #termination - state with highest P(x) in T-1 
    path[T-1] = np.argmax(delta[:,T-1])

    #backtracking of sequence, starting form the end and going -1
    # reading from phi what was the previous state  
    for t in range(T-2,-1,-1):

        path[t] = phi[int(path[t+1]), int(t+1)]

    return path, delta, phi

#ADDED, intron count in seq of states 
def count_introns(states):
    intron_count = 0 
    intron_positions = []
    i = 0
    n = len(states)

    while i < n:
        if states[i] == "I":
            if i == 0 or states[i-1] == "E":
                start = i+1
                j = i

                while j < n and states[j] == "I":
                    j += 1

                if j < n and states[j] == "E":
                    intron_count += 1
                    intron_positions.append({
                        "intron_number": intron_count,
                        "start": start,
                        "end": j,
                        "length": j - i
                    })
                i = j 
                continue
        i += 1 
    
    #poprawa: length, nie lengtj
    for pos in intron_positions:
        pos["end"] = pos["start"] + pos["length"] - 1 

    return intron_count, intron_positions

#CHANGED function from originial code 
def get_gene_structure(states):
   
    genes = []  
    current_gene = [] 
    in_gene = False 
    i = 0 

    while i < len(states):

        #find first exon 
        if states[i] == "E" and not in_gene:
            start_exon = i + 1
            in_gene = True 

            j = i
            exon_length = 0

            while j < len(states) and states[j] == "E":
                exon_length += 1
                j += 1

            intron_count_in_gene = 0 

            #check if intron 
            if j < len(states) and states[j] == "I":
                k = j

                while k < len(states):
                    if k < len(states) and states[k] == "I":
                        intron_count_in_gene += 1
                        while k < len(states) and states[k] == "I":
                            k += 1
                    #poprawa: states[k], nie states
                    if k < len(states) and states[k] == "E":
                        while k < len(states) and states[k] == "E":
                            k += 1
                    else:
                        break 
            
                end_exon = k

                genes.append({
                    'Gene': len(genes) + 1,
                    'StartExon': start_exon,
                    'EndExon': end_exon,
                    'ExonLength': exon_length,
                    'IntronCount': intron_count_in_gene,
                    'GeneLength': (end_exon - start_exon + 1)
                })
                
                i = k
                in_gene = False
                continue
            
            else:
                # gene wo intron
                end_exon = j
                
                genes.append({
                    'Gene': len(genes) + 1,
                    'StartExon': start_exon,
                    'EndExon': end_exon,
                    'ExonLength': exon_length,
                    'IntronCount': 0,
                    'GeneLength': exon_length
                })
                
                i = j
                in_gene = False
                continue
        
        i += 1
    
    return pd.DataFrame(genes)
                                                         


#ADDED parsing fasta files 
def parsing_fasta(fasta_file):

    with open(fasta_file, "r") as file:
        header = file.readline().strip()
        sequence = file.read()
        sequence = "".join(sequence.split("\n")).upper()
    
    return header, sequence

#ADDED parsing gff files, 9col type 
def parsing_gff(gff_file):

    exons = []
    with open(gff_file) as file:
        for line in file:
            if line.startswith("#"):
                continue

            parts = line.strip().split("\t")

            #9-col gff check 
            if len(parts) != 9:
                continue

            #poprawa: lower() z nawiasami
            if parts[2].lower() == "exon":
                start = int(parts[3])
                end = int(parts[4])
                exons.append((start, end)) #tup

    return exons 


#building sequence of states EI based on seq lentgh and exons pos 
def build_sequence(seq_len, exons):

    states = ["I"] * seq_len #default, experiemnt, everything is intron until proven otherwise 

    for start, end in exons:
        for i in range(start - 1, end):  #indexing change, 0based 
            if i < seq_len:
                states[i] = "E"

    return states 

#generates all possible 6nt sequences 
def generate_hexamers():

    nts = ["A","T","C","G"]
    hexamers = [''.join(p) for p in product(nts, repeat=6)]

    return hexamers


#6 nt window sequence
def sequential_hexamers(seq):
    hexamers = []

    for i in range(len(seq) - 5):
        h = seq[i:i+6]

        if all(x in ["A","T","C","G"] for x in h): 
            hexamers.append(h)

    return hexamers


"""
multiple files hmm training 
fasta_pattern - data/*.fasta
gff_pattern = None, map {fasta_name}.fasta = {fasta_name}.gff in dir 

"""

def train_hmm_files(fasta_pattern, gff_pattern=None):

    if isinstance(fasta_pattern, str):
        fasta_files = glob.glob(fasta_pattern)
    else:
        fasta_files = fasta_pattern 

    if not fasta_files:
        raise ValueError(f"no fasta files found")
    
    print(f"Parsing {len(fasta_files)} fasta files")

    #generating all hexamers 
    all_hexamers = generate_hexamers()
    hexamers_to_indices = {h: i for i, h in enumerate(all_hexamers)}

    #counting inizialitoan 
    transition_counts = np.ones((2,2))
    emission_counts = np.ones((2, len(all_hexamers)))
    pi_counts = np.ones(2)

    total_exons = 0
    total_seq = 0

    for fasta_file in fasta_files:
        gff_file = None

        #
        if gff_pattern is None:
            base = os.path.splitext(fasta_file)[0]
            for ext in [".gff3", ".gff", ".gtf"]:
                test_file = base + ext
                if os.path.exists(test_file):
                    gff_file = test_file
                    break
        
        elif isinstance(gff_pattern, str):
            #pattern in gff files 
            if '*' in gff_pattern or '?' in gff_pattern:
                gff_files = glob.glob(gff_pattern)
                base = os.path.splitext(os.path.basename(fasta_file))[0]
                for gf in gff_files:
                    if base in gf:
                        gff_file = gf
                        break
            else:
                gff_file = gff_pattern
        
        elif isinstance(gff_pattern, list):
            idx = fasta_files.index(fasta_file) if fasta_file in fasta_files else -1
            if idx >= 0 and idx < len(gff_pattern):
                gff_file = gff_pattern[idx]

        
        if not gff_file or not os.path.exists(gff_file):
            print(f"for file {fasta_file} no annotation file, skipping")
            continue  #skiping missing files, in not paired 

        print(f"loaded {os.path.basename(fasta_file)} + {os.path.basename(gff_file)}")

        try:
            header, sequence = parsing_fasta(fasta_file)
            exons = parsing_gff(gff_file)

            #states building 
            states = build_sequence(len(sequence), exons)
            hexamers = sequential_hexamers(sequence)
            hexamer_states = states[:len(hexamers)]

            state_to_indices = {"E": 0, "I": 1}

            #first state for pi vector for viterbi
            if hexamer_states:
                first_state = state_to_indices[hexamer_states[0]]
                pi_counts[first_state] += 1

            for i, (h, s_char) in enumerate(zip(hexamers, hexamer_states)):

                if h not in hexamers_to_indices:
                    continue
                s = state_to_indices[s_char]
                h_idx = hexamers_to_indices[h]

                #emission and transistion values 
                emission_counts[s, h_idx] += 1

                if i < len(hexamer_states) - 1:
                    next_s_char = hexamer_states[i + 1]
                    if next_s_char in state_to_indices:
                        next_s = state_to_indices[next_s_char]
                        transition_counts[s, next_s] += 1

            total_exons += len(exons)
            total_seq += 1

        except Exception as e:
            print(f"sth went wrong {fasta_file} {e}")
            continue
    
    if total_seq == 0:
        raise ValueError("no file was processed correctly - check if GFF files exist and contain exons")
    
    #conctruction of pi, a, b + normalization 

    pi = pi_counts / np.sum(pi_counts)
    a = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    b = emission_counts / emission_counts.sum(axis=1, keepdims=True)

    print(f"Transition matrix")
    matrix_A = pd.DataFrame(a, index=["E","I"], columns=["E","I"])
    print(matrix_A.round(4))
    
    print(f"Initial probabilities: E={pi[0]:.4f}, I={pi[1]:.4f}")

    return pi, a, b, hexamers_to_indices


'''
single file analysis based on trained hmm values from provided files 
file - input new file 
pi a b hexamer_to_indices - hmm trained data 
'''

def analyze_sequence(file, pi, a, b, hexamer_to_indices):
    try:
        with open(file, "r") as f:
            accession = f.readline().strip()
            sequence = f.read()
            sequence = "".join(sequence.split("\n")).upper()

    except FileNotFoundError:
        print(f"{file} no such file or directory")
        return None, None, None 

    print(f"Analysis of {os.path.basename(file)}")

    hexamers = sequential_hexamers(sequence)

    #mapping hexamers to states 
    mapped = []
    skipped = 0 
    for element in hexamers:
        if element in hexamer_to_indices:
            mapped.append(hexamer_to_indices[element])
        else:
            skipped += 1

    if len(mapped) == 0:
        print(f"Something wrong with hexamers mapping")
        return None, None, None 
    
    obs = np.array(mapped)


    path, delta, phi = viterbi(pi, a, b, obs)

    #back again to states 
    state_map = {0: "E", 1: "I"}
    state_path = [state_map[int(v)] for v in path]    

    gene_df = get_gene_structure(state_path)

    intron_count, intron_positions = count_introns(state_path)

    exon_hexamers = state_path.count("E")
    intron_hexamers = state_path.count("I")
    
    stats = {
        'accession': accession,
        'filename': os.path.basename(file),
        'sequence_length': len(sequence),
        'exon_hexamers': exon_hexamers,
        'intron_hexamers': intron_hexamers,
        'genes_found': len(gene_df) if gene_df is not None else 0,
        'intron_count': intron_count,
        'total_gene_length': gene_df['GeneLength'].sum() if gene_df is not None and not gene_df.empty else 0
    }

    if intron_count > 0:
        print(f"\nINTRONS")
        print(f"Intron count: {intron_count}")
        print("\nIntron positions:")
        for pos in intron_positions:
            print(f"Intron {pos['intron_number']}: positions {pos['start']}-{pos['end']} (length: {pos['length']} nt)")
    
    return gene_df, state_path, stats


def main():
    
    print("-"*80)
    print("HMM BASED GENE PREDICTION")
    print("-"*80)


    fasta_pattern = "hmm_data/*.fasta"
    gff_pattern = None 


    pi = None
    a = None
    b = None
    hexamer_to_indices = None

    try:
        pi, a, b, hexamer_to_indices = train_hmm_files(fasta_pattern, gff_pattern)
        print("\n✓ Model training successful!")

    except ValueError as e:
        print(f"\n✗ Training failed: {e}")
        print("\nPlease check:")
        print("  1. Do the FASTA files exist in 'hmm_data/' folder?")
        print("  2. Do the GFF files exist (same name as FASTA files)?")
        print("  3. Do the GFF files contain 'exon' entries?")
        print("\nExiting program.")
        return  

    #list test files
    test_files = ["seq1.txt", "seq2.txt", "seq3.txt"]

    all_results = []
    all_stats = []

    for file in test_files:
        if not os.path.exists(file):
            print(f"File {file} does not exists, skipping")
            continue
        try:
            gene_df, state_path, stats = analyze_sequence(file, pi, a, b, hexamer_to_indices)

            if gene_df is not None and not gene_df.empty:
                print("\n✓ Genes identified:")
                print(gene_df.to_string(index=False))
            elif gene_df is not None:
                print("\nNo gene-like structures were found in this sequence")

            if stats:
                all_stats.append(stats)
                print("-"*80)
                print("SUMMARY")
                print(f"Genes found: {stats['genes_found']}")
                print(f"Introns found: {stats['intron_count']}")
                print(f"Exon hexamers: {stats['exon_hexamers']}")
                print(f"Intron hexamers: {stats['intron_hexamers']}")

            if state_path:
                path_preview = ''.join(state_path[:50])
                if len(state_path) > 50:
                    path_preview += "..."
                print(f"\nState path (first 50): {path_preview}")
                
        except Exception as e:
            print(f"\nError with file analysis {file}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()