#!/usr/bin/env python3 

import argparse
import glob
import os
import re
from itertools import product
import numpy as np
import pandas as pd 
from Bio import SeqIO

import time
import sys


'''
Hidden Markov Model implementation for detecting introns in phages genomes with hidden states:
- N - non intron 
- I - intron 
- ENDO - endonuclease encoded inside intron 

For training: DNA seq. as fasta, and annotation file as .gtf. 
'''

STATE_TO_INDEX = {"N":0,"I":1,"ENDO":2}
INDEX_TO_STATE = {0:"N",1:"I",2:"ENDO"}

def parse_fasta(fasta_file):
    sequences = {}

    for record in SeqIO.parse(fasta_file,"fasta"):
        sequence = str(record.seq).upper()

        clean_sequence = ""
        for nt in sequence:
            if nt in ["A","C","T","G"]: 
                clean_sequence += nt 
        sequence = clean_sequence

        sequences[record.id] = sequence

    if len(sequences) == 0:
        raise ValueError("Something wrong with fasta file")
    
    return sequences

def parse_gtf(gtf_file):
    #dicts for annotation elements 
    introns_seq = {}
    parts_seq = {}
    cds_seq = {}

    with open(gtf_file,"r") as file:
        for line in file: 
            if line.startswith("#") or line.strip() == "": continue

            columns = line.strip().split("\t")
            seqid = columns[0]
            feature = columns[2].lower()
            start = int(columns[3])
            end = int(columns[4])
            strand = columns[6]
            attributes = columns[8]

            #
            if feature not in ["intron", "exon", "cds"]:
                continue

            gene_id = attributes.split('gene_id "')[1].split('"')[0]
            transcript_id = attributes.split('transcript_id "')[1].split('"')[0]

            transcript_key = transcript_id + "|" + strand + "|" + gene_id   #unique key for each transcript 

            if seqid not in introns_seq:
                introns_seq[seqid] = []
                parts_seq[seqid] = {}
                cds_seq[seqid] = []

            if feature == "intron":
                introns_seq[seqid].append((start, end))

            elif feature == "exon" or feature == "cds":
                if transcript_key not in parts_seq[seqid]:
                    parts_seq[seqid][transcript_key] = []

                parts_seq[seqid][transcript_key].append((start, end))

                if feature == "cds":
                    cds_seq[seqid].append((start, end))

    return introns_seq, parts_seq, cds_seq

#for better intron detecing if intron encoded: eg. join(43535..44814,45848..46385) becomes 43535..46385, the same transcript is dispersed or overlaps 
def merging_intervals(intervals):
    if len(intervals) == 0: 
        return []

    intervals = sorted(intervals)
    merged = [] 
    curr_start = intervals[0][0]
    curr_end = intervals[0][1]

    for start, end in intervals[1:]:
        if start <= curr_end +1:
            if end > curr_end: 
                curr_end = end 
        else:
            merged.append((curr_start,curr_end))
            curr_start = start 
            curr_end = end

    merged.append((curr_start,curr_end)) 

    return merged  

#if introns are not explicit in annotation, counts intron in files that dont have them, in case of t4 gtf useless 
def infer_introns_from_parts(parts_by_transcript):
    introns = []

    for transcript_key in parts_by_transcript:
        parts = parts_by_transcript[transcript_key]
        parts = merging_intervals(parts)  #merge two transcripts that have the same key, intron between 

        for i in range(len(parts)-1):
            intron_start = parts[i][1] +1
            intron_end = parts[i+1][0] - 1

            if intron_start <= intron_end: 
                introns.append((intron_start,intron_end))

    return introns

def find_endonuclease_regions(introns, cds_regions):
#if cds is located between intron[pos[0]],intron[pos[-1]] that must be endonuclease, could indicate false endonucelase state if intron is detected wrong???

    endo_regions = []

    for cds_start, cds_end in cds_regions:
        for intron_start, intron_end in introns:
            if cds_start >= intron_start and (cds_end <= intron_end):
                endo_regions.append((cds_start,cds_end))

    return endo_regions 

#logic entirely changed from before - no exon state, only endonuc, intron, the rest. what was done in get_gene_structure before is here 
def bulid_states(sequence_length, introns, endo_regions):
    states = ["N"] * sequence_length #list of states for each nt., all introns are unknown so N 

    #build full introns - overwritting the original N state 
    for start, end in introns:
        start_index = start - 1
        end_index = end 

        if start_index < 0:
            start_index = 0 
        if end_index > sequence_length:
            end_index = sequence_length

        for i in range(start_index, end_index):
            states[i] = "I"

    #overwrting intron state with endonuclease regions state  
    for start, end in endo_regions:
        start_index = start -1 
        end_index = end 

        if start_index < 0:
            start_index = 0
        if end_index > sequence_length:
            end_index = sequence_length
        
        for i in range(start_index,end_index):
            states[i] = "ENDO"

    return states 

#generation of kmers of length k, k default 6, can be changed (TOADD parser)

def generate_kmers(k):
    nucleotides = ["A","C","T","G"]
    kmers = []

    for p in product(nucleotides,repeat = k ): 
        kmer = "".join(p)
        kmers.append(kmer)

    return kmers

def kmers_from_sequence(sequence,k):
    kmers = []

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        oki = True 
        for nt in kmer:
            if nt not in set("ACTG"):
                oki = False
                break 
        if oki: kmers.append((i,kmer))

    return kmers 

#decide what state does the current window belong to
#TOADD change with padding??

#now it checks what is the dominating state in current window, checks how many times each state occurs
#works within traning function, which state to encode when hexamer is like: NNIIII
def kmer_window_state(states, start, k):
    counts = {"N":0,"I":0,"ENDO":0}

    window = states[start:start + k]
    for state in window:
        counts[state] += 1 

    #choosing the most common state in the current window
    #for most general N, most commonly occuring, only sometimes overwritten 
    best_state = "N"
    best_count=counts["N"]

    if counts["I"] > best_count:
        best_state = "I"
        best_count = counts["I"]

    if counts["ENDO"] > best_count:
        best_state = "ENDO"

    return best_state 


'''
Implementation of Viterbi algorithm with log scale for low probabilites of emitting 
certain states like intron, endonuclease 
'''

#chamged from original code 
#pi - initial states vector 
#a - transitions matrix (nStates x nStates)
#b - emission matrix (nStates x nObs)
#obs - observation sequence of size T
def viterbi(pi,a,b,obs):
    n_states = b.shape[0]  #number of states, first matrix size
    T = len(obs)   #length of observation sequence 

    #p(x) changed to log to avoid ignoring small nums.
    log_pi = np.log(np.maximum(pi,1e-300))
    log_a = np.log(np.maximum(a,1e-300))
    log_b = np.log(np.maximum(b,1e-300))

    delta = np.full((n_states, T), -np.inf)
    phi = np.zeros((n_states, T), dtype=int)
    path = np.zeros(T, dtype=int)

    delta[:, 0] = log_pi + log_b[:, obs[0]]

    for t in range(1, T):
        for s in range(n_states):
            scores = delta[:, t - 1] + log_a[:, s]
            best_previous_state = np.argmax(scores)

            phi[s, t] = best_previous_state
            delta[s, t] = scores[best_previous_state] + log_b[s, obs[t]]

    path[T - 1] = np.argmax(delta[:, T - 1])

    for t in range(T - 2, -1, -1):
        path[t] = phi[path[t + 1], t + 1]

    return path

'''
Training markov model based on input fasta and gtf files, transmatrix 
'''

def markov_train(train_fasta_files,train_annotation_files,k):
    #generate kmers 
    all_kmers = generate_kmers(k)
    kmer_to_indicies = {} 

    for i, kmer in enumerate(all_kmers): 
        kmer_to_indicies[kmer] = i

    #laplace smoothing, fill array with ones instead of zeros 
    pi_counts = np.ones(3) 
    transitions_counts = np.ones((3,3))
    emission_counts = np.ones((3,len(all_kmers)))

    process = 0

    #in loop for mulit_file training, read fasta and corresponding gtf 
    for file_index in range(len(train_fasta_files)):
        fasta_file = train_fasta_files[file_index]
        sequences = parse_fasta(fasta_file)

        annotation_file = train_annotation_files[file_index]
        introns_seq, parts_seq, cds_seq = parse_gtf(annotation_file)

        for seqid in sequences: 
            sequence = sequences[seqid]
            if seqid not in (introns_seq): continue

            #gets introns from both functions seekeing introns, in the original exons(CDS) were counted along - build states 
            introns = introns_seq[seqid]

            if len(introns) == 0:
                introns = infer_introns_from_parts(parts_seq[seqid]) #ale gdzie ty je akt?? 
            if len(introns) == 0: continue

            cds_regions = cds_seq[seqid]
            endo_regions = find_endonuclease_regions(introns,cds_regions)  #checks if endonuclease was found in found introns 
            nt_states = bulid_states(len(sequence),introns, endo_regions)  #creates nt states from annotation 

            windows = kmers_from_sequence(sequence, k)

            obs = []
            path = []

            for pos,kmer in windows:
                state = kmer_window_state(nt_states,pos,k) #chcecks for dominating state in kmer, that is the kmers state 
                state_index = STATE_TO_INDEX[state]
                kmer_index = kmer_to_indicies[kmer]

                obs.append(kmer_index)
                path.append(state_index)

            pi_counts[path[0]] += 1

            for i in range(len(obs)):
                state_index = path[i]
                kmer_index = obs [i]

                emission_counts[state_index,kmer_index] +=1  #final emission count, which kmers is emitted by which state 

                if i < len(obs) -1:
                    next_state_index = path[i+1]
                    transitions_counts[state_index,next_state_index] +=1 #for transition matrix 

            process += 1

    if process == 0:
        raise ValueError("No sequence was processesed during training")
    
    pi = pi_counts/pi_counts.sum() #initial p(x)
    a = transitions_counts/transitions_counts.sum(axis=1,keepdims=True) #transition matrix 
    b= emission_counts/emission_counts.sum(axis=1,keepdims=True) #emissions matrix 

    states = ["N","I","ENDO"]

    #better looking than np array, not mixing up nt.array nad pd.dataframe, only for printing 
    transition_matrix = pd.DataFrame(a,index=states,columns=states)

    print("\nTransition matrix")
    print(transition_matrix.round(8))

    return pi,a,b,kmer_to_indicies

def intron_states(positions, states, sequence_length,k,min_intron_len):
    nt_states = ["N"] * sequence_length #first each nt is unkown, def not intron 
    #prev. hmm works on kmers, here kmer state is its middle mt 

    #middle nt as representation of kmer state -> enhance introns sensivity 
    for i in range(len(positions)):
        pos = positions[i]
        state = states[i]
        center = pos + k//2

        if center < sequence_length: 
            nt_states[center] = state 

    introns = []
    i = 0

    while i < sequence_length: 
        if nt_states[i] != "I" and nt_states[i] != "ENDO":
            i += 1
            continue
        start=i+1

        while i < sequence_length and(nt_states[i] == "I" or nt_states[i] == "ENDO"):
            i += 1 
        end = i

        #min_length of intron required, without it detetcs small intron-like emissions in sequence with <70nt (what are those??)
        if (end - start + 1) >= min_intron_len:
            introns.append((start,end, end-start+1))

    return introns 

#returns only intron states in predicted, no CDS will be there, including endonuclease 
def analyse_sequence(input_fasta_file,pi,a,b,kmer_to_indicies,k,min_intron_len):
    #read sequence 
    sequences = parse_fasta(input_fasta_file)
    prediction = {}

    #makes kmers form sequence 
    for seqid in sequences:
        sequence = sequences[seqid]
        windows = kmers_from_sequence(sequence,k)

        if len(windows) == 0:
            prediction[seqid] = []
            continue 

        positions = []
        obs = []

    #kmers to indices and viterbei path 
        for pos, kmer in windows: 
            if kmer in kmer_to_indicies:
                positions.append(pos)
                obs.append(kmer_to_indicies[kmer])

        if len(obs) == 0:
            prediction[seqid] = []
            continue

        obs = np.array(obs)
        path = viterbi(pi,a,b,obs)

#states to normal states and gets correct introns with given min_intron_len 
        prediction_states = []
        for state_index in path:
            prediction_states.append(INDEX_TO_STATE[int(state_index)])

        introns = intron_states(positions,prediction_states,len(sequence),k,min_intron_len)
        prediction[seqid] = introns 

    return prediction

#gft saving for visualization sake 
def save_to_gtf(output_file,prediction):

    with open(output_file,'w') as file:
        for seqid in prediction:
            introns = prediction[seqid]
            
            for i,intron in enumerate(introns):
                start = intron[0]
                end = intron [1]
                number = i + 1

                attri = (
                    f'gene_id,predicted_intron{number};" '
                    f'transcript_id,"predicted_intron{number}"; '
                )
                
                #9col GTF saving file, for not important . xd
                row = [seqid,"HMM","intron",str(start),str(end),".",".",".",attri]

                file.write("\t".join(row) + "\n")


#corr for only multiple file training system 
def multi_file(files):
    expanded = []
    for file_name in files:
        if "*" in file_name or "?" in file_name:
            matched = sorted(glob.glob(file_name))
            
            for m in matched:
                expanded.append(m)
        else: 
            expanded.append(file_name)

    return expanded


def fish():
    for i in range(20):
        print("\r" + " " * i + "><(((º>", end="")
        time.sleep(0.08)
    print()


def main():

    fish()
    parser = argparse.ArgumentParser(description="Three state HMM detection for introns")

    parser.add_argument("--train-fasta",nargs="+",required=True)
    parser.add_argument("--train-annotation",nargs="+",required=True)
    parser.add_argument("--input",required=True)
    parser.add_argument("--output-gtf",default=None)
    parser.add_argument("--k",type=int,default=6)
    parser.add_argument("--min-intron-len",type=int,default=500)

    args = parser.parse_args()

    train_fasta_files = multi_file(args.train_fasta)
    train_annotation_files = multi_file(args.train_annotation)

    if len(train_fasta_files) != len(train_annotation_files):
        raise ValueError("The same number of fasta and gtf files must be provided")
    
    pi,a,b,kmers_to_index = markov_train(train_fasta_files,train_annotation_files,args.k)

    prediction = analyse_sequence(args.input,pi,a,b,kmers_to_index,args.k,args.min_intron_len,)

    for seqid in prediction:
        print(f"Sequence: {seqid}")

        if len(prediction[seqid]) == 0:
            print("No introns found")
        else:
            for i, intron in enumerate(prediction[seqid]):
                start = intron[0]
                end = intron[1]
                lentgh = intron[2]

                print(f"Intron {i+1}: {start}-{end}, length: {lentgh}")


    if args.output_gtf is not None:
        save_to_gtf(args.output_gtf, prediction)
        print(f"gtf files written to: {args.output_gtf}")


if __name__ == "__main__":
    main()









    





