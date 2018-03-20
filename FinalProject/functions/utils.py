#this is a library of useful functions
import numpy as np

def seq_encode(seq):
    #accept a list of sequences encode them and output a matrix where
    #each column is a training example (encoded sequence)

    A='0001'
    T='0010'
    C='0100'
    G='1000'
    
    encodedseqlist=[]
    for item in seq:
        item=item.upper()
        encodedseq=''
        for letter in item:
            if letter == 'A':
                encodedseq+=A
            if letter == 'T':
                encodedseq+=T
            if letter == 'C':
                encodedseq+=C
            if letter == 'G':
                encodedseq+=G
        encodedseqlist.append(encodedseq)
        #encodedseqlist.append(list(encodedseq))
        
        #print(list(encodedseq))
    #encodedseqlist.append(np.transpose(list(encodedseq)))
    #encodedseqlist=np.matrix(encodedseqlist)
    #encodedseqlist=(np.transpose(encodedseqlist))
    #encodedseqlist=encodedseqlist.astype(int)
    return encodedseqlist

def make_mats(seqlist):

    encodedseqlist=np.matrix(seqlist)
    encodedseqlist=(np.transpose(encodedseqlist))
    encodedseqlist=encodedseqlist.astype(int)
    
    return encodedseqlist

def read_pos(file_name):
    seqs=[]
    with open(file_name,'r') as f:
        for line in f:
            curline=line.rstrip()
            seqs.append(curline)

    return seqs

def read_fasta(file_name):
    seqs=[]
    with open(file_name,'r') as f:
        seq=''
        for line in f:
            line=line.rstrip()
            if line[0]!='>':
                seq+=line  
            else:
                seqs.append(seq)
                seq=''
    seqs=seqs[1:]
    return seqs

#test=['aaattccggtgtcacgt','gggtgccaagagtcgat','gagttgaccagtcagtt']

#print(seq_encode(test))

