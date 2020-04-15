#!/usr/bin/env python
# coding: utf-8

# # Librarires ##

# In[2]:


import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


# # Loading the Original Dataset ##

# In[3]:


df = pd.read_csv("C:/Users/sheha/Desktop/CRISPR_Cas9_Dataset.csv", 
                 names = ['30_mer_Sequence', 'Target_Gene', 'Percent_Peptide', 'Amino_Acid_Cut_Position', 'Score_Drug_Gene_Rank', 'Score_Drug_Gene_Threshold', 'Drug', 'Predictions'])
df = df.iloc[1:,:]
df = pd.DataFrame({'Predictions': df.iloc[0:,7], '30_mer_Sequence': df.iloc[0:,0], 'Percent_Peptide': df.iloc[0:,2], 'Amino_Acid_Cut_Position': df.iloc[0:,3]}) 
df.head(10)


# # Feature Creation 120
# For each index of sequence checking whether it contains certain neucliotide

# In[4]:


single_neucliotides = ['A', 'C', 'G', 'T']

for s_neucliotide in single_neucliotides:
    for i in range(30):
        feature_name = 'Index' + str(i) + '_' + s_neucliotide
        values = []
        for j in range(df.shape[0]):
            sequence = df.iloc[j,1]
            if sequence[i] == s_neucliotide:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[5]:


df.head()


# # Feature Creation 464
# For each two index of sequence checking whether it contains certain neucliotides

# In[6]:


di_neucliotides = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

for di_neucliotide in di_neucliotides:
    for i in range(30-1):
        feature_name = 'Index' + str(i) + "," + str(i+1) + '_' + di_neucliotide
        values = []
        neucliotides = list(di_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,1]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[7]:


df.head()


# # Feature Creation 1792
# For each three index of sequence checking whether it contains certain neucliotides

# In[8]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT", "CAA", 
                    "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT", "CGA", "CGC","CGG", "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG",
                     "GAT", "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT", "TAA", "TAC", "TAG", "TAT",
                     "TCA", "TCC", "TCG", "TCT", "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT"]

for tri_neucliotide in tri_neucliotides:
    for i in range(30-2):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + '_' + tri_neucliotide
        values = []
        neucliotides = list(tri_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,1]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+2] == neucliotides[2]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[9]:


df.head()


# # Feature Creation 6912
# For each four index of sequence checking whether it contains certain neucliotides

# In[10]:


tetra_neucliotides = ["AAAA", "AAAC", "AAAG", "AAAT", "AACA", "AACC", "AACG", "AACT", "AAGA", "AAGC", "AAGG", "AAGT", "AATA", "AATC", "AATG",
                        "AATT", "ACAA", "ACAC", "ACAG", "ACAT", "ACCA", "ACCC", "ACCG", "ACCT", "ACGA", "ACGC", "ACGG", "ACGT", "ACTA", "ACTC",
                        "ACTG", "ACTT", "AGAA", "AGAC", "AGAG", "AGAT", "AGCA", "AGCC", "AGCG", "AGCT", "AGGA", "AGGC", "AGGG", "AGGT", "AGTA",
                        "AGTC", "AGTG", "AGTT", "ATAA", "ATAC", "ATAG", "ATAT", "ATCA", "ATCC", "ATCG", "ATCT", "ATGA", "ATGC", "ATGG", "ATGT",
                        "ATTA", "ATTC", "ATTG", "ATTT", "CAAA", "CAAC", "CAAG", "CAAT", "CACA", "CACC", "CACG", "CACT", "CAGA", "CAGC", "CAGG",
                        "CAGT", "CATA", "CATC", "CATG", "CATT", "CCAA", "CCAC", "CCAG", "CCAT", "CCCA", "CCCC", "CCCG", "CCCT", "CCGA", "CCGC",
                        "CCGG","CCGT", "CCTA", "CCTC", "CCTG", "CCTT", "CGAA", "CGAC", "CGAG", "CGAT", "CGCA", "CGCC", "CGCG", "CGCT", "CGGA",
                        "CGGC", "CGGG", "CGGT", "CGTA", "CGTC", "CGTG", "CGTT", "CTAA", "CTAC", "CTAG", "CTAT", "CTCA", "CTCC", "CTCG", "CTCT",
                        "CTGA", "CTGC", "CTGG", "CTGT", "CTTA", "CTTC", "CTTG", "CTTT", "GAAA", "GAAC", "GAAG", "GAAT", "GACA", "GACC", "GACG",
                        "GACT", "GAGA", "GAGC", "GAGG", "GAGT", "GATA", "GATC", "GATG", "GATT", "GCAA", "GCAC", "GCAG", "GCAT", "GCCA", "GCCC",
                        "GCCG", "GCCT", "GCGA", "GCGC", "GCGG", "GCGT", "GCTA", "GCTC", "GCTG", "GCTT", "GGAA", "GGAC", "GGAG", "GGAT", "GGCA",
                        "GGCC", "GGCG", "GGCT", "GGGA", "GGGC", "GGGG", "GGGT", "GGTA", "GGTC", "GGTG", "GGTT", "GTAA", "GTAC", "GTAG", "GTAT",
                        "GTCA", "GTCC", "GTCG", "GTCT", "GTGA", "GTGC", "GTGG", "GTGT", "GTTA", "GTTC", "GTTG", "GTTT", "TAAA", "TAAC", "TAAG",
                        "TAAT", "TACA", "TACC", "TACG", "TACT", "TAGA", "TAGC", "TAGG", "TAGT", "TATA", "TATC", "TATG", "TATT", "TCAA", "TCAC", 
                        "TCAG", "TCAT", "TCCA", "TCCC", "TCCG", "TCCT", "TCGA", "TCGC", "TCGG", "TCGT", "TCTA", "TCTC", "TCTG", "TCTT", "TGAA",
                         "TGAC", "TGAG", "TGAT", "TGCA", "TGCC", "TGCG", "TGCT", "TGGA", "TGGC", "TGGG", "TGGT", "TGTA", "TGTC", "TGTG", "TGTT",
                        "TTAA", "TTAC", "TTAG", "TTAT", "TTCA", "TTCC", "TTCG", "TTCT", "TTGA", "TTGC", "TTGG", "TTGT", "TTTA", "TTTC", "TTTG",
                        "TTTT"]

for tetra_neucliotide in tetra_neucliotides:
    for i in range(30-3):
        feature_name = 'Index' + str(i) + "," + str(i+1) + "," + str(i+2) + "," + str(i+3) + '_' + tetra_neucliotide
        values = []
        neucliotides = list(tetra_neucliotide)
        for j in range(df.shape[0]):
            sequence = df.iloc[j,1]
            if sequence[i] == neucliotides[0] and sequence[i+1] == neucliotides[1] and sequence[i+2] == neucliotides[2] and sequence[i+3] == neucliotides[3]:
                values.append(1)
            else:
                values.append(0)
        df[feature_name] = values


# In[11]:


df.head()


# # Feature Creation 4
# From each sgRNA Sequence find occurrence of individual neucliotide

# In[12]:


single_neucliotides = ['A', 'C', 'G', 'T']

for s_neucliotide in single_neucliotides:
    feature_name = s_neucliotide + '_Count'
    values = []
    for j in range(df.shape[0]):
        sequence = df.iloc[j,1]
        count = sequence.count(s_neucliotide)
        values.append(count)
    df[feature_name] = values


# In[13]:


df.head()


# # Feature Creation 16
# From each sgRNA Sequence find occurrence of individual di_neucliotides

# In[14]:


di_neucliotides = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]

for di_neucliotide in di_neucliotides:
    feature_name = di_neucliotide + '_Count'
    values = []
    for j in range(df.shape[0]):
        sequence = df.iloc[j,1]
        count = sequence.count(di_neucliotide)
        values.append(count)
    df[feature_name] = values


# In[15]:


df.head()


# # Feature Creation 64
# From each sgRNA Sequence find occurrence of individual tri_neucliotides

# In[16]:


tri_neucliotides = ["AAA", "AAC", "AAG", "AAT", "ACA", "ACC", "ACG", "ACT", "AGA", "AGC", "AGG", "AGT", "ATA", "ATC", "ATG", "ATT", "CAA", 
                    "CAC", "CAG", "CAT", "CCA", "CCC", "CCG", "CCT", "CGA", "CGC","CGG", "CGT", "CTA", "CTC", "CTG", "CTT", "GAA", "GAC", "GAG",
                     "GAT", "GCA", "GCC", "GCG", "GCT", "GGA", "GGC", "GGG", "GGT", "GTA", "GTC", "GTG", "GTT", "TAA", "TAC", "TAG", "TAT",
                     "TCA", "TCC", "TCG", "TCT", "TGA", "TGC", "TGG", "TGT", "TTA", "TTC", "TTG", "TTT"]

for tri_neucliotide in tri_neucliotides:
    feature_name = tri_neucliotide + '_Count'
    values = []
    for j in range(df.shape[0]):
        sequence = df.iloc[j,1]
        count = sequence.count(tri_neucliotide)
        values.append(count)
    df[feature_name] = values


# In[17]:


df.head()


# # Feature Creation 256
# From each sgRNA Sequence find occurrence of individual tetra_neucliotides

# In[18]:


tetra_neucliotides = ["AAAA", "AAAC", "AAAG", "AAAT", "AACA", "AACC", "AACG", "AACT", "AAGA", "AAGC", "AAGG", "AAGT", "AATA", "AATC", "AATG",
                        "AATT", "ACAA", "ACAC", "ACAG", "ACAT", "ACCA", "ACCC", "ACCG", "ACCT", "ACGA", "ACGC", "ACGG", "ACGT", "ACTA", "ACTC",
                        "ACTG", "ACTT", "AGAA", "AGAC", "AGAG", "AGAT", "AGCA", "AGCC", "AGCG", "AGCT", "AGGA", "AGGC", "AGGG", "AGGT", "AGTA",
                        "AGTC", "AGTG", "AGTT", "ATAA", "ATAC", "ATAG", "ATAT", "ATCA", "ATCC", "ATCG", "ATCT", "ATGA", "ATGC", "ATGG", "ATGT",
                        "ATTA", "ATTC", "ATTG", "ATTT", "CAAA", "CAAC", "CAAG", "CAAT", "CACA", "CACC", "CACG", "CACT", "CAGA", "CAGC", "CAGG",
                        "CAGT", "CATA", "CATC", "CATG", "CATT", "CCAA", "CCAC", "CCAG", "CCAT", "CCCA", "CCCC", "CCCG", "CCCT", "CCGA", "CCGC",
                        "CCGG","CCGT", "CCTA", "CCTC", "CCTG", "CCTT", "CGAA", "CGAC", "CGAG", "CGAT", "CGCA", "CGCC", "CGCG", "CGCT", "CGGA",
                        "CGGC", "CGGG", "CGGT", "CGTA", "CGTC", "CGTG", "CGTT", "CTAA", "CTAC", "CTAG", "CTAT", "CTCA", "CTCC", "CTCG", "CTCT",
                        "CTGA", "CTGC", "CTGG", "CTGT", "CTTA", "CTTC", "CTTG", "CTTT", "GAAA", "GAAC", "GAAG", "GAAT", "GACA", "GACC", "GACG",
                        "GACT", "GAGA", "GAGC", "GAGG", "GAGT", "GATA", "GATC", "GATG", "GATT", "GCAA", "GCAC", "GCAG", "GCAT", "GCCA", "GCCC",
                        "GCCG", "GCCT", "GCGA", "GCGC", "GCGG", "GCGT", "GCTA", "GCTC", "GCTG", "GCTT", "GGAA", "GGAC", "GGAG", "GGAT", "GGCA",
                        "GGCC", "GGCG", "GGCT", "GGGA", "GGGC", "GGGG", "GGGT", "GGTA", "GGTC", "GGTG", "GGTT", "GTAA", "GTAC", "GTAG", "GTAT",
                        "GTCA", "GTCC", "GTCG", "GTCT", "GTGA", "GTGC", "GTGG", "GTGT", "GTTA", "GTTC", "GTTG", "GTTT", "TAAA", "TAAC", "TAAG",
                        "TAAT", "TACA", "TACC", "TACG", "TACT", "TAGA", "TAGC", "TAGG", "TAGT", "TATA", "TATC", "TATG", "TATT", "TCAA", "TCAC", 
                        "TCAG", "TCAT", "TCCA", "TCCC", "TCCG", "TCCT", "TCGA", "TCGC", "TCGG", "TCGT", "TCTA", "TCTC", "TCTG", "TCTT", "TGAA",
                         "TGAC", "TGAG", "TGAT", "TGCA", "TGCC", "TGCG", "TGCT", "TGGA", "TGGC", "TGGG", "TGGT", "TGTA", "TGTC", "TGTG", "TGTT",
                        "TTAA", "TTAC", "TTAG", "TTAT", "TTCA", "TTCC", "TTCG", "TTCT", "TTGA", "TTGC", "TTGG", "TTGT", "TTTA", "TTTC", "TTTG",
                        "TTTT"]


for tetra_neucliotide in tetra_neucliotides:
    feature_name = tetra_neucliotide + '_Count'
    values = []
    for j in range(df.shape[0]):
        sequence = df.iloc[j,1]
        count = sequence.count(tetra_neucliotide)
        values.append(count)
    df[feature_name] = values


# In[19]:


df.head()


# # FEATURE GENERATION COMPLETE

# In[21]:


df.to_csv("C:/Users/sheha/Desktop/Feature_Generated_Dataset.csv",index=False)

