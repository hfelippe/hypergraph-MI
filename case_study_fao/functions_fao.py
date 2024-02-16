# functions to work with the FAO trade multiplex (possibly other datasets)

import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import loggamma

def lGogchoose(N,K): 
    """logarithm of binomial coefficient"""
    return loggamma(N+1) - loggamma(N-K+1) - loggamma(K+1)

def lGogmultiset(N,K): 
    """logarithm of multiset coefficient"""
    return loggamma(N+K-1+1) - loggamma(K+1) - loggamma(N-1+1)

def zero_log(x):
    """log of zero is zero"""
    if x <= 0: return 0
    else: return np.log(x)
     
def ent(X):
    """entropy of random variable X"""
    X = np.array(X)/sum(X)
    return -sum(x*zero_log(x) for x in X)

def graph_ent(N, S):
    """entropy of graph w/ N nodes and edge set S"""
    NC2 = N*(N-1)/2 
    E = len(S)
    return ent([E, NC2-E])

def graph_Gset(Gset, N):
    """generate NetworkX Graph object from an edge set Gset"""
    # N = 1 + max(max(edge) for edge in Gset) # number of nodes
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(Gset)
    return G

def comb_laplacian(g, weight=None):
    """combinatorial laplacian graph according to manlio (2015)"""
    K=g.number_of_nodes()
    L=nx.laplacian_matrix(g, weight=None).toarray()
    return L / (2*K)

def vne_ent(rho):
    """von neumann entropy according to braunstein (2006, see sec 3.1 arxiv.org/abs/quant-ph/0406165)"""
    eigs=np.linalg.eigvals(rho)
    return -sum(x*zero_log(x) for x in eigs) / np.log2(len(eigs) - 1)

def d_jensen_shannon(g1, g2):
    """jensen-shannon distance between graphs (see manlio 2015)"""
    rho,sigma=comb_laplacian(g1),comb_laplacian(g2)
    mu=.5*(rho + sigma)
    return vne_ent(mu)-.5*(vne_ent(rho)+vne_ent(sigma))

def graphNMI(S1, S2, N):
    """normalized mutual information of N-graphs w/ edge sets S1 and S2"""
    NC2 = N*(N-1)/2
    p1, p2, p12 = len(S1)/NC2, len(S2)/NC2, len(S1.intersection(S2))/NC2
    H12  = ent([p12, p1-p12, p2-p12, 1-p1-p2+p12]) + 1e-100
    H1H2 = ent([p1, 1-p1]) + ent([p2, 1-p2]) + 1e-100
    return 2.-2.*H12/H1H2

def graphDCNMI(S1, S2, N):
    """degree-corrected NMI between graphs G1 and G2"""
    G1, G2 = graph_Gset(S1,N), graph_Gset(S2,N)
    adj1, adj2 = dict(G1.adjacency()), dict(G2.adjacency())
    N = len(adj1)
    num, denom = 0, 0
    for i in range(N):
        p1, p2 = len(adj1[i])/N, len(adj2[i])/N
        p12 = len(set(adj1[i].keys()).intersection(set(adj2[i].keys())))/N
        H12 = ent([p12, p1-p12, p2-p12, 1-p1-p2+p12])
        H1H2 = ent([p1, 1-p1]) + ent([p2, 1-p2])
        num += H1H2 - H12
        denom += H1H2/2
    return num/denom

def mesoNMI(G1, G2, partition):
    """normalized mesoscale mutual information of graphs w/ edge sets G1, G2"""
    
    def H(G1): 
        """entropy of multiset of edge set G1"""
        B = len(partition)
        BC2 = B * (B - 1) / 2
        return lGogmultiset(BC2 + B,len(G1))

    def get_E12(G1,G2,partition):
        """intersection of edge sets G1, G2"""
        e1,e2 = {},{}
        groups = list(set(partition))
        
        for i,r in enumerate(groups):
            for s in groups[i:]:
                e1[(r,s)] = 0
                e2[(r,s)] = 0
        for edge in G1:
            i,j = edge
            r,s = sorted([partition[i],partition[j]])
            e1[(r,s)] += 1
        for edge in G2:
            i,j = edge
            r,s = sorted([partition[i],partition[j]])
            e2[(r,s)] += 1
        E12 = 0
        for rs in e1:
            E12 += min(e1[rs],e2[rs])
            
        return E12
        
    def MI(G1, G2, E12): 
        """joint entropy of multisets G1 and G2"""
        B = len(partition)
        BC2 = B * (B - 1) / 2
        E1  = len(G1)
        E2  = len(G2)
        
        H1 = lGogmultiset(BC2 + B,E1)
        H2 = lGogmultiset(BC2 + B,E2)
        H12 = lGogmultiset(BC2 + B,E1+E2-E12)

        return H1 + H2 - H12
    
    Imeso = MI(G1,G2,get_E12(G1,G2,partition))
    I0 = MI(G1,G2,0)
    H1H2 = H(G1) + H(G2)

    return (Imeso - I0) / (.5 * H1H2 - I0)
