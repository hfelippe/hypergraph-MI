import numpy as np
import itertools
import math
from collections import Counter
from mpmath import loggamma

def logchoose(n,k): # using mpmath's loggamma
    """
    log binomial coefficient
    """
    return loggamma(n+1) - loggamma(k+1) - loggamma(n-k+1)

def logmultiset(n,k):
    """
    log multiset coefficient
    """
    return logchoose(n+k-1,k)

def coarse_grain(G,partition):
    """
    return coarse-grained hypergraph of G according to node partition
    returns a dict of form {tuple:count} to represent a multiset
    """
    Gc = {}
    for e in G:

        ec = [partition[i] for i in e]
        new_e = tuple(sorted(ec))

        if not(new_e in Gc): Gc[new_e] = 0
        Gc[new_e] += 1

    return Gc

def logfact(n):
    return loggamma(n+1)

def entropy(probabilities):
    """Calculate entropy from a list of probabilities, handling zeros safely."""
    return -sum(p * np.log(p) for p in probabilities if p > 0)

def NMI_Bulk(N, e1, e2):
    """
    Compute NMI_bulk(e1,e2) for edge sets e1, e2
    Stable for very large N by working in log space when necessary.
    """
    E1,E2,E12 = len(e1),len(e2),len(e1.intersection(e2))

    if e1 == e2:
        return 1.

    if N > 1000:

        H1,H2 = E1*N*np.log(2) - logfact(E1), E2*N*np.log(2) - logfact(E2)
        H12 = (E1+E2-E12)*N*np.log(2) - logfact(E1-E12) - logfact(E2-E12) - logfact(E12)
        MI = H1 + H2 - H12

    else:

        H = 2**N - N - 1
        p1,p2,p12 = E1/H,E2/H,E12/H
        H1,H2 = entropy([p1,1-p1]), entropy([p2,1-p2])
        MI = H1 + H2 - entropy([p12,p1-p12,p2-p12,1-p1-p2+p12])

    norm = (H1 + H2) / 2

    return MI / norm

def get_layers(G,partition=None):
    """
    gets layers of hypergraph G (set of multiple tuple sizes) and puts them into a dict "layers"
    partition is list of N node labels if multiscale measure required
    layers[l] = set of tuples if partition is None, else layers[l] = dict (multiset) of tuples
        with corresponding counts
    """
    indices = list(Counter([len(tup) for tup in G]).keys())
    layers = {}

    if partition is not None:

        G = coarse_grain(G,partition)
        for l in indices:
            layers[l] = {}
        for tup in G:
            l = len(tup)
            layers[l][tup] = G[tup]

    else:
        for l in indices:
            layers[l] = set()
        for tup in G:
            l = len(tup)
            layers[l].add(tup)

    return layers

def H(N,G,partition=None):
    """
    microcanonical entropy of hypergraph set G with N nodes
    partition is list of N node labels if multiscale measure required
    transmits all layers individually
    """
    if partition is not None:

        B = len(set(partition))
        coarse_layers = get_layers(G,partition)
        return sum(logmultiset(math.comb(B+l-1,l),sum(coarse_layers[l].values())) \
                                   for l in coarse_layers)

    else:

        layers = get_layers(G)
        return sum(logchoose(math.comb(N,l),len(layers[l])) for l in layers)

def get_projections(G,layers,partition=None):
    """
    returns dict P such that P[l] is projection of G onto all tuples of size l in list layers
    if partition is not None, then P[l] is a multiset (dict) instead of a set
    """
    P = {}
    for l in layers: P[l] = set()

    for e in G:

        k = len(e)
        layers_to_check = [l for l in layers if l <= k]

        for l in layers_to_check:
            for tup in itertools.combinations(list(e),l):
                P[l].add(tup)

    if partition is not None:
        for l in P:
            P[l] = coarse_grain(P[l],partition)

    return P

def get_sizes_proj(G,indices,max_proj_count,partition=None):
    """
    calculates size of projection of hypergraph G onto tuples of size l for all l in indices
    recurses to get lower-level overlap sizes to subtract off
    does not explicitly compute projections (for this, use get_projections)
    max_proj_count is max # of tuple combinations we allow when explicitly computing projections
    """
    if len(G) == 0: #termination condition for recursion
        return Counter()

    lmax = max(len(t) for t in G)
    sizes = Counter({l:0 for l in indices if (l <= lmax)})

    if (max(math.comb(lmax,l) for l in indices) < max_proj_count) or (partition is not None):
        proj = get_projections(G,indices,partition)
        return Counter({l:len(proj[l]) for l in proj})

    checked = []
    G = list(G)
    for e in G:

        overlaps = set()
        for epast in checked:

            inter = set(list(e)).intersection(set(list(epast)))
            overlaps.add(tuple(sorted(list(inter))))

        sizes_e_max = Counter({l:math.comb(len(e),l) for l in indices if (l <= len(e))})
        sizes += sizes_e_max - get_sizes_proj(overlaps,indices,max_proj_count)
        checked.append(e)

    return sizes

def get_overlap_size(layerk,layerl,partition=None,mode='count'):
    """
    gets size of overlap of layerk and layerl after projecting layerk to order l
    if mode == 'count', runs over layerk to check overlaps with layerk in O(EkEl) time
    if mode == 'direct', projects layerk directly in order O((k choose l)Ek) time
    allows for node partition, in which case uses multiset interaction rather than set interaction
        and only uses mode = 'direct'
    """
    item = layerl.pop()
    layerl.add(item)
    l = len(item)

    if partition is None:

        if mode == 'direct':
            projk = get_projections(layerk,[l],partition)[l]
            return len(projk.intersection(layerl))

        else:
            Eoverlap = 0
            lower_tmp = layerl.copy()
            for eh in layerk:

                higher_set = set(list(eh))
                overlapping_tups = set()
                for el in lower_tmp:

                    overlap = set(list(el)).intersection(higher_set)
                    if len(overlap) == l: overlapping_tups.add(el)

                for t in overlapping_tups:
                    lower_tmp.remove(t)
                    Eoverlap += 1
            return Eoverlap

    else:
        proj1,proj2 = Counter(get_projections(layerk,[l],partition)[l]),\
                        Counter(get_projections(layerl,[l],partition)[l])
        intersection = proj1-(proj1-proj2)
        return sum(intersection.values())

def CE_matrices(N,G1,G2,partition=None,max_proj_count=1000000):
    """
    M2given1[k][l] gives conditional entropy to transmit layer l in G2 from layer k in G1
    M1given2[k][l] gives conditional entropy to transmit layer l in G1 from layer k in G2
    sets M[l][l] = H(l) by default if layer l is missing in the other layer
    allows input node partition
        only computes overlap with mode='direct' when partition is given
    """

    layers1,layers2 = get_layers(G1),get_layers(G2)

    if partition is not None:
        B = len(set(partition))

    M2given1 = {}
    for k in layers1:

        below_k = [l for l in layers2 if l <= k]
        if len(below_k) == 0:
            continue
        sizes1to2 = get_sizes_proj(layers1[k],below_k,max_proj_count)

        M2given1[k] = {}
        for l in sizes1to2:

            size_k2l = sizes1to2[l]

            if (math.comb(k,l) > max_proj_count) and (partition is None): mode = 'count'
            else: mode = 'direct'
            size_k2landl = get_overlap_size(layers1[k],layers2[l],partition,mode)

            size_l = len(layers2[l])

            if partition is None:

                M2given1[k][l] = logchoose(size_k2l,size_k2landl) \
                                + logchoose(math.comb(N,l)-size_k2l,size_l-size_k2landl)
            else:

                M2given1[k][l] = logchoose(size_k2l,size_k2landl) \
                                + logmultiset(math.comb(B+l-1,l),size_l-size_k2landl)

    for l in layers2:
        if not(l in layers1):
            M2given1[l] = {}
            M2given1[l][l] = H(N,layers2[l],partition)

    M1given2 = {}
    for k in layers2:

        below_k = [l for l in layers1 if l <= k]
        if len(below_k) == 0:
            continue
        sizes2to1 = get_sizes_proj(layers2[k],below_k,max_proj_count)

        M1given2[k] = {}
        for l in sizes2to1:

            size_k2l = sizes2to1[l]

            if (math.comb(k,l) > max_proj_count) and (partition is None): mode = 'count'
            else: mode = 'direct'
            size_k2landl = get_overlap_size(layers2[k],layers1[l],partition,mode)

            size_l = len(layers1[l])

            if partition is None:

                M1given2[k][l] = logchoose(size_k2l,size_k2landl) \
                                + logchoose(math.comb(N,l)-size_k2l,size_l-size_k2landl)
            else:

                M1given2[k][l] = logchoose(size_k2l,size_k2landl) \
                                + logmultiset(math.comb(B+l-1,l),size_l-size_k2landl)
    for l in layers1:
        if not(l in layers2):
            M1given2[l] = {}
            M1given2[l][l] = H(N,layers1[l],partition)

    return M1given2,M2given1

def NMIaligned(G1,G2,partition=None):
    """
    compute Ialigned measure between hypergraph sets G1,G2 over N nodes
    """
    if len(G1) == 0:
        if len(G2) == 0: return 1.
        else: return 0.
    elif len(G2) == 0: return 0.

    nodes1 = set(np.concatenate([list(t) for t in G1]))
    nodes2 = set(np.concatenate([list(t) for t in G2]))
    N = len(nodes1.union(nodes2))

    H1,H2 = H(N,G1,partition),H(N,G2,partition)

    M1given2,M2given1 = CE_matrices(N,G1,G2,partition)
    layers1,layers2 = get_layers(G1),get_layers(G2)

    CE1given2 = 0
    for l in layers1:
        CE1given2 += min(H(N,layers1[l],partition),M1given2[l][l])
    
    CE2given1 = 0
    for l in layers2:
        CE2given1 += min(H(N,layers2[l],partition),M2given1[l][l])

    nmi12 = (H1 - CE1given2)/(H1 + 1e-100)
    nmi21 = (H2 - CE2given1)/(H2 + 1e-100)

    return max(nmi12,nmi21)

def NMIcross(G1,G2,partition=None):
    """
    compute Icrossed measure between hypergraph sets G1,G2 over N nodes
    """
    if len(G1) == 0:
        if len(G2) == 0: return 1.
        else: return 0.
    elif len(G2) == 0: return 0.

    nodes1 = set(np.concatenate([list(t) for t in G1]))
    nodes2 = set(np.concatenate([list(t) for t in G2]))
    N = len(nodes1.union(nodes2))

    H1,H2 = H(N,G1,partition),H(N,G2,partition)

    M1given2,M2given1 = CE_matrices(N,G1,G2,partition)
    layers1,layers2 = get_layers(G1),get_layers(G2)

    CE1given2 = 0
    for l in layers1:
        
        k_to_ls = []
        for k in layers2:
            if (k >= l) and (l in M1given2[k]): 
                k_to_ls.append(M1given2[k][l])
                
        if k_to_ls:
            CE1given2 += min(H(N,layers1[l],partition),min(k_to_ls))
        else:
            CE1given2 += H(N,layers1[l],partition)

    CE2given1 = 0
    for l in layers2:
        
        k_to_ls = []
        for k in layers1:
            if (k >= l) and (l in M2given1[k]): 
                k_to_ls.append(M2given1[k][l])
                
        if k_to_ls:
            CE2given1 += min(H(N,layers2[l],partition),min(k_to_ls))
        else:
            CE2given1 += H(N,layers2[l],partition)
        
    nmi12 = (H1 - CE1given2)/(H1 + 1e-100)
    nmi21 = (H2 - CE2given1)/(H2 + 1e-100)

    return max(nmi12,nmi21)
