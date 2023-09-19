import random
import numpy as np
import networkx as nx
from scipy.special import loggamma

def lGogchoose(N,K): 
    """logarithm of binomial coefficient"""
    return loggamma(N+1) - loggamma(N-K+1) - loggamma(K+1)

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

def graphNMI(N, S1, S2):
    """normalized mutual information of N-graphs w/ edge sets S1 and S2"""
    NC2 = N*(N-1)/2
    p1, p2, p12 = len(S1)/NC2, len(S2)/NC2, len(S1.intersection(S2))/NC2
    H12  = ent([p12, p1-p12, p2-p12, 1-p1-p2+p12]) + 1e-100
    H1H2 = ent([p1, 1-p1]) + ent([p2, 1-p2]) + 1e-100
    return 2.-2.*H12/H1H2

def graphDCNMI(G1, G2):
    """degree-corrected NMI between graphs G1 and G2"""
    G1, G2 = graph_Gset(G1), graph_Gset(G2)
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
    """normalized mesoscale mutual information of graphs"""

    def entropy_multiset(Gset, partition):
        """entropy of individual multiset"""
        E = len(Gset)
        B = len(set(partition))
        BC2 = B * (B - 1) / 2 # B choose 2
        return lGogchoose(BC2 + B + E - 1, E)

    def entropy_joint_multiset(G1, G2, partition):
        """joint entropy of multisets G1 and G2"""
        E1  = len(G1)
        E2  = len(G2)
        E12 = len(G1.intersection(G2))

        B = len(set(partition))
        BC2 = B * (B - 1) / 2
        n = BC2 + B
        k = E1 + E2 - E12
        
        return lGogchoose(n + k - 1, k)

    def mesoMI(G1, G2, partition):
        """mesoscale mutual information of graphs with respect to partition"""
        H1  = entropy_multiset(G1, partition)
        H2  = entropy_multiset(G2, partition)
        H12 = entropy_joint_multiset(G1, G2, partition)
        return H1 + H2 - H12
        
    def mesoMI_nonoverlap(G1, G2, partition):
        """mesoMI of graphs without overlapping edges"""
        N = len(partition) 
        comms = sorted(list(set(partition))) 
        B = len(comms)
        # obtain dict of community-community edges
        e1 = 0
        edges1 = {}
        for edge1 in G1:
            i,j = edge1
            r,s = sorted([partition[i],partition[j]]) 
            if not((r,s) in edges1): 
                edges1[(r,s)] = 0
            edges1[(r,s)] += 1
            e1 += 1      
        e2 = 0
        edges2 = {}
        for edge2 in G2:
            i,j = edge2
            r,s = sorted([partition[i],partition[j]]) 
            if not((r,s) in edges2): 
                edges2[(r,s)] = 0
            edges2[(r,s)] += 1
            e2 += 1      
        for r in set(partition):
            for s in set(partition):
                if r <= s:
                    if not((r,s) in edges1):
                        edges1[(r,s)] = 0
                    if not((r,s) in edges2):
                        edges2[(r,s)] = 0
        e12 = 0
        for r in set(partition):
            for s in set(partition):
                if r <= s:
                    e12 += min(edges1[(r,s)], edges2[(r,s)])   
        E1  = e1
        E2  = e2
        BC2 = B * (B - 1) / 2
        E12 = e12
        
        n1, k1   = BC2 + B + E1 - 1, E1
        n2, k2   = BC2 + B + E2 - 1, E2
        n12, k12 = BC2 + B + E1 + E2 - 1, E1 + E2 - E12
        
        return lGogchoose(n1 + k1 - 1, k1) + lGogchoose(n2 + k2 - 1, k2) - lGogchoose(n12 + k12 - 1, k12)

    ImesoG1G2 = mesoMI(G1, G2, partition)
    ImesoE12  = mesoMI_nonoverlap(G1, G2, partition)
    H1H2 = entropy_multiset(G1, partition) + entropy_multiset(G2, partition) 
    return (ImesoG1G2 - ImesoE12) / (.5 * H1H2 - ImesoE12)

def graph_Gset(Gset):
    """generate NetworkX Graph object from an edge set Gset"""
    Gset = Gset.copy()
    N = 1 + max(max(edge) for edge in Gset) # number of nodes
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(Gset)
    return G
    
def jaccard(A, B):
    """Jaccard index of sets A and B"""
    return len(A & B) / (len(A) + len(B) - len(A & B))

def gen_SBM_set(N,kavg,B,eta):
    """Generates SBM w/ N nodes, average degree kavg, B groups, and mixing eta"""  
    # assign nodes to communities
    comms = dict.fromkeys(range(B)) 
    for r in range(B):
        comms[r] = [] 
    partition = np.zeros(N).astype('int') 
    for i in range(N):
        r = int(i*B/N)
        partition[i] = r
        comms[r].append(i)

    # elements of mixing matrix
    diag = N*kavg*eta/B 
    if B == 1: offdiag = N*kavg*(1-eta)/B/(B-1 + 1e-100)
    else: offdiag = N*kavg*(1-eta)/B/(B-1)
    
    # edges generation
    edges = set()
    for r in range(B):
        for s in range(r,B):
            e_rs = int( diag*(r == s)/2 + offdiag*(r != s) )
            count = 0
            while count < e_rs: 
                i, j = random.choice(comms[r]), random.choice(comms[s]) 
                i, j = sorted([i,j])
                if not((i,j) in edges) and (i != j):
                    edges.add((i,j))
                    count += 1
                else:
                    do_nothing = 1
                
    return edges, partition


def typeI(Gset, eps):
    """Type I noise over nodes"""
    N = 1 + max(max(edge) for edge in Gset) # number of nodes
    
    # create NetworkX Graph object to further obtain nodes' neighbors
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(Gset)

    # create placeholders for both the addition and removal of edges from graph G
    new_edges = set()
    old_edges = list()
    
    # take the neighbors of all nodes
    neighbors = [G.neighbors(i) for i in G.nodes()]
    
    # loop through epsilon*N nodes
    for i in range(int(eps * N)):
        for neigs in neighbors[i]:
            if neigs > i: # take only neighbors j such that i < j for all pairs (i,j)
                while True:
                    to_add = (i, random.randint(i+1, N-1)) # randomly selects pairs (i,k) such that i < k
                    if not(to_add in Gset) and not(to_add in new_edges):
                        old_edges.append((i, neigs))
                        new_edges.add(to_add)
                        break
    
    # remove and add the attacked edges to G
    G.remove_edges_from(old_edges)
    G.add_edges_from(new_edges)
    
    return set(G.edges()) # returns the set of edges of G


def typeII(Gset, eps):
    """Type II noise over edges"""
    N = 1 + max(max(edge) for edge in Gset) # number of nodes
    edges = Gset.copy()
    new_edges = set()
    rand_ij = eps*len(edges)
    count = 0
    while count < rand_ij:
        to_add = (random.choice(range(N)), random.choice(range(N)))
        if to_add[0]!=to_add[1] and not(to_add in edges) and not((to_add[1], to_add[0]) in edges) and not(to_add in new_edges) and not((to_add[1], to_add[0]) in new_edges):
            to_add = (min(to_add), max(to_add)) # imposes i < j for all edges (i,j)
            edges.pop()
            new_edges.add(to_add)
            count += 1
        else:
            pass
    
    return new_edges.union(edges)


def typeIII(Gset, partition, eps):
    """Type III noise over community-community edges"""   
    N = len(partition) # number of nodes
    comms = sorted(list(set(partition)))
    B = len(comms)

    edges = {}
    for edge in Gset:
        i,j = edge
        r,s = sorted([partition[i],partition[j]]) 
        if not((r,s) in edges): 
            edges[(r,s)] = set()
        edges[(r,s)].add((i,j)) 
    
    comm_sets = {l:[] for l in comms}
    for i in range(N):
        comm_sets[partition[i]].append(i)
    
    for rs in edges.keys():
        new_edges=set()
        r,s = rs           
        rand_rs = eps*len(edges[(r,s)])
        count = 0
        while count < rand_rs:
            to_add = (random.choice(comm_sets[r]),random.choice(comm_sets[s]))
            if not(to_add in new_edges) and not((to_add[1],to_add[0]) in new_edges) and (to_add[0] != to_add[1]) and not(to_add in edges[(r,s)]) and not((to_add[1],to_add[0]) in edges[(r,s)]):
                edges[(r,s)].pop()
                new_edges.add(to_add)
                count += 1
            else:
                pass            
        edges[(r,s)] = new_edges.union(edges[(r,s)])
    
    return set().union(*list(edges.values()))
