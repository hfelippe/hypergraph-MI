# for gt environment -- networkx not allowed!

import random
import numpy as np
from scipy.special import loggamma

# not the best import practice, but easy to fix issues if library is not updated
from graph_tool.all import *  
import graph_tool as gt

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

def graph_ent(S):
    """entropy of graph with edge set S"""
    N = 1 + max(max(edge) for edge in S1) # number of nodes
    NC2 = N*(N-1)/2 
    E = len(S)
    return ent([E, NC2-E])

def graphNMI(S1, S2):
    """normalized mutual information of N-graphs w/ edge sets S1 and S2"""
    N = 1 + max(max(edge) for edge in S1) # number of nodes
    NC2 = N*(N-1)/2
    p1, p2, p12 = len(S1)/NC2, len(S2)/NC2, len(S1.intersection(S2))/NC2
    H12  = ent([p12, p1-p12, p2-p12, 1-p1-p2+p12]) + 1e-100
    H1H2 = ent([p1, 1-p1]) + ent([p2, 1-p2]) + 1e-100
    return 2.-2.*H12/H1H2

def graphDCNMI(S1, S2):
    """degree-corrected NMI between edge sets S1 and S2"""
    g1, g2 = gtgraph_Gset(S1), gtgraph_Gset(S2)
    adj1, adj2 = gt_neigh(g1), gt_neigh(g2)
    N = len(adj1)
    num, denom = 0, 0
    for i in range(N):
        p1, p2 = len(adj1[i])/N, len(adj2[i])/N
        p12 = len(set(adj1[i]).intersection(set(adj2[i])))/N
        H12 = ent([p12, p1-p12, p2-p12, 1-p1-p2+p12])
        H1H2 = ent([p1, 1-p1]) + ent([p2, 1-p2])
        num += H1H2 - H12
        denom += H1H2/2
    return num/denom

def fit_sbm(gt_graph,B=None):
    """fit SBM to gt graph object, returning group membership vector"""
    step = max(int(250000 / gt_graph.num_edges()), 10)
    if B is not None:
        state = minimize_blockmodel_dl(gt_graph,state_args={'B':B,'deg_corr':False},multilevel_mcmc_args=dict(B_min=B, B_max=B))
    else:
        state = minimize_blockmodel_dl(gt_graph,state_args={'B':B,'deg_corr':False})
    stable_max,n_stable = 30,0
    max_sweeps = 500
    
    for _ in range(max_sweeps): 
        ret = state.multiflip_mcmc_sweep(niter=step, beta=1e100)
        if ret[0] < 1e-6:
            n_stable += 1
        else:
            n_stable = 0
        if n_stable > stable_max:
            break
    
    return list(state.b)

def mesoNMI(G1, G2, partition):
    """normalized mesoscale mutual information of edge sets G1, G2"""

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

def gtgraph_Gset(Gset):
    """generate graph-tool graph object from an edge set Gset"""
    g = Graph(directed=False)
    g.add_edge_list(Gset)
    return g

def gt_neigh(g):
    """obtain dictionary of neighbors for all nodes in a graph-tool graph"""
    adj = {}
    for v in g.vertices():
        v_id = int(v)
        neighbors = [int(neighbor) for neighbor in v.all_neighbours()]
        adj[v_id] = neighbors
    return adj

def gt_edges(g):
    """returns edge set of graph-tool graph g"""
    return set(zip(*g.get_edges().T))

def gt_er(N,kavg):
    """function that returns ER graph with kavg in gt"""
    sbm,_=gen_SBM_set(N=N,kavg=kavg,B=2,eta=.5)
    er=gtgraph_Gset(sbm)
    random_rewire(er,model="erdos")
    return er

def gt_ba(N,m):
    """this will return BA graph in gt"""
    return price_network(N=N,m=m,directed=False)

def gt_kavg(g):
    """returns kavg in gt"""
    return vertex_average(g, "total")[0]
    
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
    
    adjlist = {}
    for e in Gset:
        i,j = e
        if not(i in adjlist):
            adjlist[i] = []
        if not(j in adjlist):
            adjlist[j] = []
        adjlist[i].append(j)
        adjlist[j].append(i)
    N = len(adjlist)
    
    # create placeholders for both the addition and removal of edges from graph G
    new_edges = set()
    old_edges = set()
    
    # loop through epsilon*N nodes
    node_order = np.random.permutation(list(adjlist.keys()))
    for i in node_order[:int(eps * N)]:
        for neig in adjlist[i]:
            repeated = True
            while repeated == True:
                to_add = tuple(sorted([i, random.randint(0, N-1)])) # randomly selects pairs (i,k) such that i < k
                if not(to_add in new_edges) and not(to_add in Gset) and not(to_add in old_edges):
                    old_edges.add(tuple(sorted([i,neig])))
                    new_edges.add(to_add)
                    repeated = False
                    
    Gset_new = Gset.difference(old_edges)
    Gset_new = Gset_new.union(new_edges)
    
    return Gset_new


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
