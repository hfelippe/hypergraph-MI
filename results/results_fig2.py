from extrafunctions import *

"""
    (a)
"""


N,d,E=100,6,5
Edict={l: 5**(l) for l in range(2,7)} 

system=f"Hrandom-N-{N}-lmax-{d}-dense"

print(Edict)
ps=np.linspace(0,1,15)

IAs=[]
IBs=[]
Ms=[]
ys,xs=[],[]

nsim=10
for n_sim in range(nsim):
    
    Msim=[]
    IBsim,IAsim=[],[]

    # attack
    for count,p in enumerate(ps):

        ####Initization before randomization
        h1=random_hypergraph(N,Edict)
        h2=h1.copy()

        ## Randomization of hypergraphs
        random_shuffle_all_orders(h1, p)
        random_shuffle_all_orders(h2, p)

        G1=set(h1.get_edges())
        G2=set(h2.get_edges())

        ###Computation of metrics (matrix similarity, all I metrics)
        IA=NMI_Bulk(N,G1,G2)
        IB= NMIaligned(G1,G2,partition=None)
       
        IAsim.append(IA),IBsim.append(IB)
        
        if count in [0, 2, 4, 14]:
            D = 2
            dmax = h1.max_size()
            M = -1*np.ones((dmax-1, dmax-1)) ###IMPORTANT: initialize with -1

            """2-step filling process of M"""
            for i in range(M.shape[0]):
                for j in range(i, M.shape[1]):
                    e1=h1.get_edges(size=D+j)
                    e2=h2.get_edges(size=D+i) 
                    M[j,i] = hNMI_project(N, e1, e2, D+i)
            for i in range(M.shape[0]):
                for j in range(i+1, M.shape[1]):
                    e1=h2.get_edges(size=D+j)
                    e2=h1.get_edges(size=D+i) 
                    M[i,j] = hNMI_project(N, e1, e2, D+i)
           
            Msim.append(M)
            
    IAs.append(IAsim), IBs.append(IBsim), Ms.append(Msim)
    
## saving curves and matrices
#np.save(f"results_{system}_NMIbulk_nsim-{nsim}", IAs)
#np.save(f"results_{system}_NMIalign_nsim-{nsim}", IBs)
#np.save(f"results_{system}_heatmaps_nsim-{nsim}", Ms)




"""
    (b)
"""


N,d,E=100,6,15625 
Edict={l: 5**(7-l) for l in range(2,7)} 

system=f"Hrandom-N-{N}-lmax-{d}-sparse"

print(Edict)
ps=np.linspace(0,1,15)

IAs=[]
IBs=[]
ICs=[]
Ms=[]
ys,xs=[],[]

nsim=10
for n_sim in range(nsim):
    
    Msim=[]
    IBsim,IAsim=[],[]

    # attack
    for count,p in enumerate(ps):

        ####Initization before randomization
        h1=random_hypergraph(N,Edict)
        h2=h1.copy()

        ## Randomization of hypergraphs
        random_shuffle_all_orders(h1, p)
        random_shuffle_all_orders(h2, p)

        G1=set(h1.get_edges())
        G2=set(h2.get_edges())

        ###Computation of metrics (matrix similarity, all I metrics)
        IA=NMI_Bulk(N,G1,G2)
        IB=NMIaligned(G1,G2,partition=None)
        
        IAsim.append(IA),IBsim.append(IB)
        
        if count in [0, 2, 4, 14]:
            D = 2
            dmax = h1.max_size()
            M = -1*np.ones((dmax-1, dmax-1)) ###IMPORTANT: initialize with -1

            """2-step filling process of M"""
            for i in range(M.shape[0]):
                for j in range(i, M.shape[1]):
                    e1=h1.get_edges(size=D+j)
                    e2=h2.get_edges(size=D+i) 
                    M[j,i] = hNMI_project(N, e1, e2, D+i)
            for i in range(M.shape[0]):
                for j in range(i+1, M.shape[1]):
                    e1=h2.get_edges(size=D+j)
                    e2=h1.get_edges(size=D+i) 
                    M[i,j] = hNMI_project(N, e1, e2, D+i)
           
            Msim.append(M)
            
    IAs.append(IAsim), IBs.append(IBsim), Ms.append(Msim)
    
## saving curves and matrices
#np.save(f"results_{system}_NMIbulk_nsim-{nsim}", IAs)
#np.save(f"results_{system}_NMIalign_nsim-{nsim}", IBs)
#np.save(f"results_{system}_heatmaps_nsim-{nsim}", Ms)
