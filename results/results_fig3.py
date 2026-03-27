from extrafunctions import *

matrices_NMIs=[]


"""
                first model
"""

N,d,E=100,7,10

ps=np.linspace(0,1,10)
ps=[0]

IAs=[]
IBs=[]
ICs=[]
Ms=[]

   
#####Initialization of 3 equal block-nested hypergraphs

"""we need three reference nested hypergraphs for blocks [2,3], [4,5], and [6,7]"""
hA=nested_hypergraph(N,3,350)
hB=nested_hypergraph(N,5,35)
hC=nested_hypergraph(N,7,10)
"""now we create an empty graph and put these respective blocks there"""
h1=random_hypergraph(N, {0:0})
for blockA, blockB, blockC in zip([2,3], [4,5], [6,7]):
    h1.add_edges(hA.get_edges(size=blockA))
    h1.add_edges(hB.get_edges(size=blockB))
    h1.add_edges(hC.get_edges(size=blockC))  

"""second hypergraph is just a copy"""
h2=h1.copy()

G1=set(h1.get_edges())
G2=set(h2.get_edges())

###Computation of metrics (matrix similarity, all I metrics)
#IA=NMI_Bulk(N,G1,G2)
IB= NMIaligned(N,G1,G2,partition=None)
IC= NMIcross(N,G1,G2)
#IAs.append(IA)
IBs.append(IB)
ICs.append(IC)
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

Ms.append(M)

matrices_NMIs.append((M,IB,IC))

"""
            second model
"""

N,d,E=100,7,10

ps=np.linspace(0,1,10)
ps=[0]

IAs=[]
IBs=[]
ICs=[]
Ms=[]
    
#####Initialization of 2 equal block-nested hypergraphs with "hole" in 4-5

"""we need two reference nested hypergraphs for blocks [2,3] and [4,5]"""
hA=nested_hypergraph(N,3,350)
hB=nested_hypergraph(N,5,35)
"""now we create an empty graph and put these respective blocks there"""
h1=random_hypergraph(N, {0:0})
for blockA, blockB in zip([2,3], [4,5]):
    h1.add_edges(hA.get_edges(size=blockA))
    h1.add_edges(hB.get_edges(size=blockB))
## we can create the second hypergraph here, as a copy of the first
h2=h1.copy()
"""now let's create the holed-block"""
"""we need two reference hypergraphs for the hole in block [6,7]"""
"""layer 7 of one is projected to layer 6 of the other, and vice-versa"""
hC1=nested_hypergraph(N,7,10)
hC2=nested_hypergraph(N,7,10)
for layer6, layer7 in zip([6], [7]):
    h1.add_edges(hC1.get_edges(size=layer6))
    h2.add_edges(hC1.get_edges(size=layer7))
    h1.add_edges(hC2.get_edges(size=layer7))
    h2.add_edges(hC2.get_edges(size=layer6))     

G1=set(h1.get_edges())
G2=set(h2.get_edges())

###Computation of metrics (matrix similarity, all I metrics)
#IA=NMI_Bulk(N,G1,G2)
IB= NMIaligned(N,G1,G2,partition=None)
IC= NMIcross(N,G1,G2)
#IAs.append(IA)
IBs.append(IB)
ICs.append(IC)
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
        
matrices_NMIs.append((M,IB,IC))


"""
                third model
"""

N,d,E=100,5,35

ps=np.linspace(0,1,10)
ps=[0]

IAs=[]
IBs=[]
ICs=[]
Ms=[]
    
#####Initialization of 2 equal block-nested hypergraphs with "hole" in 4-5

"""we need a reference nested hypergraphs for block [2,3]"""
hA=nested_hypergraph(N,3,350)
"""now we create an empty graph and put this block there"""
h1=random_hypergraph(N, {0:0})
for size in [2,3]:
    h1.add_edges(hA.get_edges(size=size))
## we can create the second hypergraph here, as a copy of the first
h2=h1.copy()
"""now we have to create two holed-blocks"""
"""we need four reference hypergraphs"""
"""projecting layers 5 and 4 and vice versa"""
hB1=nested_hypergraph(N,5,35)
hB2=nested_hypergraph(N,5,35)
for layer4, layer5 in zip([4], [5]):
    h1.add_edges(hB1.get_edges(size=layer4))
    h2.add_edges(hB1.get_edges(size=layer5))
    h1.add_edges(hB2.get_edges(size=layer5))
    h2.add_edges(hB2.get_edges(size=layer4))
"""layer 7 of one is projected to layer 6 of the other, and vice-versa"""
hC1=nested_hypergraph(N,7,10)
hC2=nested_hypergraph(N,7,10)
for layer6, layer7 in zip([6], [7]):
    h1.add_edges(hC1.get_edges(size=layer6))
    h2.add_edges(hC1.get_edges(size=layer7))
    h1.add_edges(hC2.get_edges(size=layer7))
    h2.add_edges(hC2.get_edges(size=layer6))  

G1=set(h1.get_edges())
G2=set(h2.get_edges())

###Computation of metrics (matrix similarity, all I metrics)
#IA=NMI_Bulk(N,G1,G2)
IB= NMIaligned(N,G1,G2,partition=None)
IC= NMIcross(N,G1,G2)
#IAs.append(IA)
IBs.append(IB)
ICs.append(IC)
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

Ms.append(M)

matrices_NMIs.append((M,IB,IC))


"""
                fourth model
"""


N,d,E=100,5,35

ps=np.linspace(0,1,10)
ps=[0]

IAs=[]
IBs=[]
ICs=[]
Ms=[]
    
#####Initialization of 2 equal block-nested hypergraphs with "hole" in 4-5

"""we start by initializing the hypergraphs"""
h1=random_hypergraph(N, {0:0}) # empty
h2=h1.copy()

"""now we have to create three holed-blocks, six reference hypergraphs"""
"""projecting layers 3 and 2 and vice versa"""
hA1=nested_hypergraph(N,3,350)
hA2=nested_hypergraph(N,3,350)
for layer2, layer3 in zip([2], [3]):
    h1.add_edges(hA1.get_edges(size=layer2))
    h2.add_edges(hA1.get_edges(size=layer3))
    h1.add_edges(hA2.get_edges(size=layer3))
    h2.add_edges(hA2.get_edges(size=layer2))
"""projecting layers 5 and 4 and vice versa"""
hB1=nested_hypergraph(N,5,35)
hB2=nested_hypergraph(N,5,35)
for layer4, layer5 in zip([4], [5]):
    h1.add_edges(hB1.get_edges(size=layer4))
    h2.add_edges(hB1.get_edges(size=layer5))
    h1.add_edges(hB2.get_edges(size=layer5))
    h2.add_edges(hB2.get_edges(size=layer4))
"""layer 7 of one is projected to layer 6 of the other, and vice-versa"""
hC1=nested_hypergraph(N,7,10)
hC2=nested_hypergraph(N,7,10)
for layer6, layer7 in zip([6], [7]):
    h1.add_edges(hC1.get_edges(size=layer6))
    h2.add_edges(hC1.get_edges(size=layer7))
    h1.add_edges(hC2.get_edges(size=layer7))
    h2.add_edges(hC2.get_edges(size=layer6)) 
    
G1=set(h1.get_edges())
G2=set(h2.get_edges())

###Computation of metrics (matrix similarity, all I metrics)
#IA=NMI_Bulk(N,G1,G2)
IB= NMIaligned(N,G1,G2,partition=None)
IC= NMIcross(N,G1,G2)
#IAs.append(IA)
IBs.append(IB)
ICs.append(IC)
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

Ms.append(M)

matrices_NMIs.append((M,IB,IC))

## save the heatmaps and bars -- complicated data structure requires pickle
with open("results_blocknested_heatmap-NMIalign-NMI-cross.pkl", 'wb') as file:
    pickle.dump(matrices_NMIs, file)
