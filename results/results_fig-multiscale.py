from extrafunctions import *

N = 1000
layers = [2,3,4,5,6,7,8,9,10]
layer_sizes = {d:2**(12-d) for d in range(2,11)}
B1, B2 = 50, 50
pwithin = np.linspace(0,1,50) # the size of this array will determine the size of the heatmap
num_shuffles=0
max_shuffle=500
nsim=10
Ms=[]

for sim in range(nsim):

    n = len(pwithin)
    M = -1*np.ones((n, n)) # initialize with -1 (best practice)

    """2-step filling process of M"""

    for i in range(M.shape[0]):
        for j in range(i, M.shape[1]):

            G1,b1 = block_hypergraph(N,B1,layers,layer_sizes,pwithin[i])
            G2,b2 = block_hypergraph(N,B2,layers,layer_sizes,pwithin[j],shuffle_b=num_shuffles)

            M[i,j] = NMIcross(N,G1,G2,partition=b1)

    for i in range(M.shape[0]):
        for j in range(i+1, M.shape[1]):

            G1,b1 = block_hypergraph(N,B1,layers,layer_sizes,pwithin[i])
            G2,b2 = block_hypergraph(N,B2,layers,layer_sizes,pwithin[j],shuffle_b=num_shuffles)

            M[j,i] = NMIcross(N,G1,G2,partition=b1)
            
    Ms.append(M)
    
M = np.mean(Ms, axis=0) # avg matrix

#np.save(f"results_p1p2_heatmap_{len(pwithin)}x{len(pwithin)}_nsim-{nsim}.npy",M)
