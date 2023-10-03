fig,ax = fig(5 by 3) #can't remember syntax
num_sims = 10
sim_names = ['NMI','DC-NMI','Jaccard','Meso, B = 2','Meso, B = 10','Meso, B = 100']
epsilons = np.linspace(0.,1.,20)
noise_types = ['TypeI','TypeII','TypeIII']
graph_types = ['ER','BA','SBM_eta_0.1','SBM_eta_0.5','SBM_eta_0.9']
colors = ['blue','red','black','light gray','gray','dark gray']


for row in range(5):
  graph_type = graph_types[row]
  G = #generate graph corresponding to graph_type 
  partitions = [comms(G,2),comms(G,2),comms(G,2),comms(G,2),comms(G,10),comms(G,100)] #comms(G,B) returns partition of G's nodes into B communities using some method. one fixed partition for each of the 6 curves
  for column in range(3):
    noise_type = noise_types[column]
    for curve in range(6):
      if noise_type in ['TypeI','TypeII']:
        noisy_graphs = [[eval(noise_type)(G,eps) for _ in range(num_sims)] for eps in epsilons] #or could be 'exec', cant remember.
      else:
        noisy_graphs = [[eval(noise_type)(G,eps,partition=partitions[curve]) for _ in range(num_sims)] for eps in epsilons] #or could be 'exec', cant remember.
      sim_name = sim_names[curve]
      sim_measure = #function for similarity measure corresponding to sim_name
      color = colors[curve]
      if sim_measure in ['NMI','DC-NMI','Jaccard']:
        means = [np.mean([sim_measure(G,Gnoisy) for Gnoisy in samples]) for samples in noisy_graphs]
        errors = [3*np.std([sim_measure(G,Gnoisy) for Gnoisy in samples])/np.sqrt(len(samples)) for samples in noisy_graphs]
      else:
        means = [np.mean([sim_measure(G,Gnoisy,partition=partitions[curve]) for Gnoisy in samples]) for samples in noisy_graphs]
        errors = [3*np.std([sim_measure(G,Gnoisy,partition=partitions[curve]) for Gnoisy in samples])/np.sqrt(len(samples)) for samples in noisy_graphs]
      ax[row,column].error_bar(epsilons,means,errors,label=sim_name,color=color)
      
      
    
