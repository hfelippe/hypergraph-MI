import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from functions import *

# ------------------------------------------------------------------------

from collections import defaultdict

"""Priority queue class with updatable priorities.
"""

import heapq

__all__ = ["MappedQueue"]

class _HeapElement:
    __slots__ = ["priority", "element", "_hash"]

    def __init__(self, priority, element):
        self.priority = priority
        self.element = element
        self._hash = hash(element)

    def __lt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority < other
        # assume comparing to another _HeapElement
        if self.priority == other_priority:
            try:
                return self.element < other.element
            except TypeError as err:
                raise TypeError(
                    "Consider using a tuple, with a priority value that can be compared."
                )
        return self.priority < other_priority

    def __gt__(self, other):
        try:
            other_priority = other.priority
        except AttributeError:
            return self.priority > other
        # assume comparing to another _HeapElement
        if self.priority == other_priority:
            try:
                return self.element > other.element
            except TypeError as err:
                raise TypeError(
                    "Consider using a tuple, with a priority value that can be compared."
                )
        return self.priority > other_priority

    def __eq__(self, other):
        try:
            return self.element == other.element
        except AttributeError:
            return self.element == other

    def __hash__(self):
        return self._hash

    def __getitem__(self, indx):
        return self.priority if indx == 0 else self.element[indx - 1]

    def __iter__(self):
        yield self.priority
        try:
            yield from self.element
        except TypeError:
            yield self.element

    def __repr__(self):
        return f"_HeapElement({self.priority}, {self.element})"


class MappedQueue:
    """The MappedQueue class implements a min-heap with removal and update-priority.

    The min heap uses heapq as well as custom written _siftup and _siftdown
    methods to allow the heap positions to be tracked by an additional dict
    keyed by element to position. The smallest element can be popped in O(1) time,
    new elements can be pushed in O(log n) time, and any element can be removed
    or updated in O(log n) time. The queue cannot contain duplicate elements
    and an attempt to push an element already in the queue will have no effect.

    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it adds element removal, lookup, and priority update.

    Parameters
    ----------
    data : dict or iterable

    Examples
    --------

    A `MappedQueue` can be created empty, or optionally, given a dictionary
    of initial elements and priorities.  The methods `push`, `pop`,
    `remove`, and `update` operate on the queue.

    >>> colors_nm = {'red':665, 'blue': 470, 'green': 550}
    >>> q = MappedQueue(colors_nm)
    >>> q.remove('red')
    >>> q.update('green', 'violet', 400)
    >>> q.push('indigo', 425)
    True
    >>> [q.pop().element for i in range(len(q.heap))]
    ['violet', 'indigo', 'blue']

    A `MappedQueue` can also be initialized with a list or other iterable. The priority is assumed
    to be the sort order of the items in the list.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> [q.pop() for i in range(len(q.heap))]
    [50, 916, 1117, 4609]

    An exception is raised if the elements are not comparable.

    >>> q = MappedQueue([100, 'a'])
    Traceback (most recent call last):
    ...
    TypeError: '<' not supported between instances of 'int' and 'str'

    To avoid the exception, use a dictionary to assign priorities to the elements.

    >>> q = MappedQueue({100: 0, 'a': 1 })

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=None):
        """Priority queue class with updatable priorities."""
        if data is None:
            self.heap = list()
        elif isinstance(data, dict):
            self.heap = [_HeapElement(v, k) for k, v in data.items()]
        else:
            self.heap = list(data)
        self.position = dict()
        self._heapify()

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        heapq.heapify(self.heap)
        self.position = {elt: pos for pos, elt in enumerate(self.heap)}
        if len(self.heap) != len(self.position):
            raise AssertionError("Heap contains duplicate elements")

    def __len__(self):
        return len(self.heap)

    def push(self, elt, priority=None):
        """Add an element to the queue."""
        if priority is not None:
            elt = _HeapElement(priority, elt)
        # If element is already in queue, do nothing
        if elt in self.position:
            return False
        # Add element to heap and dict
        pos = len(self.heap)
        self.heap.append(elt)
        self.position[elt] = pos
        # Restore invariant by sifting down
        self._siftdown(0, pos)
        return True

    def pop(self):
        """Remove and return the smallest element in the queue."""
        # Remove smallest element
        elt = self.heap[0]
        del self.position[elt]
        # If elt is last item, remove and return
        if len(self.heap) == 1:
            self.heap.pop()
            return elt
        # Replace root with last element
        last = self.heap.pop()
        self.heap[0] = last
        self.position[last] = 0
        # Restore invariant by sifting up
        self._siftup(0)
        # Return smallest element
        return elt

    def update(self, elt, new, priority=None):
        """Replace an element in the queue with a new one."""
        if priority is not None:
            new = _HeapElement(priority, new)
        # Replace
        pos = self.position[elt]
        self.heap[pos] = new
        del self.position[elt]
        self.position[new] = pos
        # Restore invariant by sifting up
        self._siftup(pos)

    def remove(self, elt):
        """Remove an element from the queue."""
        # Find and remove element
        try:
            pos = self.position[elt]
            del self.position[elt]
        except KeyError:
            # Not in queue
            raise
        # If elt is last item, remove and return
        if pos == len(self.heap) - 1:
            self.heap.pop()
            return
        # Replace elt with last element
        last = self.heap.pop()
        self.heap[pos] = last
        self.position[last] = pos
        # Restore invariant by sifting up
        self._siftup(pos)

    def _siftup(self, pos):
        """Move smaller child up until hitting a leaf.

        Built to mimic code for heapq._siftup
        only updating position dict too.
        """
        heap, position = self.heap, self.position
        end_pos = len(heap)
        startpos = pos
        newitem = heap[pos]
        # Shift up the smaller child until hitting a leaf
        child_pos = (pos << 1) + 1  # start with leftmost child position
        while child_pos < end_pos:
            # Set child_pos to index of smaller child.
            child = heap[child_pos]
            right_pos = child_pos + 1
            if right_pos < end_pos:
                right = heap[right_pos]
                if not child < right:
                    child = right
                    child_pos = right_pos
            # Move the smaller child up.
            heap[pos] = child
            position[child] = pos
            pos = child_pos
            child_pos = (pos << 1) + 1
        # pos is a leaf position. Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if not newitem < parent:
                break
            heap[pos] = parent
            position[parent] = pos
            pos = parent_pos
        heap[pos] = newitem
        position[newitem] = pos

    def _siftdown(self, start_pos, pos):
        """Restore invariant. keep swapping with parent until smaller.

        Built to mimic code for heapq._siftdown
        only updating position dict too.
        """
        heap, position = self.heap, self.position
        newitem = heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > start_pos:
            parent_pos = (pos - 1) >> 1
            parent = heap[parent_pos]
            if not newitem < parent:
                break
            heap[pos] = parent
            position[parent] = pos
            pos = parent_pos
        heap[pos] = newitem
        position[newitem] = pos


def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    directed = G.is_directed()
    N = G.number_of_nodes()

    # Count edges (or the sum of edge-weights for weighted graphs)
    m = G.size(weight)
    q0 = 1 / m

    # Calculate degrees (notation from the papers)
    # a : the fraction of (weighted) out-degree for each node
    # b : the fraction of (weighted) in-degree for each node
    if directed:
        a = {node: deg_out * q0 for node, deg_out in G.out_degree(weight=weight)}
        b = {node: deg_in * q0 for node, deg_in in G.in_degree(weight=weight)}
    else:
        a = b = {node: deg * q0 * 0.5 for node, deg in G.degree(weight=weight)}

    # this preliminary step collects the edge weights for each node pair
    # It handles multigraph and digraph and works fine for graph.
    dq_dict = defaultdict(lambda: defaultdict(float))
    for u, v, wt in G.edges(data=weight, default=1):
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt

    # now scale and subtract the expected edge-weights term
    for u, nbrdict in dq_dict.items():
        for v, wt in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])

    # Use -dq to get a max_heap instead of a min_heap
    # dq_heap holds a heap for each node's neighbors
    dq_heap = {u: MappedQueue({(u, v): -dq for v, dq in dq_dict[u].items()}) for u in G}
    # H -> all_dq_heap holds a heap with the best items for each node
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])

    # Initialize single-node communities
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()

    # Merge the two communities that lead to the largest modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        # Remove best merge from row u heap
        dq_heap[u].pop()
        # Push new row max onto H
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        # If this element was also at the root of row v, we need to remove the
        # duplicate entry from H
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            # Remove best merge from row v heap
            dq_heap[v].remove((v, u))
            # Push new row max onto H
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            # Duplicate wasn't in H, just remove from row v heap
            dq_heap[v].remove((v, u))

        # Perform merge
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]

        # Get neighbor communities connected to the merged communities
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        # Update dq for merge of u into v
        for w in all_nbrs:
            # Calculate new dq value
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:  # w in u_nbrs
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            # Update rows v and w
            for row, col in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                # Update dict for v,w only (u is removed below)
                dq_dict[row][col] = dq_vw
                # Save old max of per-row heap
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (row, col)
                d_negdq = -dq_vw
                # Save old value for finding heap index
                if w in v_nbrs:
                    # Update existing element in per-row heap
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    # We're creating a new nonzero element, add to heap
                    dq_heap_row.push(d, priority=d_negdq)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d, priority=d_negdq)
                else:
                    # We've updated an entry in this row, has the max changed?
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)

        # Remove row/col u from dq_dict matrix
        for w in dq_dict[u]:
            # Remove from dict
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            # Remove from heaps if we haven't already
            if w != v:
                # Remove both row and column
                for row, col in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    # Check if replaced dq is row max
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        # Only update per-row heap
                        dq_heap_row.remove(d_old)

        del dq_dict[u]
        # Mark row u as deleted, but keep placeholder
        dq_heap[u] = MappedQueue()
        # Merge u into v and update a
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0

        yield communities.values()

def greedy_modularity_communities(
    G,
    weight=None,
    resolution=1,
    cutoff=1,
    best_n=None,
):
    if (cutoff < 1) or (cutoff > G.number_of_nodes()):
        raise ValueError(f"cutoff must be between 1 and {len(G)}. Got {cutoff}.")
    if best_n is not None:
        if (best_n < 1) or (best_n > G.number_of_nodes()):
            raise ValueError(f"best_n must be between 1 and {len(G)}. Got {best_n}.")
        if best_n < cutoff:
            raise ValueError(f"Must have best_n >= cutoff. Got {best_n} < {cutoff}")
        if best_n == 1:
            return [set(G)]
    else:
        best_n = G.number_of_nodes()

    # retrieve generator object to construct output
    community_gen = _greedy_modularity_communities_generator(
        G, weight=weight, resolution=resolution
    )

    # construct the first best community
    communities = next(community_gen)

    # continue merging communities until one of the breaking criteria is satisfied
    while len(communities) > cutoff:
        try:
            dq = next(community_gen)
        # StopIteration occurs when communities are the connected components
        except StopIteration:
            communities = sorted(communities, key=len, reverse=True)
            # if best_n requires more merging, merge big sets for highest modularity
            while len(communities) > best_n:
                comm1, comm2, *rest = communities
                communities = [comm1 ^ comm2]
                communities.extend(rest)
            return communities

        # keep going unless max_mod is reached or best_n says to merge more
        if dq < 0 and len(communities) <= best_n:
            break
        communities = next(community_gen)

    return sorted(communities, key=len, reverse=True)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

path = "data/"

N,kavg,B,eta = 10000,10,2,0.1
Gset_0,partition = gen_SBM_set(N,kavg,B,eta)
G0nx = graph_Gset(Gset_0)
n_sims = 10

print(f"generated SBM w/ eta={eta}")

part = greedy_modularity_communities(G0nx,cutoff=2) #can use SBM with fixed B = 2 instead
partition_B2 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B2[i] = c
part = greedy_modularity_communities(G0nx,cutoff=10) #can use SBM with fixed B = 10 instead
partition_B10 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B10[i] = c
part = greedy_modularity_communities(G0nx,cutoff=100) #can use SBM with fixed B = 100 instead
partition_B100 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B100[i] = c
noises = np.linspace(0,1,20)
noisy_graphs_t1 = [[typeI(Gset_0, eps) for trial in range(n_sims)]for eps in noises]
noisy_graphs_t2 = [[typeII(Gset_0, eps) for trial in range(n_sims)] for eps in noises]
noisy_graphs_t3 = [[typeIII(Gset_0, partition, eps) for trial in range(n_sims)] for eps in noises]

print(f"generated partitions B=2,10,100")

# type I
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t1]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t1]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t1]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
# metrics
np.savetxt(path + "jacc_sbm_eta_01_typeI.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_01_typeI.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_01_typeI.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_01_typeI.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_01_typeI.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_01_typeI.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_01_typeI.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_01_typeI.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_01_typeI.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_01_typeI.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_01_typeI.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_01_typeI.txt", meso100_err)

print("computed and saved type I")

# type II
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t2]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t2]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t2]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
# metrics
np.savetxt(path + "jacc_sbm_eta_01_typeII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_01_typeII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_01_typeII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_01_typeII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_01_typeII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_01_typeII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_01_typeII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_01_typeII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_01_typeII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_01_typeII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_01_typeII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_01_typeII.txt", meso100_err)

print("computed and saved type II")

# type III
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t3]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t3]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t3]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
# metrics
np.savetxt(path + "jacc_sbm_eta_01_typeIII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_01_typeIII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_01_typeIII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_01_typeIII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_01_typeIII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_01_typeIII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_01_typeIII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_01_typeIII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_01_typeIII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_01_typeIII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_01_typeIII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_01_typeIII.txt", meso100_err)

print("computed and saved type III")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

N,kavg,B,eta = 10000,10,2,0.5
Gset_0,partition = gen_SBM_set(N,kavg,B,eta)
G0nx = graph_Gset(Gset_0)
n_sims = 10

print(f"generated SBM w/ eta={eta}")

part = greedy_modularity_communities(G0nx,cutoff=2) #can use SBM with fixed B = 2 instead
partition_B2 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B2[i] = c
part = greedy_modularity_communities(G0nx,cutoff=10) #can use SBM with fixed B = 10 instead
partition_B10 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B10[i] = c
part = greedy_modularity_communities(G0nx,cutoff=100) #can use SBM with fixed B = 100 instead
partition_B100 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B100[i] = c
noises = np.linspace(0,1,20)
noisy_graphs_t1 = [[typeI(Gset_0, eps) for trial in range(n_sims)]for eps in noises]
noisy_graphs_t2 = [[typeII(Gset_0, eps) for trial in range(n_sims)] for eps in noises]
noisy_graphs_t3 = [[typeIII(Gset_0, partition, eps) for trial in range(n_sims)] for eps in noises]

print(f"generated partitions B=2,10,100")

# type I
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t1]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t1]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t1]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
# metrics
np.savetxt(path + "jacc_sbm_eta_05_typeI.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_05_typeI.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_05_typeI.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_05_typeI.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_05_typeI.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_05_typeI.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_05_typeI.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_05_typeI.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_05_typeI.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_05_typeI.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_05_typeI.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_05_typeI.txt", meso100_err)

print("computed and saved type I")

# type II
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t2]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t2]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t2]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
# metrics
np.savetxt(path + "jacc_sbm_eta_05_typeII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_05_typeII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_05_typeII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_05_typeII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_05_typeII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_05_typeII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_05_typeII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_05_typeII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_05_typeII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_05_typeII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_05_typeII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_05_typeII.txt", meso100_err)

print("computed and saved type II")

# type III
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t3]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t3]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t3]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
# metrics
np.savetxt(path + "jacc_sbm_eta_05_typeIII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_05_typeIII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_05_typeIII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_05_typeIII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_05_typeIII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_05_typeIII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_05_typeIII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_05_typeIII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_05_typeIII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_05_typeIII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_05_typeIII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_05_typeIII.txt", meso100_err)

print("computed and saved type III")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

N,kavg,B,eta = 10000,10,2,0.9
Gset_0,partition = gen_SBM_set(N,kavg,B,eta)
G0nx = graph_Gset(Gset_0)
n_sims = 10

print(f"generated SBM w/ eta={eta}")

part = greedy_modularity_communities(G0nx,cutoff=2) #can use SBM with fixed B = 2 instead
partition_B2 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B2[i] = c
part = greedy_modularity_communities(G0nx,cutoff=10) #can use SBM with fixed B = 10 instead
partition_B10 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B10[i] = c
part = greedy_modularity_communities(G0nx,cutoff=100) #can use SBM with fixed B = 100 instead
partition_B100 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B100[i] = c
noises = np.linspace(0,1,20)
noisy_graphs_t1 = [[typeI(Gset_0, eps) for trial in range(n_sims)]for eps in noises]
noisy_graphs_t2 = [[typeII(Gset_0, eps) for trial in range(n_sims)] for eps in noises]
noisy_graphs_t3 = [[typeIII(Gset_0, partition, eps) for trial in range(n_sims)] for eps in noises]

print(f"generated partitions B=2,10,100")

# type I
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t1]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t1]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t1]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
# metrics
np.savetxt(path + "jacc_sbm_eta_09_typeI.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_09_typeI.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_09_typeI.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_09_typeI.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_09_typeI.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_09_typeI.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_09_typeI.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_09_typeI.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_09_typeI.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_09_typeI.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_09_typeI.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_09_typeI.txt", meso100_err)

print("computed and saved type I")

# type II
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t2]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t2]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t2]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
# metrics
np.savetxt(path + "jacc_sbm_eta_09_typeII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_09_typeII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_09_typeII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_09_typeII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_09_typeII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_09_typeII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_09_typeII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_09_typeII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_09_typeII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_09_typeII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_09_typeII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_09_typeII.txt", meso100_err)

print("computed and saved type II")

# type III
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t3]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t3]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t3]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
# metrics
np.savetxt(path + "jacc_sbm_eta_09_typeIII.txt", Jacs)
np.savetxt(path + "nmi_sbm_eta_09_typeIII.txt", NMI_regs)
np.savetxt(path + "dcnmi_sbm_eta_09_typeIII.txt", NMI_DCs)
np.savetxt(path + "meso2_sbm_eta_09_typeIII.txt", meso2)
np.savetxt(path + "meso10_sbm_eta_09_typeIII.txt", meso10)
np.savetxt(path + "meso100_sbm_eta_09_typeIII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_sbm_eta_09_typeIII.txt", Jacs_err)
np.savetxt(path + "err_nmi_sbm_eta_09_typeIII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_sbm_eta_09_typeIII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_sbm_eta_09_typeIII.txt", meso2_err)
np.savetxt(path + "err_meso10_sbm_eta_09_typeIII.txt", meso10_err)
np.savetxt(path + "err_meso100_sbm_eta_09_typeIII.txt", meso100_err)

print("computed and saved type III")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# ER
N = 10000
p = 1 / N
G0nx = nx.fast_gnp_random_graph(n=N, p=5/N, seed=None, directed=False)
Gset_0 = set(G0nx.edges())
n_sims = 10

print(f"generated ER")

part = greedy_modularity_communities(G0nx,cutoff=2) #can use SBM with fixed B = 2 instead
partition_B2 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B2[i] = c
part = greedy_modularity_communities(G0nx,cutoff=10) #can use SBM with fixed B = 10 instead
partition_B10 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B10[i] = c
part = greedy_modularity_communities(G0nx,cutoff=100) #can use SBM with fixed B = 100 instead
partition_B100 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B100[i] = c
noises = np.linspace(0,1,20)
noisy_graphs_t1 = [[typeI(Gset_0, eps) for trial in range(n_sims)]for eps in noises]
noisy_graphs_t2 = [[typeII(Gset_0, eps) for trial in range(n_sims)] for eps in noises]
noisy_graphs_t3 = [[typeIII(Gset_0, partition, eps) for trial in range(n_sims)] for eps in noises]

print(f"generated partitions B=2,10,100")

# type I
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t1]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t1]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t1]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
# metrics
np.savetxt(path + "jacc_er_typeI.txt", Jacs)
np.savetxt(path + "nmi_er_typeI.txt", NMI_regs)
np.savetxt(path + "dcnmi_er_typeI.txt", NMI_DCs)
np.savetxt(path + "meso2_er_typeI.txt", meso2)
np.savetxt(path + "meso10_er_typeI.txt", meso10)
np.savetxt(path + "meso100_er_typeI.txt", meso100)
# errors
np.savetxt(path + "err_jacc_er_typeI.txt", Jacs_err)
np.savetxt(path + "err_nmi_er_typeI.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_er_typeI.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_er_typeI.txt", meso2_err)
np.savetxt(path + "err_meso10_er_typeI.txt", meso10_err)
np.savetxt(path + "err_meso100_er_typeI.txt", meso100_err)

print("computed and saved type I")

# type II
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t2]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t2]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t2]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
# metrics
np.savetxt(path + "jacc_er_typeII.txt", Jacs)
np.savetxt(path + "nmi_er_typeII.txt", NMI_regs)
np.savetxt(path + "dcnmi_er_typeII.txt", NMI_DCs)
np.savetxt(path + "meso2_er_typeII.txt", meso2)
np.savetxt(path + "meso10_er_typeII.txt", meso10)
np.savetxt(path + "meso100_er_typeII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_er_typeII.txt", Jacs_err)
np.savetxt(path + "err_nmi_er_typeII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_er_typeII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_er_typeII.txt", meso2_err)
np.savetxt(path + "err_meso10_er_typeII.txt", meso10_err)
np.savetxt(path + "err_meso100_er_typeII.txt", meso100_err)

print("computed and saved type II")

# type III
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t3]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t3]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t3]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
# metrics
np.savetxt(path + "jacc_er_typeIII.txt", Jacs)
np.savetxt(path + "nmi_er_typeIII.txt", NMI_regs)
np.savetxt(path + "dcnmi_er_typeIII.txt", NMI_DCs)
np.savetxt(path + "meso2_er_typeIII.txt", meso2)
np.savetxt(path + "meso10_er_typeIII.txt", meso10)
np.savetxt(path + "meso100_er_typeIII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_er_typeIII.txt", Jacs_err)
np.savetxt(path + "err_nmi_er_typeIII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_er_typeIII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_er_typeIII.txt", meso2_err)
np.savetxt(path + "err_meso10_er_typeIII.txt", meso10_err)
np.savetxt(path + "err_meso100_er_typeIII.txt", meso100_err)

print("computed and saved type III")

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# BA
# BA (for k=10, N/m=2,000)
N = 10000
m = int(.0005 * N)
G0nx = nx.barabasi_albert_graph(n=N, m=m, seed=None)
Gset_0 = set(G0nx.edges())
n_sims = 10

print(f"generated BA")

part = greedy_modularity_communities(G0nx,cutoff=2) #can use SBM with fixed B = 2 instead
partition_B2 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B2[i] = c
part = greedy_modularity_communities(G0nx,cutoff=10) #can use SBM with fixed B = 10 instead
partition_B10 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B10[i] = c
part = greedy_modularity_communities(G0nx,cutoff=100) #can use SBM with fixed B = 100 instead
partition_B100 = np.zeros(N).astype('int')
for c,Set in enumerate(part):
    for i in Set:
        partition_B100[i] = c
noises = np.linspace(0,1,20)
noisy_graphs_t1 = [[typeI(Gset_0, eps) for trial in range(n_sims)]for eps in noises]
noisy_graphs_t2 = [[typeII(Gset_0, eps) for trial in range(n_sims)] for eps in noises]
noisy_graphs_t3 = [[typeIII(Gset_0, partition, eps) for trial in range(n_sims)] for eps in noises]

print(f"generated partitions B=2,10,100")

# type I
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t1]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t1]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t1]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t1]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t1]
# metrics
np.savetxt(path + "jacc_ba_typeI.txt", Jacs)
np.savetxt(path + "nmi_ba_typeI.txt", NMI_regs)
np.savetxt(path + "dcnmi_ba_typeI.txt", NMI_DCs)
np.savetxt(path + "meso2_ba_typeI.txt", meso2)
np.savetxt(path + "meso10_ba_typeI.txt", meso10)
np.savetxt(path + "meso100_ba_typeI.txt", meso100)
# errors
np.savetxt(path + "err_jacc_ba_typeI.txt", Jacs_err)
np.savetxt(path + "err_nmi_ba_typeI.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_ba_typeI.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_ba_typeI.txt", meso2_err)
np.savetxt(path + "err_meso10_ba_typeI.txt", meso10_err)
np.savetxt(path + "err_meso100_ba_typeI.txt", meso100_err)

print("computed and saved type I")

# type II
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t2]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t2]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t2]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t2]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t2]
# metrics
np.savetxt(path + "jacc_ba_typeII.txt", Jacs)
np.savetxt(path + "nmi_ba_typeII.txt", NMI_regs)
np.savetxt(path + "dcnmi_ba_typeII.txt", NMI_DCs)
np.savetxt(path + "meso2_ba_typeII.txt", meso2)
np.savetxt(path + "meso10_ba_typeII.txt", meso10)
np.savetxt(path + "meso100_ba_typeII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_ba_typeII.txt", Jacs_err)
np.savetxt(path + "err_nmi_ba_typeII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_ba_typeII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_ba_typeII.txt", meso2_err)
np.savetxt(path + "err_meso10_ba_typeII.txt", meso10_err)
np.savetxt(path + "err_meso100_ba_typeII.txt", meso100_err)

print("computed and saved type II")

# type III
NMI_regs = [np.mean([graphNMI(N,G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
NMI_DCs = [np.mean([graphDCNMI(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
Jacs = [np.mean([jaccard(G,Gset_0) for G in samples]) for samples in noisy_graphs_t3]
meso2 = [np.mean([mesoNMI(G,Gset_0,partition_B2) for G in samples]) for samples in noisy_graphs_t3]
meso10 = [np.mean([mesoNMI(G,Gset_0,partition_B10) for G in samples]) for samples in noisy_graphs_t3]
meso100 = [np.mean([mesoNMI(G,Gset_0,partition_B100) for G in samples]) for samples in noisy_graphs_t3]
NMI_regs_err = [3*np.std([graphNMI(N,G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
NMI_DCs_err = [3*np.std([graphDCNMI(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
Jacs_err = [3*np.std([jaccard(G,Gset_0) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso2_err = [3*np.std([mesoNMI(G,Gset_0,partition_B2) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso10_err = [3*np.std([mesoNMI(G,Gset_0,partition_B10) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
meso100_err = [3*np.std([mesoNMI(G,Gset_0,partition_B100) for G in samples])/np.sqrt(len(samples)) for samples in noisy_graphs_t3]
# metrics
np.savetxt(path + "jacc_ba_typeIII.txt", Jacs)
np.savetxt(path + "nmi_ba_typeIII.txt", NMI_regs)
np.savetxt(path + "dcnmi_ba_typeIII.txt", NMI_DCs)
np.savetxt(path + "meso2_ba_typeIII.txt", meso2)
np.savetxt(path + "meso10_ba_typeIII.txt", meso10)
np.savetxt(path + "meso100_ba_typeIII.txt", meso100)
# errors
np.savetxt(path + "err_jacc_ba_typeIII.txt", Jacs_err)
np.savetxt(path + "err_nmi_ba_typeIII.txt", NMI_regs_err)
np.savetxt(path + "err_dcnmi_ba_typeIII.txt", NMI_DCs_err)
np.savetxt(path + "err_meso2_ba_typeIII.txt", meso2_err)
np.savetxt(path + "err_meso10_ba_typeIII.txt", meso10_err)
np.savetxt(path + "err_meso100_ba_typeIII.txt", meso100_err)

print("computed and saved type III")

print("finished!")
