from Digraph import Digraph


class DirectedDFS:
    """Given a digraph and sources to find which point can sources arrived.
     uses depth-first search to solve this problem.
    """
    def __init__(self, G, sources):
        self._marked = [False] * G.V
        for s in sources:
            if not self._marked[s]:
                self.dfs(G, s)

    def dfs(self, G, v):
        self._marked[v] = True
        for w in G.adj(v):
            if not self._marked[w]:
                self.dfs(G, w)

    def marked(self, v):
        return self._marked[v]

# unittest
if __name__ == '__main__':
    with open('tinyDG.txt', 'r') as g:
        V = int(g.readline().split()[0])
        E = int(g.readline().split()[0])
        DG = Digraph(V)
        for e in range(E):
            v, w = g.readline().split()
            DG.add_edge(v, w)

    # sources: 1, 2, 6
    # reachable: 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12   # don't have 7
    sources = [1, 2, 6]
    reachable = DirectedDFS(DG, sources)
    for i in range(V):
        if i == 7:
            assert reachable.marked(i) is False
        else:
            assert reachable.marked(i) is True
