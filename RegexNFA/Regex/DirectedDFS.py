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
