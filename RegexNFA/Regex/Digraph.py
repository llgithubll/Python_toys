class Digraph:
    """Construct the directed graph for NFA
    """
    def __init__(self, V):
        self._V = V
        self._E = 0
        self._adj = [[] for i in range(self._V)]

    def add_edge(self, v, w):
        self._adj[int(v)].append(int(w))
        self._E += 1

    def adj(self, v):
        return self._adj[v]

    @property
    def V(self):
        return self._V

    @property
    def E(self):
        return self._E

    @property
    def str_digraph(self):
        s = '{0} vertices, {1} edges\n'.format(str(self._V), str(self._E))
        for v in range(self._V):
            s += '{0}: '.format(str(v))
            for w in self._adj[v]:
                s += '{0} '.format(str(w))
            s += '\n'
        return s

