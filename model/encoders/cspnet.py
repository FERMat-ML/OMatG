from diffcsp.pl_modules.cspnet import CSPNet

class CSPNetEncoder(CSPNet):

    def __init__(self, *args, **kwargs):
        super(CSPNetEncoder, self).__init__(*args, **kwargs)

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph):

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)


        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)

        return node_features, frac_diff
