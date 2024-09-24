from diffcsp.pl_modules.cspnet import CSPNet
import torch.nn as nn

MAX_ATOMIC_NUM=100

class CSPNetHead(CSPNet):

    def __init__(self, *args, **kwargs):

        super(CSPNetHead, self).__init__(*args, **kwargs)
        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias = False)
        self.type_out_2 = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    # TODO: Look at outputs
    def forward(self, self, t, node_features, frac_coords, lattices, num_atoms, node2graph, frac_diff = frac_diff):

        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        for i in range(0, self.num_layers):
            node_features = self._modules["csp_layer_%d" % i](node_features, frac_coords, lattices, edges, edge2graph, frac_diff = frac_diff)

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        coord_b = self.coord_out(node_features)
        coord_eta = self.coord_out_2(node_features)

        graph_features = scatter(node_features, node2graph, dim = 0, reduce = 'mean')

        if self.pred_scalar:
            return self.scalar_out(graph_features)

        lattice_b = self.lattice_out(graph_features)
        lattice_b = lattice_b.view(-1, 3, 3)
        lattice_eta = self.lattice_out_2(graph_features)
        lattice_eta = lattice_b.view(-1, 3, 3)
        if self.ip:
            lattice_b = torch.einsum('bij,bjk->bik', lattice_b, lattices)
            lattice_eta = torch.einsum('bij,bjk->bik', lattice_eta, lattices)
        type_b = self.type_out(node_features)
        type_eta = self.type_out_2(node_features)
        return (lattice_b, lattice_eta), (coord_b, coord_eta), (type_b, type_eta)

