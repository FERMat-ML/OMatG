from diffcsp.pl_modules.cspnet import CSPNet
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from einops import rearrange, repeat

from diffcsp.common.data_utils import lattice_params_to_matrix_torch, get_pbc_distances, radius_graph_pbc, frac_to_cart_coords, repeat_blocks

from diffcsp.pl_modules.cspnet import CSPLayer
from diffcsp.pl_modules.cspnet import SinusoidsEmbedding

from .encoder import Encoder
from omg.globals import MAX_ATOM_NUM 


class CSPNetFull(Encoder, CSPNet):

    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = MAX_ATOM_NUM,
        act_fn = 'silu',
        dis_emb = 'sin',
        num_freqs = 128,
        edge_style = 'fc',
        cutoff = 7.0,
        max_neighbors = 20,
        ln = True,
        ip = True,
        smooth = False,
        pred_type = True,
        pred_scalar = False
    ):

        super().__init__()

        self.ip = ip
        self.smooth = smooth
        if self.smooth:
            self.node_embedding = nn.Linear(max_atoms, hidden_dim)
        else:
            self.node_embedding = nn.Embedding(max_atoms, hidden_dim)
        self.atom_latent_emb = nn.Linear(hidden_dim + latent_dim, hidden_dim)
        if act_fn == 'silu':
            self.act_fn = nn.SiLU()
        if dis_emb == 'sin':
            self.dis_emb = SinusoidsEmbedding(n_frequencies = num_freqs)
        elif dis_emb == 'none':
            self.dis_emb = None
        for i in range(0, num_layers):
            self.add_module(
                "csp_layer_%d" % i, CSPLayer(hidden_dim, self.act_fn, self.dis_emb, ln=ln, ip=ip)
            )
        self.num_layers = num_layers
        self.coord_out = nn.Linear(hidden_dim, 3, bias = False)
        self.coord_out_2 = nn.Linear(hidden_dim, 3, bias = False)
        self.lattice_out = nn.Linear(hidden_dim, 9, bias = False)
        self.lattice_out_2 = nn.Linear(hidden_dim, 9, bias = False)
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.pred_type = pred_type
        self.max_atoms = max_atoms
        self.ln = ln
        self.edge_style = edge_style
        if self.ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)
        if self.pred_type:
            self.type_out = nn.Linear(hidden_dim, self.max_atoms)
            self.type_out_2 = nn.Linear(hidden_dim, self.max_atoms)
        self.pred_scalar = pred_scalar
        if self.pred_scalar:
            self.scalar_out = nn.Linear(hidden_dim, 1)

    def _convert_inputs(self, x, **kwargs):
        atom_types = x.species
        frac_coords = x.pos
        lattices = x.cell
        num_atoms = x.n_atoms
        node2graph = x.batch
        return atom_types, frac_coords, lattices, num_atoms, node2graph


    def _forward(self, atom_types, frac_coords, lattices, num_atoms, node2graph, t=0.0, prop=None):
        # taken from DiffCSP with additional output layers included

        edges, frac_diff = self.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        if self.smooth:
            node_features = self.node_embedding(atom_types)
        else:
            node_features = self.node_embedding(atom_types - 1)

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
        lattice_eta = self.lattice_out_2(graph_features)
        lattice_b = lattice_b.view(-1, 3, 3)
        lattice_eta = lattice_eta.view(-1, 3, 3)
        if self.ip:
            lattice_b = torch.einsum('bij,bjk->bik', lattice_b, lattices)
            lattice_eta = torch.einsum('bij,bjk->bik', lattice_eta, lattices)
        if self.pred_type:
            type_b = self.type_out(node_features)
            type_eta = self.type_out_2(node_features)
            data = Data(
                species_b = type_b,
                species_eta = type_eta,
                pos_b = coord_b,
                pos_eta = coord_eta,
                cell_b = lattice_b,
                cell_eta = lattice_eta
            )
            return data
        data = Data( pos_b=coord_b, pos_eta=coord_eta, cell_b=lattice_b, cell_eta=lattice_eta)
        return data

    def _convert_outputs(self, x, **kwargs):
        return x
