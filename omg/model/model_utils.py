import torch.nn as nn
import torch
import math

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AdapterModule(nn.Module):
    def __init__(self, input_dim, am_hidden_dim, property_dim):  # input_dim=dim of hidden variable, and property_dim=property embedding
        super(AdapterModule, self).__init__()
        # Adapter is a two-layer MLP
        self.adapter_fc1 = nn.Linear(property_dim, am_hidden_dim)
        # TODO: Make choosing activation possible
        self.adapter_relu = nn.ReLU() # not sure if matergen does this but it seems a good idea
        self.adapter_fc2 = nn.Linear(am_hidden_dim, input_dim)
        
        # Mixin is a zero-initialized linear layer without bias
        self.mixin = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.zeros_(self.mixin.weight)  # Zero-initialize mixin weights

    # Add activation, change loop to multiplication so we can nullify property within a batch
    def forward(self, H_L, property_emb, property_indicator, num_atoms):
        f_adapter_L = self.adapter_fc1(property_emb)
        f_adapter_L = self.adapter_relu(f_adapter_L)
        f_adapter_L = self.adapter_fc2(f_adapter_L)
        # add another activation?
        f_adapter_L = self.adapter_relu(f_adapter_L)
            
        f_mixin_L = self.mixin(f_adapter_L)

        result = property_indicator.view(-1,1) * f_mixin_L
        repeated_result = result.repeat_interleave(num_atoms, dim=0)
        
        H_prime_L = H_L + torch.squeeze(repeated_result)

        return H_prime_L

def PropIndicator(batch_size, p_uncond):
    property_indicator = torch.bernoulli(torch.ones(batch_size)*(1.-p_uncond))
    return property_indicator

