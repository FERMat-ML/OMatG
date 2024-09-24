import torch.nn as nn

class Model(nn.module):
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

    def forward(self,x, t, prop=None): # t should probably be embedded beforehand
        x = self.encoder(x)
        x = self.head(x, t, prop)
        return x


if __name__ == "__main__":
    # test model structure
    from encoders.cspnet import CSPNetEncoder
    from heads.cspnet import CSPNetHead
    encoder = CSPNetEncoder(
        hidden_dim = 512,
        latent_dim = 256,
        num_layers = 4,
        max_atoms = 100,
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
    )
    head = CSPNetHead(
        hidden_dim = 512,
        latent_dim = 256,
        num_layers = 1,
        max_atoms = 100,
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
        )

    model = Model(encoder, head)
    result = model(???)
    print (results)

        
