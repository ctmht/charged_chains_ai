from torch import nn
import spdlayers
import torch

from model.trunk_net_module import TrunkNetModule


class PolymerTransformerModule(nn.Module):
    """ Transformer-based sequence processing model (nn.Module subclass) """
    
    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.0,
        mha_layers: int = 2,
        hidden_dims: list[int] = None
    ):
        """
        
        """
        super(PolymerTransformerModule, self).__init__()

        hidden_dims.insert(0, embedding_dim)
        
        self.trunk_net = TrunkNetModule(
            embedding_dim = embedding_dim,
            num_heads = num_heads,
            dropout = dropout,
            mha_layers = mha_layers,
            hidden_dims = hidden_dims
        )

        # Gyration tensor eigenvalues
        gte_head_dim = 16
        self.gte_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], gte_head_dim),   # transform pooled embeddings
            nn.LayerNorm(gte_head_dim),                 # add (gradient) stability
            nn.Tanh()                                   # bound activations, can be negative
        )
        self.gte_oem_head = nn.Linear(gte_head_dim, 3)
        self.gte_oev_head = nn.Sequential(
            # output the 6 independent entries of covar + enforce symmetric positive definiteness
            nn.Linear(gte_head_dim, spdlayers.in_shape_from(3)),
            spdlayers.Cholesky(3, positive = 'Softplus')
        )

        # Potential energy
        poteng_head_dim = 8
        self.poteng_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], poteng_head_dim),# transform pooled embeddings
            nn.LayerNorm(poteng_head_dim),              # add (gradient) stability
            nn.Sigmoid()                                # make outputs positive
        )
        self.poteng_mean_head = nn.Linear(poteng_head_dim, 1)
        self.poteng_var_head = nn.Sequential(
            nn.Linear(poteng_head_dim, 1),
            nn.Softplus()
        )

        # Mean neighbour counts
        nbc_head_dim = 16
        self.nbc_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], nbc_head_dim),   # transform pooled embeddings
            nn.LayerNorm(nbc_head_dim),                 # add (gradient) stability
            nn.Sigmoid(),                               # make outputs positive
        )
        self.nbc_mean_head = nn.Linear(nbc_head_dim, 4)
        self.nbc_var_head = nn.Linear(nbc_head_dim, 4)
        
        # self.apply(self._init_weights)
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param.data)
            else:
                nn.init.normal_(param.data, mean = 0.0, std = 1.0)
    
    def _init_weights(
        self,
        module
    ):
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_uniform_(param.data)
                else:
                    torch.nn.init.zeros_(param.data)
        if isinstance(module, nn.MultiheadAttention):
            for param in module.parameters():
                if param.dim() > 1:
                    torch.nn.init.xavier_normal_(param.data)
                else:
                    torch.nn.init.normal_(param.data, mean = 0, std = 1)
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                torch.nn.init.zeros_(param.data)

        

    def forward(
        self,
        data
    ):
        # def _reshape(_sixvec):
        #     """ Make a batch_size 3 x 3 symmetric matrices from a tensor of shape (batch_size, 6) """
        #     result = torch.zeros(_sixvec.size(0), 3, 3, device=_sixvec.device, dtype=_sixvec.dtype)
        #     triu_indices = torch.triu_indices(3, 3, device=_sixvec.device)
        #     result[:, triu_indices[0], triu_indices[1]] = _sixvec
        #     result = result + result.transpose(1, 2) - torch.diag_embed(torch.diagonal(result, dim1=1, dim2=2))
        #     return result
        
        pooled_emb, attn_maps = self.trunk_net(data)

        # Gyration tensor ordered eigenvalues mean + covariance heads
        _gte_out = self.gte_head(pooled_emb)
        _gte_oem_out = self.gte_oem_head(_gte_out)
        _gte_oev_out = self.gte_oev_head(_gte_out)

        # Potential energy mean + std dev head
        _poteng_head = self.poteng_head(pooled_emb)
        _poteng_mean_out = self.poteng_mean_head(_poteng_head)
        _poteng_var_out = self.poteng_var_head(_poteng_head)
        
        # Mean neighbour counts head
        _nbc_out = self.nbc_head(pooled_emb)
        _nbc_mean_out = self.nbc_mean_head(_nbc_out)
        _nbc_var_out = torch.square(self.nbc_var_head(_nbc_out))

        return {
            'gte_oem': _gte_oem_out,
            'gte_oev': _gte_oev_out,
            'pe_mean': _poteng_mean_out,
            'pe_var': _poteng_var_out,
            'nbc_mean': _nbc_mean_out,
            'nbc_var': _nbc_var_out,
            'pooled_emb': pooled_emb,
            'attn_maps': attn_maps
        }