import torch
from torch import nn
from torch.nn.functional import relu
from transformer_lens.hook_points import HookedRootModule, HookPoint



class MLP(HookedRootModule):

    def __init__(self, n: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.n = n 
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.embed = nn.Linear(in_features=self.n, out_features=self.embed_dim, bias=False)
        self.linear = nn.Linear(in_features=self.embed_dim, out_features=self.model_dim, bias=False)
        self.unembed = nn.Linear(in_features=self.model_dim, out_features=2)
        self.hook_embed = HookPoint()
        self.hook_linear = HookPoint()
        self.hook_unembed = HookPoint()
        # Gives each module a parameter with its name (relative to this root module)
        # Needed for HookPoints to work
        self.setup()

    @classmethod
    def from_config(cls, config):
        n = config['n']
        embed_dim = config['embed_dim']
        model_dims = config['model_dim']
        
        return cls(n, embed_dim, model_dims)
    
    def forward(self, x):
        embed = relu(self.hook_embed(self.embed(x)))
        hidden = relu(self.hook_linear(self.linear(embed)))
        logits = self.hook_unembed(self.unembed(hidden))
        return logits
    