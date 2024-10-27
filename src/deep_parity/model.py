import torch
from torch import nn
from torch.nn.functional import relu

from transformer_lens.hook_points import HookedRootModule, HookPoint


class MLPv1(HookedRootModule):

    def __init__(self, n: int, embed_dim: int, model_dim: int):
        super().__init__()
        self.n = n 
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.embed = nn.Linear(in_features=self.n, out_features=self.embed_dim, bias=False)
        self.linear1 = nn.Linear(in_features=self.embed_dim, out_features=self.model_dim, bias=False)
        self.linear2 = nn.Linear(in_features=self.model_dim, out_features=self.model_dim, bias=False)
        self.unembed = nn.Linear(in_features=self.model_dim, out_features=2)
        self.hook_embed = HookPoint()
        self.hook_linear1 = HookPoint()
        self.hook_linear2 = HookPoint()
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
        hidden1 = relu(self.hook_linear1(self.linear1(embed)))
        hidden2 = relu(self.hook_linear2(self.linear2(hidden1)))
        logits = self.hook_unembed(self.unembed(hidden2))
        return logits
    


class MLP(HookedRootModule):

    def __init__(self, n: int, layer_sizes: list[int]):
        super().__init__()
        self.n = n
        self.l0_dim, self.l1_dim, self.l2_dim  = layer_sizes
        self.linear0 = nn.Linear(in_features=self.n, out_features=self.l0_dim, bias=False)
        self.linear1 = nn.Linear(in_features=self.l0_dim, out_features=self.l1_dim, bias=True)
        self.linear2 = nn.Linear(in_features=self.l1_dim, out_features=self.l2_dim, bias=True)
        self.unembed = nn.Linear(in_features=self.l2_dim, out_features=2, bias=False)

    @classmethod
    def from_config(cls, config):
        n = config['n']
        embed_dim = config['embed_dim']
        model_dims = config['model_dim']
        
        return cls(n, embed_dim, model_dims)
    
    def forward(self, x):
        embed = relu(self.linear0(x))
        hidden1 = relu(self.linear1(embed))
        hidden2 = relu(self.linear2(hidden1))
        logits = self.unembed(hidden2)
        return logits
    


class Perceptron(nn.Module):

    def __init__(self, n: int, layer_dim: int):
        super().__init__()
        self.n = n
        self.layer_dim = layer_dim
        self.linear = nn.Linear(in_features=self.n, out_features=self.layer_dim, bias=True)
        self.unembed = nn.Linear(in_features=self.layer_dim, out_features=2, bias=False)

    @classmethod
    def from_config(cls, config):
        n = config['n']
        model_dims = config['model_dim']
        
        return cls(n, model_dims)
    
    def forward(self, x):
        activations = relu(self.linear(x))
        logits = self.unembed(activations)
        return logits
    