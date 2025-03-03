import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        hidden_dim = [1024, 512]
        if isinstance(input_dim, int):
            input_dim = (input_dim,)
        if isinstance(output_dim, int):
            output_dim = (output_dim,)
            
        all_dims = list(input_dim) + hidden_dim + list(output_dim)
        layers = []
        print(all_dims, 'all dimensions')
        for i in range(len(all_dims) - 1):
            layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            if i < len(all_dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    mlp = MLP(10, 10)
    x = torch.randn(10)
    print(mlp(x))