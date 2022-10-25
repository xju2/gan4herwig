"""Conditional GAN implemented in Pytorch"""
import torch

class Generator(torch.nn.Module):
    """Generator for the GAN"""
    def __init__(self, noise_dim: int, cond_dim: int, output_dim: int,
                 num_layers: int, hidden_dim: int):
        super(Generator, self).__init__()
        layer_list = []
        for i in range(num_layers):
            if i == 0:
                layer_list.append(torch.nn.Linear(noise_dim + cond_dim, hidden_dim))
            else:
                layer_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(torch.nn.LeakyReLU(0.2))
        
        layer_list.append(torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*layer_list)
        

    def forward(self, x):
        return self.model(x)
    
    
class Discriminator(torch.nn.Module):
    """Discriminator for the GAN"""
    def __init__(self, input_dim: int, cond_dim: int, num_layers: int, hidden_dim: int):
        super(Discriminator, self).__init__()
        layer_list = []
        for i in range(num_layers):
            if i == 0:
                layer_list.append(torch.nn.Linear(input_dim + cond_dim, hidden_dim))
            else:
                layer_list.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layer_list.append(torch.nn.LeakyReLU(0.2))
        
        layer_list.append(torch.nn.Linear(hidden_dim, 1))
        self.model = torch.nn.Sequential(*layer_list)
        
    
    def forward(self, x):
        return self.model(x)