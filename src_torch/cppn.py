import torch
import torch.nn as nn

class CPPN(nn.Module):
    def __init__(self, n_layers: int, d_hidden: int, nonlin: str = 'relu'):
        super().__init__()
        self.n_layers, self.d_hidden, self.nonlin = n_layers, d_hidden, nonlin
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(d_hidden if i>0 else 3, d_hidden))

            if nonlin == 'relu':
                layers.append(nn.ReLU())
            elif nonlin == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f'Unknown nonlinearity {nonlin}')
        layers.append(nn.Linear(d_hidden, 3))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
                # nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                # nn.init.normal_(layer.bias, mean=0.0, std=0.02)
        
    def forward(self, x):
        return self.net(x)
    
    def generate_image(self, img_size=128, device=None):
        x = y = torch.linspace(-1, 1, img_size, device=device)
        x, y = torch.meshgrid(x, y, indexing='ij')
        d = torch.sqrt(x**2 + y**2)
        xyd = torch.stack([x, y, d], dim=-1)
        xyd = torch.reshape(xyd, (-1, 3))
        rgb = self(xyd)
        rgb = rgb.view(img_size, img_size, 3)
        return rgb


class PixelDofs(nn.Module):
    def __init__(self, init_std=1e-3):
        super().__init__()
        self.dofs = nn.Parameter(torch.randn(224, 224, 3) * init_std)
        
    def generate_image(self, img_size=224, device=None):
        rgb = torch.sigmoid(self.dofs)
        return rgb







