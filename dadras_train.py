import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from scipy.optimize import minimize
from scipy.linalg import orth
from tqdm import trange
import helpers_attractors as h
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline, RectBivariateSpline
from matplotlib.path import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

def dadras_vec_torch(x, p = 3, o = 2.7, r = 1.7, c = 2, e = 9):
    is_1d = (x.dim() == 1)
    if is_1d:
        x = x.unsqueeze(0)   # (1,2)

    x0 = x[:, 0]   # x
    x1 = x[:, 1]   # y
    x2 = x[:, 2]   # z

    f0 = x1 - p*x0 + o*x1*x2
    f1 = r*x1 -x0*x2 + x2
    f2 = c*x0*x1-e*x2

    out = torch.stack([f0, f1, f2], dim=1)  # (B,2)

    if is_1d:
        return out[0]
    return out

class TwoLayerMLP(nn.Module):
    def __init__(self, latent_dim=64, num_noise_sources = 8, proj_dim = 2):
        super().__init__()
        self.Gamma = nn.Linear(proj_dim, latent_dim, bias=False)   
        self.input_bias = nn.Parameter(torch.zeros(latent_dim))  
        self.W_s = nn.Linear(latent_dim, proj_dim, bias= False)
        self.I_s = nn.Parameter(torch.zeros(proj_dim))
        self.B_s = nn.Parameter(torch.randn(proj_dim, num_noise_sources))

    def forward(self, x):
        z = self.Gamma(x) + self.input_bias       
        z = torch.tanh(z)
        out = -x + self.W_s(z) + self.I_s  
        return out

latent_dim = 1024
model = TwoLayerMLP(latent_dim, proj_dim = 3).to(device)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

n_samples = 150_000
min_vals = torch.tensor([-20.0, -12.0,  -15.0]).to(device)
max_vals = torch.tensor([ 20.0,  10.0, 15.0]).to(device)
ranges = max_vals - min_vals  
x_rand = torch.rand(n_samples, 3).to(device)
x_train = x_rand * ranges + min_vals
y_train = dadras_vec_torch(x_train).to(device)
_eye2 = torch.eye(3, device=device)

nu = 0.01
D_target = (nu**2) * _eye2  

lambda_diff = 20
epochs = 100_000

for epoch in trange(epochs, desc= 'training MLP'):
    opt.zero_grad()
    
    
    pred = model(x_train)
    
    drift_loss = loss_fn(pred, y_train)

    D_model = model.B_s@(model.B_s.T)
    diffusion_loss = torch.norm(D_model - D_target, p='fro')**2
    
    loss = drift_loss + lambda_diff*diffusion_loss
    loss.backward()
    opt.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"epoch {epoch+1:4d} loss: {loss.item():.6f} drift: {drift_loss:.6f} diff: {diffusion_loss:.6f}")

Gamma = model.Gamma.weight.detach()
Gamma_pinv = torch.linalg.pinv(Gamma).detach()
b = model.input_bias.detach()
W = Gamma @ model.W_s.weight.detach() 
I = Gamma @ model.I_s.detach() +  b
B = Gamma @ model.B_s.detach()


torch.save({
    'W': W.cpu(),
    'I': I.cpu(),
    'Gamma': Gamma.cpu(),
    'Gamma_pinv': Gamma_pinv.cpu(),
    'B': B.cpu(),
    'b': b.cpu()
}, "dadras_rnn.pt")

print("Training finished and parameters saved.")