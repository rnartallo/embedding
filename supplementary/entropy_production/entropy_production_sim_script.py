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

def hopf_vec_torch(x, w=1.0):
    is_1d = (x.dim() == 1)
    if is_1d:
        x = x.unsqueeze(0)   # (1,2)

    if not torch.is_tensor(w):
        w = x.new_tensor(w)
    else:
        w = w.to(dtype=x.dtype, device=x.device)

    x0 = x[:, 0]   # x
    x1 = x[:, 1]   # y

    f0 = (1-x0**2-x1**2)*x0 - w*x1
    f1 = (1-x0**2-x1**2)*x1 + w*x0

    out = torch.stack([f0, f1], dim=1)  # (B,2)

    if is_1d:
        return out[0]
    return out

nu = 0.25

class TwoLayerMLP(nn.Module):
    def __init__(self, latent_dim=64, num_noise_sources = 8):
        super().__init__()
        self.Gamma = nn.Linear(2, latent_dim, bias=False)   
        self.input_bias = nn.Parameter(torch.zeros(latent_dim))  
        self.W_s = nn.Linear(latent_dim, 2, bias= False)
        self.I_s = nn.Parameter(torch.zeros(2))
        self.B_s = nn.Parameter(torch.randn(2, num_noise_sources))

    def forward(self, x):
        z = self.Gamma(x) + self.input_bias       
        z = torch.tanh(z)
        out = -x + self.W_s(z) + self.I_s  
        return out


def train_model(w):
    latent_dim = 64
    model = TwoLayerMLP(latent_dim).to(device)
    
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    n_samples = 25_000
    x_train = (torch.rand(n_samples, 2) * 8.0 - 4.0).to(device)
    y_train = hopf_vec_torch(x_train, w = w).to(device)
    _eye2 = torch.eye(2, device=device)
    D_target = (nu**2) * _eye2  
    
    lambda_diff = 20
    epochs = 30_000
    
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


    return model

def asym_norm(W_sym, W_asym):
    fa = np.linalg.norm(W_asym, 'fro')
    fs = np.linalg.norm(W_sym, 'fro')
    den = fs + fa
    return 0.0 if den == 0 else fa / den

w_range = torch.linspace(0,5,21)

filename_normal = "normal_decomp_hopf.txt"
filename_LR = "lr_decomp_hopf.txt"

reps = 100

for idx, w in enumerate(w_range):

    normal_reps = np.zeros(reps)
    lr_reps = np.zeros(reps)
    
    for r in range(reps):
    
        model = train_model(w)
    
        Gamma = model.Gamma.weight.detach()
        Gamma_pinv = torch.linalg.pinv(Gamma).detach()
        b = model.input_bias.detach()
        W = Gamma @ model.W_s.weight.detach() 
        I = Gamma @ model.I_s.detach() +  b
        B = Gamma @ model.B_s.detach()
    
        W_np = W.cpu().numpy()
        Gamma_np = Gamma.cpu().numpy()
        Gamma_pinvnp = np.linalg.pinv(Gamma_np)
        
        C = 0.5*(W_np+W_np.T)
        
        Omega = Gamma_pinvnp@C@Gamma_np@(Gamma_pinvnp)

        normal_reps[r] = asym_norm(C, W_np-C)
        lr_reps[r] = asym_norm(Gamma_np@Omega, W_np-Gamma_np@Omega)
        
    row_normal = np.concatenate(([float(w.item())], normal_reps))
    row_lr = np.concatenate(([float(w.item())], lr_reps))

    with open(filename_normal, "a") as f:
        np.savetxt(f, row_normal[None, :], fmt="%.6f")
    with open(filename_LR, "a") as f:
        np.savetxt(f, row_lr[None, :], fmt="%.6f")
    
