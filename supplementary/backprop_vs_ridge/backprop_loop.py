from tqdm import trange
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

def vanderpol_vec_torch(x, mu=1.0):
    is_1d = (x.dim() == 1)
    if is_1d:
        x = x.unsqueeze(0)   # (1,2)

    if not torch.is_tensor(mu):
        mu = x.new_tensor(mu)
    else:
        mu = mu.to(dtype=x.dtype, device=x.device)

    x0 = x[:, 0]   # x
    x1 = x[:, 1]   # y

    f0 = x1
    f1 = -x0 + mu * x1 * (1 - x0 * x0)

    out = torch.stack([f0, f1], dim=1)  # (B,2)

    if is_1d:
        return out[0]
    return out

network_sizes = [64, 256, 1024, 4096]
n_sims = 50
k_dim = 2
nu = 0.25

data_filename = "vdp_training_data.pt"
n_samples = 25_000

if os.path.exists(data_filename):
    print(f"Loading existing training data from {data_filename}...")
    data = torch.load(data_filename, map_location=device, weights_only=True)
    x_train = data['x_train'].to(device)
    y_train = data['y_train'].to(device)
else:
    print(f"Generating new training data and saving to {data_filename}...")
    x_train = (torch.rand(n_samples, 2) * 8.0 - 4.0).to(device)
    y_train = vanderpol_vec_torch(x_train).to(device)
    
    torch.save({
        'x_train': x_train.cpu(), 
        'y_train': y_train.cpu()
    }, data_filename)

Z = y_train + x_train

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


output_filename = "vdp_backprop_errors.csv"

with open(output_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n_neurons", "sim_id", "train_mse"])
    
    print(f"Starting VDP regression experiments. Saving results to {output_filename}")
    print("-" * 50)
    
    for N_neurons in network_sizes:
        print(f"Training ensemble for N = {N_neurons} neurons...")
        
        for sim in range(n_sims):

            latent_dim = N_neurons
            model = TwoLayerMLP(latent_dim).to(device)

            opt = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            _eye2 = torch.eye(2, device=device)
            D_target = (nu**2) * _eye2  

            lambda_diff = 0
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
            
            train_mse = torch.mean((y_train - pred)**2).item()
            
            writer.writerow([N_neurons, sim, train_mse])
            
        print(f"  -> Completed {n_sims} sims. Last MSE: {train_mse:.6f}")

print("-" * 50)
print("All experiments finished successfully.")
