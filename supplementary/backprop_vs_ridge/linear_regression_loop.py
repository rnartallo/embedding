import os
import torch
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
rho_in = 0.1
lambda_ridge = 0.01

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

output_filename = "vdp_regression_errors.csv"

with open(output_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["n_neurons", "sim_id", "train_mse"])
    
    print(f"Starting VDP regression experiments. Saving results to {output_filename}")
    print("-" * 50)
    
    for N_neurons in network_sizes:
        print(f"Training ensemble for N = {N_neurons} neurons...")
        
        for sim in range(n_sims):
            
            Gamma = torch.randn(N_neurons, k_dim, device=device) * (rho_in / np.sqrt(k_dim))
            b = torch.randn(N_neurons, device=device) * 0.01
            
            H = torch.tanh(x_train @ Gamma.T + b)
            
            ones = torch.ones(x_train.shape[0], 1, device=device)
            H_aug = torch.cat([H, ones], dim=1)
            
            H_T_H = H_aug.T @ H_aug
            H_T_Z = H_aug.T @ Z
            
            reg_matrix = lambda_ridge * torch.eye(N_neurons + 1, device=device)
            reg_matrix[-1, -1] = 0.0 
            
            Theta_T = torch.linalg.solve(H_T_H + reg_matrix, H_T_Z)
            
            Z_pred = H_aug @ Theta_T
            train_mse = torch.mean((Z_pred - Z)**2).item()
            
            writer.writerow([N_neurons, sim, train_mse])
            
        print(f"  -> Completed {n_sims} sims. Last MSE: {train_mse:.6f}")

print("-" * 50)
print("All experiments finished successfully.")