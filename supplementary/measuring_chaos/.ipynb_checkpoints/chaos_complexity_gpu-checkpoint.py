import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.func import vmap, functional_call, stack_module_state
from tqdm import trange, tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def lorenz_vec_torch(x, sigma=10, rho=28, beta=8/3):
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    f0 = sigma * (x1 - x0)
    f1 = x0 * (rho - x2) - x1
    f2 = x0 * x1 - beta * x2
    return torch.stack([f0, f1, f2], dim=1)

def dadras_vec_torch(x, p = 3, o = 2.7, r = 1.7, c = 2, e = 9):
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    f0 = x1 - p*x0 + o*x1*x2
    f1 = r*x1 -x0*x2 + x2
    f2 = c*x0*x1-e*x2
    return torch.stack([f0, f1, f2], dim=1)

def rossler_vec_torch(x, a = 0.2, b = 0.2, c = 5.7):
    is_1d = (x.dim() == 1)
    if is_1d:
        x = x.unsqueeze(0)   # (1,2)

    x0 = x[:, 0]   # x
    x1 = x[:, 1]   # y
    x2 = x[:, 2]   # z

    f0 = -x1-x2
    f1 = x0+a*x1
    f2 = b+x2*(x0-c)

    out = torch.stack([f0, f1, f2], dim=1)  # (B,2)

    if is_1d:
        return out[0]
    return out

def havlorsen_vec_torch(x, a = 1.4):
    is_1d = (x.dim() == 1)
    if is_1d:
        x = x.unsqueeze(0)   # (1,2)

    x0 = x[:, 0]   # x
    x1 = x[:, 1]   # y
    x2 = x[:, 2]   # z

    f0 = -a*x0 -4*x1-4*x2 -x1*x1
    f1 = -a*x1-4*x2-4*x0-x2*x2
    f2 = -a*x2-4*x0-4*x1-x0*x0

    out = torch.stack([f0, f1, f2], dim=1)  # (B,2)

    if is_1d:
        return out[0]
    return out

class TwoLayerMLP(nn.Module):
    def __init__(self, latent_dim=1024, num_noise_sources=8, proj_dim=3):
        super().__init__()
        self.Gamma = nn.Linear(proj_dim, latent_dim, bias=False)
        self.input_bias = nn.Parameter(torch.zeros(latent_dim))
        self.W_s = nn.Linear(latent_dim, proj_dim, bias=False)
        self.I_s = nn.Parameter(torch.zeros(proj_dim))
        # Kept for structure
        self.B_s = nn.Parameter(torch.randn(proj_dim, num_noise_sources))

    def forward(self, x):
        z = self.Gamma(x) + self.input_bias
        z = torch.tanh(z)
        out = -x + self.W_s(z) + self.I_s
        return out

reps = 100
latent_dim = 1024
epochs = 2000  
batch_size = 1024
n_samples = 125_000

system_name = 'havlorsen'
system_field = havlorsen_vec_torch
min_vals = torch.tensor([-15.0, -15.0,  -15.0]).to(device)
max_vals = torch.tensor([ 12.0,  12.0,  12]).to(device)
ranges = max_vals - min_vals

x_rand = torch.rand(n_samples, 3).to(device)
x_train = x_rand * ranges + min_vals
y_train = system_field(x_train).to(device)

dataset = torch.utils.data.TensorDataset(x_train, y_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

models = [TwoLayerMLP(latent_dim=latent_dim, proj_dim=3).to(device) for _ in range(reps)]
params, buffers = stack_module_state(models)

for p in params.values():
    p.requires_grad = True

def fmodel(params, buffers, x):
    return functional_call(models[0], (params, buffers), (x,))

def compute_loss_per_model(params, buffers, x, y):
    pred = fmodel(params, buffers, x)
    return nn.MSELoss()(pred, y)

calc_loss_vmap = vmap(compute_loss_per_model, in_dims=(0, 0, None, None))

optimizer = optim.Adam(params.values(), lr=1e-3)

print(f"Training {reps} models in parallel with batch size {batch_size}...")

for epoch in trange(epochs, desc='Vectorized Training'):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        
        losses = calc_loss_vmap(params, buffers, batch_x, batch_y)
        
        total_loss = losses.sum()
        total_loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        tqdm.write(f"Epoch {epoch+1}: Avg Loss {losses.mean().item():.6f}")

print("Computing spectral metrics using full matrices...")

spec_rad = torch.zeros(reps)
avg_pr = torch.zeros(reps)
dom_pr = torch.zeros(reps)

Gamma_all = params['Gamma.weight'].detach()
W_s_all = params['W_s.weight'].detach()

for r in trange(reps, desc="Analyzing Models"):
    
    Gamma = Gamma_all[r]
    W_s = W_s_all[r]
    
    W = torch.matmul(Gamma, W_s)
    
    vals, vecs = torch.linalg.eig(W)
    spec_rad[r] = torch.max(abs(vals))

    magnitudes = vecs.abs()
    
    numerator = magnitudes.pow(2).sum(dim=0).pow(2) 
    denominator = magnitudes.pow(4).sum(dim=0)
    all_prs = numerator / denominator
    
    avg_pr[r] = all_prs.mean()

    idx_max = torch.argmax(vals.abs())
    dom_pr[r] = all_prs[idx_max]

np.savetxt(f"{system_name}_spec_rad.txt", spec_rad.cpu().numpy())
np.savetxt(f"{system_name}_avg_pr.txt", avg_pr.cpu().numpy())
np.savetxt(f"{system_name}_dom_pr.txt", dom_pr.cpu().numpy())

print("Done.")
