import numpy as np
import scipy as sp
from scipy.interpolate import CubicSpline, RectBivariateSpline

def gaussian_potential(x,mu,A,sigma):
    
    N = mu.shape[1]
    V = 0

    for j in range(0,N):
        V -= A[j]*np.exp(-((np.linalg.norm(x-mu[:,j]))**2)/(2*(sigma[j]**2)))
    
    return V

def grad_gaussian_potential(x,mu,A,sigma):
    
    N = mu.shape[1]
    M = mu.shape[0]
    V = np.zeros(M)

    for j in range(0,N):
        V += (A[j]/(sigma[j]**2))*(x-mu[:,j])*np.exp(-((np.linalg.norm(x-mu[:,j]))**2)/(2*(sigma[j]**2)))

    return V

def intersecting_circle(mu):
    x1,y1 = mu[:,0]; x2,y2 = mu[:,1]; x3,y3 = mu[:,2]

    D = 2*(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    u_x = (1/D)*((x1**2 + y1**2)*(y2 - y3) + (x2**2 + y2**2)*(y3 - y1) + (x3**2 + y3**2)*(y1 - y2))
    u_y = (1/D)*((x1**2 + y1**2)*(x3 - x2) + (x2**2 + y2**2)*(x1 - x3) + (x3**2 + y3**2)*(x2 - x1))

    return (u_x,u_y), np.sqrt((x1-u_x)**2 + (y1-u_y)**2)

def gaussian_potential_vectorized(x, mu, A, sigma):
    
    diff = x[:, np.newaxis, :] - mu.T[np.newaxis, :, :]

    dist_squared = np.sum(diff**2, axis=2)
    
    exponent = np.exp(-dist_squared / (2 * sigma**2))
    
    potential_components = A * exponent
    
    total_potential = -np.sum(potential_components, axis=1)
    
    return total_potential

def grad_gaussian_potential_vectorized(x, mu, A, sigma):
    
    diff = x[:, np.newaxis, :] - mu.T[np.newaxis, :, :]
    
    dist_squared = np.sum(diff**2, axis=2)
    
    exponent = np.exp(-dist_squared / (2 * sigma**2))
    
    scalar_part = A / sigma**2
    
    grad_components = diff * (scalar_part * exponent)[:, :, np.newaxis]
    
    total_grad = np.sum(grad_components, axis=1)
    
    return total_grad

def simulate_process_EM(F,X0,T,dt,nu):

    Nt = int(T/dt)
    M = np.shape(X0)[0]
    X = np.zeros((M,Nt))
    X[:,0] = X0

    for t in range(0,Nt-1):
        X[:,t+1] = X[:,t] + dt*(F(X[:,t])) + nu*np.random.normal(0, np.sqrt(dt), size=M)
        
    return X

def simulate_process_EM_ni(F,Sigma,X0,T,dt):

    Nt = int(T/dt)
    M = np.shape(X0)[0]
    X = np.zeros((M,Nt))
    X[:,0] = X0
    
    dW = np.random.normal(0, np.sqrt(dt), size=(M, Nt))
    noise_increments = Sigma @ dW
    
    for t in range(0,Nt-1):
        X[:,t+1] = X[:,t] + dt*(F(X[:,t])) + noise_increments[:, t]

    return X