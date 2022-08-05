from scipy.stats import special_ortho_group
import numpy as np
import torch

def permute_row(x, n_particles, n_dims):
    #print(x)
    x = x.view(n_particles, n_dims)
    permutation = np.arange(n_particles)
    np.random.shuffle(permutation)
    return x[permutation].view(-1)

def permute(x, n_particles, n_dims):
    n_batch = x.shape[0]
    x = x.view(-1, n_particles * n_dims)
    for i in range(n_batch):
        x[i] = permute_row(x[i], n_particles, n_dims)
    return x

def rotate(x, n_particles, n_dims):
    n_batch = x.shape[0]
    x = x.view(-1, n_particles, n_dims)
    so = special_ortho_group(dim=n_dims)
    rotation = so.rvs(n_batch).reshape(n_batch, n_dims, n_dims)
    rotation = torch.Tensor(rotation).to(x)
    x = torch.einsum("nmd, nde -> nme", x, rotation)
    x = x @ rotation
    return x.view(n_batch, -1)

def permute_row_xv(x, v, n_particles, n_dims):
    x = x.view(n_particles, n_dims)
    v = v.view(n_particles , n_dims)
    permutation = np.arange(n_particles)
    np.random.shuffle(permutation)
    return x[permutation].view(-1), v[permutation].view(-1)

def permute_xv(x, v, n_particles, n_dims):
    n_batch = x.shape[0]
    x = x.view(n_batch, n_particles * n_dims).clone()
    v = v.view(n_batch, n_particles * n_dims).clone()

    for i in range(n_batch):
        x[i], v[i] = permute_row_xv(x[i], v[i], n_particles, n_dims)
    return x, v

def rotate_xv(x, v, n_particles, n_dims):
    n_batch = x.shape[0]
    x = x.view(-1, n_particles, n_dims)
    v = v.view(-1, n_particles, n_dims)
    so = special_ortho_group(dim=n_dims)
    rotation = so.rvs(n_batch).reshape(n_batch, n_dims, n_dims)
    rotation = torch.Tensor(rotation).to(x)
    x = torch.einsum("nmd, nde -> nme", x, rotation)
    x = x @ rotation
    v = torch.einsum("nmd, nde -> nme", v, rotation)
    v = v @ rotation
    return x.view(n_batch, -1), v.view(n_batch, -1)


def test_equivariance(bg, data, n_particles, n_dimensions, threshold=1e-3):
    dim = n_particles * n_dimensions
    prior = bg.prior
    batch_noise = bg.prior.sample(data.shape[0])
    data_aug, batch_noise_aug = rotate_xv(data.cuda(), batch_noise, n_particles, n_dimensions)
    data_aug, batch_noise_aug = permute_xv(data_aug, batch_noise_aug, n_particles, n_dimensions)
    batch = torch.cat([data.cuda(), batch_noise], dim=1)
    batch_aug = torch.cat([data_aug, batch_noise_aug], dim=1)
    with torch.no_grad():
        latent_samples, latent_dlogp = bg.flow(batch, inverse=True)
        nll = (- (1 - 1 / n_particles) * latent_dlogp 
               + prior.energy(latent_samples[:, dim:]) 
               + prior.energy(latent_samples[:, :dim]) 
              )

        latent_samples_aug, latent_dlogp_aug = bg.flow(batch_aug, inverse=True)
        nll_aug = (- (1 - 1 / n_particles) * latent_dlogp_aug
                   + prior.energy(latent_samples_aug[:, dim:])
                   + prior.energy(latent_samples_aug[:, :dim])  
                  )

    max_deviation = torch.max(nll - nll_aug)
    return (max_deviation < threshold).item()

def test_mean_free(samples, n_particles, n_dimensions, threshold=1e-5):
    max_deviation = samples.reshape(-1, n_particles, n_dimensions).mean(1).max()    
    return (max_deviation < threshold).item()
