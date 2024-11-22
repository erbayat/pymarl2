import numpy as np
from scipy import optimize
from scipy.linalg import solve
from scipy.stats import norm, uniform
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from itertools import product


def nystrom_2d(x, u, grid, a, gamma, kernel, lowrank=10):
    mat_k_uu = kernel.rbf_dist(cdist(u, u))
    mat_k_grid = kernel.rbf_dist(cdist(grid, grid))
    mat_k_xu = kernel.rbf_dist(cdist(x, u))
    mat_k_xgrid = kernel.rbf_dist(cdist(x, grid))
    eigenvalue, eigenvector = eigsh(mat_k_uu, k=lowrank)
    mat_eigenvalue = np.diag(eigenvalue)
    mat_eigenvalue_est_inv = np.linalg.inv(a * mat_eigenvalue ** 2 / u.shape[0] + gamma * mat_eigenvalue)
    mat_k_xx = mat_k_xu @ eigenvector @ mat_eigenvalue_est_inv @ eigenvector.T @ mat_k_xu.T
    mat_k_xgrid_ = solve(a * mat_k_grid / grid.shape[0] + gamma * np.eye(grid.shape[0]), mat_k_xgrid.T).T
    return mat_k_xx, mat_k_xgrid_


def loss(alpha, obs_size, mat_k_xx, a, gamma):
    loss_value = 0
    for i in range(obs_size):
        loss_value -= np.log(a * (alpha @ mat_k_xx[i]) ** 2)
    loss_value += gamma * alpha @ mat_k_xx @ mat_k_xx.T @ alpha
    return loss_value


def loss_gradient(alpha, obs_size, mat_k_xx, a, gamma):
    gradient_value = np.zeros(obs_size)
    for i in range(obs_size):
        gradient_value -= 2 * mat_k_xx[i] / (alpha @ mat_k_xx)
    gradient_value += 2 * gamma * alpha @ mat_k_xx
    return gradient_value


def run_nystrom_2d(x_grid, y_grid, obs, a, gamma, kernel, num_samples=100):
    x, y = np.meshgrid(x_grid, y_grid)
    xy_grid = np.column_stack((x.ravel(), y.ravel()))[:, ::-1]
    sample_indices = np.random.choice(xy_grid.shape[0], num_samples, replace=False)
    xy_grid_sample = xy_grid[sample_indices, :]
    mat_k_xx, mat_k_xu = nystrom_2d(obs, xy_grid_sample, xy_grid, a, gamma, kernel)
    obs_size = obs.shape[0]
    alpha_init = norm.rvs(0, 1, obs_size)
    alpha_new = optimize.minimize(loss, alpha_init, args=(obs_size, mat_k_xx, a, gamma), method='SLSQP',
                                  jac=loss_gradient, bounds=[(-x_grid.shape[0], y_grid.shape[0])] * obs_size,
                                  options={'disp': False})
    intensity = a * (alpha_new.x @ mat_k_xu) ** 2
    mat_intensity = np.reshape(intensity, (x_grid.shape[0], y_grid.shape[0])).T
    return intensity, mat_intensity


def covariance_2d(x_grid, y_grid, obs, intensity_init, grid_size,
                  kappa=lambda g: g**2,
                  kappa_inv=lambda g: np.sqrt(g),
                  kappa_grad_1=lambda g: 2 * g,
                  kappa_grad_2=lambda g: 2):
    intensity = []
    for o in obs.astype(int):
        intensity.append(intensity_init[(o[0] - 1) * x_grid.shape[0] + o[1]])
    intensity = np.array(intensity)
    intensity_inv = kappa_inv(intensity)
    gp_model = GaussianProcessRegressor(kernel=RBF(), random_state=42)
    gp_model.fit(obs, intensity_inv)
    mean, cov = gp_model.predict(np.array(list(product(x_grid, y_grid))), return_cov=True)
    cov_inv = np.linalg.inv(cov)
    vec_cov_grad = (kappa(mean) * kappa_grad_2(mean) - kappa_grad_1(mean)**2) / kappa(mean)**2 - kappa_grad_2(mean) * grid_size
    mat_cov_grad = np.diag(vec_cov_grad)
    mat_cov = np.linalg.inv(cov_inv - mat_cov_grad)
    std = np.sqrt(np.abs(np.diag(mat_cov))) * 1.0
    return mat_cov, std
