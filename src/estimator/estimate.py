import numpy as np
import matplotlib.pyplot as plt
from estimator.discretize import run_nystrom_2d, covariance_2d
from estimator.kernel import RBFKernel


def estimate(step,x_loc,y_loc,args, grid_size=1, a=8, gamma=0.5, kernel=RBFKernel(), do_visualize=False):
    grid_min = 0
    env_args = getattr(args,'env_args',{})
    grid_shape = env_args.get('grid_shape',[20,30])
    x_max=grid_shape[0]
    y_max=grid_shape[1]
    x_grid = np.arange(grid_min, x_max, grid_size)
    y_grid = np.arange(grid_min, y_max, grid_size)
    N = len(x_loc)  
    shape = (N, 2)
    location = np.zeros(shape, dtype=int)
    location[:, 0] = x_loc  # Column 0: values between 0-x_max
    location[:, 1] = y_loc  # Column 1: values between 0-y_max
    intensity_list = []
    for _ in range(step):
        intensity, _ = run_nystrom_2d(x_grid, y_grid, location, a, gamma, kernel)
        intensity_list.append(intensity)
    intensity_avg = np.average(np.array(intensity_list), axis=0)
    intensity_avg = np.array([0.0001 if i < 0 else i for i in intensity_avg])
    cov, std = covariance_2d(x_grid, y_grid, location, intensity_avg, grid_size)
    std = np.reshape(std, (x_grid.shape[0], y_grid.shape[0])).T
    intensity_max = np.max(intensity_avg)
    intensity_avg = np.reshape(intensity_avg, (x_grid.shape[0], y_grid.shape[0])).T

    if do_visualize:
        fig = plt.figure(figsize=(2, 2))
        axs = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        x, y = np.meshgrid(x_grid, y_grid)
        axs.pcolor(x, y, intensity_avg, vmax=intensity_max, rasterized=1, cmap='Blues')
        axs.scatter(location[:, 0], location[:, 1], s=2, c='k')
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set(xlabel=None, ylabel=None)
        axs.set_xlim(grid_min, x_max)
        axs.set_ylim(grid_min, y_max)
        plt.show()
        fig.savefig('./neuron.png', format='png', bbox_inches='tight', dpi=300)

    return intensity_avg, intensity_max, cov, std


def load_data(file_name):
    arr = np.load(file_name)[:]
    x = arr[:, 0]
    y = arr[:, 1]
    return np.concatenate((x[:, None], y[:, None]), axis=1)


# TEST
if __name__ == "__main__":
    estimate(10)




