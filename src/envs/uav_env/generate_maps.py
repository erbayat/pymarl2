import numpy as np
import os

def generate_binary_map_with_ratio(rows, cols, ratio_of_ones):
    # Total number of elements in the map
    total_elements = rows * cols
    
    # Calculate how many ones we need based on the ratio
    num_ones = int(total_elements * ratio_of_ones)
    
    # Create a flat array with the desired number of ones and zeros
    flat_map = np.array([1] * num_ones + [0] * (total_elements - num_ones))
    
    # Shuffle the array to distribute ones and zeros randomly
    np.random.shuffle(flat_map)
    
    # Reshape it to the desired dimensions
    binary_map = flat_map.reshape((rows, cols))
    
    return binary_map

# Set dimensions and ratio
rows, cols = 20, 30
ratio_of_ones = 0.7  # Example ratio

# Generate 15 binary maps
maps = {}
for i in range(1, 16):
    maps[f"map_{i}"] = generate_binary_map_with_ratio(rows, cols, ratio_of_ones)

# Directory to save the .npy files
save_directory = r'C:\Users\erbayat\Desktop\Topics\Bayesian Estimation - Optimization\pymarl2\src\envs\uav_env\maps'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save each map as an npy file
for i in range(1, 16):
    np.save(os.path.join(save_directory, f'map_{i}.npy'), maps[f'map_{i}'])


