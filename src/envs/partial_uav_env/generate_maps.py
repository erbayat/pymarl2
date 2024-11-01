import numpy as np
import os

# def generate_binary_map_with_ratio(rows, cols, ratio_of_ones):
#     # Total number of elements in the map
#     total_elements = rows * cols
    
#     # Calculate how many ones we need based on the ratio
#     num_ones = int(total_elements * ratio_of_ones)
    
#     # Create a flat array with the desired number of ones and zeros
#     flat_map = np.array([1] * num_ones + [0] * (total_elements - num_ones))
    
#     # Shuffle the array to distribute ones and zeros randomly
#     np.random.shuffle(flat_map)
    
#     # Reshape it to the desired dimensions
#     binary_map = flat_map.reshape((rows, cols))
    
#     return binary_map

def generate_binary_map(N, M, ones_ratio=0.7, a=1.0, b=5.0, c=1.0, d=5.0):
    x = np.arange(0, N).reshape(-1, 1)
    y = np.arange(0, M).reshape(1, -1)
    b= N//2
    d =  M//2

    # Create the intensity map
    intensity_map = a * (x - b)**2 + c * (y - d)**2
    
    # Normalize the intensity map
    intensity_map = (intensity_map - np.min(intensity_map)) / (np.max(intensity_map) - np.min(intensity_map))
    
    
    # Scale the intensity map to match the expected number of ones (target ratio)
    target_sum = ones_ratio * N * M
    current_sum = np.sum(intensity_map)
    
    # Rescale the intensity map so that the expected sum of ones matches the target number
    scaled_intensity_map = intensity_map * (target_sum / current_sum)
    
    # Ensure that the values are between 0 and 1 after scaling
    scaled_intensity_map = np.clip(scaled_intensity_map, 0, 1)
    
    # Generate a random map where each cell's probability of being 1 is given by the scaled intensity value
    binary_map = (np.random.rand(N, M) < scaled_intensity_map).astype(int)
    
    return binary_map

# Set dimensions and ratio
rows, cols = 20, 30
ratio_of_ones = 0.7  # Example ratio

# Generate 15 binary maps
maps = {}
for i in range(1, 105):
    maps[f"map_{i}"] = generate_binary_map(rows, cols,ratio_of_ones)
    print(maps[f"map_{i}"])

# Directory to save the .npy files
current_dir = os.path.dirname(os.path.abspath(__file__))
save_directory = os.path.join(current_dir, 'maps')

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Save each map as an npy file
for i in range(1, 105):
    np.save(os.path.join(save_directory, f'map_{i}.npy'), maps[f'map_{i}'])


