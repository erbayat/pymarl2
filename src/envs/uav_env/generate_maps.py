import numpy as np
import os

import numpy as np

import numpy as np

def generate_map_with_bisection_radius(rows, cols, target_mean=0.5, inside_prob=0.9, outside_prob=0.1, tol=1e-3, max_iter=100):
    """
    Generate a binary map with a random center and adjust the radius using bisection to meet the target mean.

    Args:
        rows (int): Number of rows in the map.
        cols (int): Number of columns in the map.
        target_mean (float): Desired mean probability of the map.
        inside_prob (float): Probability of an event inside the radius.
        outside_prob (float): Probability of an event outside the radius.
        tol (float): Tolerance for the difference between the current and target mean.
        max_iter (int): Maximum number of bisection iterations.

    Returns:
        np.ndarray: Generated binary map.
    """
    # Randomly select the center
    center_x = np.random.randint(0, rows)
    center_y = np.random.randint(0, cols)

    # Define the search range for the radius
    min_radius = 0
    max_radius = max(rows,cols)  # Diagonal of the map
    best_radius = None

    for _ in range(max_iter):
        # Calculate the midpoint radius
        radius = (min_radius + max_radius) / 2

        # Create grid
        x = np.arange(0, rows).reshape(-1, 1)
        y = np.arange(0, cols).reshape(1, -1)

        # Calculate distance from the center for each cell
        distance_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # Assign probabilities based on radius
        probability_map = np.where(distance_map <= radius, inside_prob, outside_prob)

        # Calculate the current mean
        current_mean = np.mean(probability_map)

        # Check if the current mean is close enough to the target
        if abs(current_mean - target_mean) < tol:
            best_radius = radius
            break

        # Adjust the search range
        if current_mean < target_mean:
            min_radius = radius  # Increase radius
        else:
            max_radius = radius  # Decrease radius

    # Finalize the radius if not set during the loop
    if best_radius is None:
        best_radius = radius

    # Generate the binary map with the final radius
    distance_map = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    probability_map = np.where(distance_map <= best_radius, inside_prob, outside_prob)
    binary_map = (np.random.rand(rows, cols) < probability_map).astype(int)

    return binary_map


def generate_binary_map_with_random_center_radius(N, M, ones_ratio=0.7, a_range=(0.5, 2.0), c_range=(0.5, 2.0)):
    """
    Generate a binary map with a latent function, where the center and scaling factors are randomized.

    Args:
        N (int): Number of rows.
        M (int): Number of columns.
        ones_ratio (float): Ratio of ones in the binary map.
        a_range (tuple): Range for the random scaling factor 'a'.
        c_range (tuple): Range for the random scaling factor 'c'.

    Returns:
        np.ndarray: Generated binary map.
    """
    # Generate random center coordinates
    b = np.random.randint(0, N)  # Random center x-coordinate
    d = np.random.randint(0, M)  # Random center y-coordinate

    # Generate random scaling factors
    a = np.random.uniform(*a_range)
    c = np.random.uniform(*c_range)

    # Create grid
    x = np.arange(0, N).reshape(-1, 1)
    y = np.arange(0, M).reshape(1, -1)

    # Compute the intensity map based on the latent function
    intensity_map = a * (x - b)**2 + c * (y - d)**2

    # Normalize the intensity map
    intensity_map = (intensity_map - np.min(intensity_map)) / (np.max(intensity_map) - np.min(intensity_map))

    # Scale the intensity map to match the expected number of ones (target ratio)
    target_sum = ones_ratio * N * M
    current_sum = np.sum(intensity_map)
    scaled_intensity_map = intensity_map * (target_sum / current_sum)

    # Clip values between 0 and 1
    scaled_intensity_map = np.clip(scaled_intensity_map, 0, 1)

    # Generate binary map based on the scaled intensity values
    binary_map = (np.random.rand(N, M) < scaled_intensity_map).astype(int)
    return binary_map


def generate_random_binary_map_with_ratio(rows, cols, ratio_of_ones):
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


def generate_binary_map_with_latent(N, M, ones_ratio=0.7, a=1.0, b=5.0, c=1.0, d=5.0):
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


def generate_leftmost_10_percent_map(rows, cols):
    # Create an empty map of all zeros
    binary_map = np.zeros((rows, cols), dtype=int)
    
    # Calculate the number of columns that should be set to 1 (leftmost 10%)
    num_leftmost_cols = int(cols * 0.3)
    
    # Set the leftmost 10% of columns in each row to 1
    binary_map[:, :num_leftmost_cols] = 1
    
    return binary_map

def generate_uav_positions(rows, cols, num_uavs):
    """
    Generate UAV positions for a given map with unique (x, y) coordinates.

    Args:
        rows (int): Number of rows in the map.
        cols (int): Number of columns in the map.
        num_uavs (int): Number of UAVs to place.

    Returns:
        np.ndarray: Array of UAV positions.
    """
    positions = []
    for _ in range(num_uavs):
        while True:
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            # Ensure the position is not on an obstacle (map value = 0) and is unique
            if (x, y) not in positions:  # Free space and unique
                positions.append((x, y))
                break
    return np.array(positions)


if __name__ == "__main__":

    # Set dimensions and ratio
    num_uavs = 7

    rows, cols = 50,50
    ratio_of_ones = 0.5  # Example ratio
    # Generate 15 binary maps
    maps = {}
    uav_positions = {}
    for i in range(1, 105):
        maps[f"map_{i}"] = generate_map_with_bisection_radius(rows, cols,ratio_of_ones)#generate_binary_map_with_random_center_radius(rows, cols,ratio_of_ones)
        uav_positions[f"map_{i}"] = generate_uav_positions(rows, cols, num_uavs)

        print(maps[f"map_{i}"])

    # Directory to save the .npy files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(current_dir, str(rows)+'_'+str(cols))
    maps_directory = os.path.join(save_directory, 'maps')
    uavs_directory = os.path.join(save_directory, "uav_positions")
    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Create directories if they don't exist
    os.makedirs(maps_directory, exist_ok=True)
    os.makedirs(uavs_directory, exist_ok=True)

    # Save each map and UAV positions as .npy files
    for i in range(1, 105):
        map_file = os.path.join(maps_directory, f"map_{i}.npy")
        uav_file = os.path.join(uavs_directory, f"uav_positions_{i}.npy")
        np.save(map_file, maps[f"map_{i}"])
        np.save(uav_file, uav_positions[f"map_{i}"])
        print(f"Saved map_{i} and UAV positions_{i}.")

    print("All maps and UAV positions generated and saved.")
