####################################################################################
#   CISC455 Group 4
#   Lung cancer Detection Using Evolutionary Algorithms for Image Segmentation
#   Names: Patrick Bernhard, Pavel-Dumitru Cernelev, Ben Tomkinson
#   Date: 4/11/2023
#####################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics import silhouette_score
from matplotlib import patches

def fitness(image, thresholds):
    '''
    Calculates the fitness score of an image based on the given thresholds.

    Args:
        image (numpy.ndarray): 2D numpy array, representing the input CT image
        thresholds (list): A list of two integers representing the lower and upper threshold values.
    
    Returns:
        fitness (float):  The fitness score of the image, calculated as the ratio of the number of pixels
        with intensities within the threshold range to the total number of pixels in the image.
    '''
    if np.isscalar(thresholds):
        thresholds = np.array([thresholds])
    mask = image > thresholds[:, np.newaxis, np.newaxis]
    fitness = np.sum(image * mask)
    return fitness

def pso(image, num_particles, num_iterations):
    '''
    Performs Particle Swarm Optimization (PSO) to segment an image using a threshold.

    Args:
        image (numpy.ndarray): 2D numpy array, representing the input CT image.
        num_particles (int): The number of particles in the PSO algorithm.
        num_iterations (int): The number of iterations to run the PSO algorithm.

    Returns:
        segmented_image (numpy.ndarray): A 2D numpy array representing the segmented image, where the pixels with intensities
        greater than the threshold are set to 1 and the rest are set to 0.
    '''
    # Set the search space for the threshold value
    search_space = (0, 255)
    
    # Initialize the particles' position1s and velocities
    positions = np.random.uniform(*search_space, size=num_particles)
    velocities = np.zeros_like(positions)
    
    # Set the best positions and fitnesses for the particles
    best_positions = positions.copy()
    best_fitnesses = np.zeros_like(best_positions)
    
    # Set the global best position and fitness
    global_best_position = positions[0]
    global_best_fitness = 0
    
    # Loop through the iterations
    for i in range(num_iterations):
        # Evaluate the fitness for each particle
        fitnesses = np.array([fitness(image, positions) for positions in positions])
        
        # Update the best positions and fitnesses for each particle
        mask = image > best_positions[:, np.newaxis, np.newaxis]
        mask_fitnesses = fitness(image, mask)
        better_positions = np.where(mask_fitnesses > best_fitnesses, best_positions, positions)
        better_fitnesses = np.where(mask_fitnesses > best_fitnesses, mask_fitnesses, fitnesses)
        
        # Update the global best position and fitness
        global_best_index = np.argmax(better_fitnesses)
        global_best_position = better_positions[global_best_index]
        global_best_fitness = better_fitnesses[global_best_index]

        # Update the particles' velocities and positions
        cognitive_velocity = np.random.uniform(0, 1, size=num_particles) * (best_positions - positions)
        social_velocity = np.random.uniform(0, 1, size=num_particles) * (global_best_position - positions)
        velocities = 0.5 * velocities + cognitive_velocity + social_velocity
        positions = np.clip(positions + velocities, *search_space)
        
        # Update the best positions and fitnesses for each particle
        mask = image > best_positions[:, np.newaxis, np.newaxis]
        mask_fitnesses = fitness(image, mask)
        best_positions = np.where(mask_fitnesses > best_fitnesses, best_positions, positions)
        best_fitnesses = np.where(mask_fitnesses > best_fitnesses, mask_fitnesses, best_fitnesses)
        
    # Threshold the image with the global best position
    mask = image > global_best_position
    segmented_image = image * mask
    
    return segmented_image

def perform_cluster(ct_image):
    '''
    Perform clustering on a CT image to extract the tumor.
    
    Args:
        ct_image (numpy.ndarray): 2D numpy array, representing the input CT image.
    
    Returns: 
        None
    '''
    #ct_image = depth_modification(ct_image, 8)
    ct_image_int_8 = ct_image.astype(np.uint8)
    
    # Perform PSO clustering
    segmented_image = pso(ct_image_int_8, num_particles=50, num_iterations=50)
    
    # Convert the segmented image to a 1D array
    pixels = segmented_image.reshape(-1, 1)
    
    # Perform KMeans clustering to get the labels of each pixel
    n_clusters = 2  # set the number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    
    # Calculate the silhouette score
    score = silhouette_score(pixels, labels)
    print(f"PSO Silhouette score: {score}")
    
    # Threshold the image with the global best position
    threshold_value = 50
    ret, binary_img = cv2.threshold(segmented_image, threshold_value, 255, cv2.THRESH_BINARY)
    reversed_img = 255 - binary_img
    
    # Apply morphological operations to remove small objects and fill in holes
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(reversed_img, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Apply edge detection
    edges = cv2.Canny(ct_image_int_8, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Set minimum area for tumor contour
    min_area = 100
    tumor_extract = np.copy(reversed_img)

    # Loop through contours and draw bounding box around tumor
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(tumor_extract, (x, y), (x + w, y + h), (0, 255, 0), 1)

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    fig.suptitle("PSO Clustering")
    ax[0].imshow(ct_image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(binary_img, cmap='gray')
    ax[1].set_title('Binary Image')
    ax[2].imshow(reversed_img, cmap='gray')
    ax[2].set_title("Reverse Binary Extraction")
    ax[3].imshow(tumor_extract, cmap='gray')
    ax[3].set_title("Tumor Extraction")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax[3].add_patch(rect)
    plt.show()
