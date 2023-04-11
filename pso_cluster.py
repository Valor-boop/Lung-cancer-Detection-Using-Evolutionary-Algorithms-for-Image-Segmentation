import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import os 
from sklearn.cluster import KMeans
from skimage.io import imread, imsave
from PIL import Image
from KmeansTest import kmeans_segmentation 
from preprocessing import depth_modification, linear_filters
# Define the fitness function
def fitness(image, thresholds):
    if np.isscalar(thresholds):
        thresholds = np.array([thresholds])
    mask = image > thresholds[:, np.newaxis, np.newaxis]
    return np.sum(image * mask)

# Define the PSO function
def pso(image, num_particles, num_iterations):
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

# Load the CT image as a NumPy array
# lets get this to iterate over all files within a folder nad then find the 
# image that best displays a lung tumor in the patient
# 35 gives good resutls 
# after getting segmented image, any intesnity values > 0, turn to max intesity else turn to 0
# want lungs to appear white, everything else black, isolate tumor based off of surrounding 
# pixel intensities 
# what exactly is the tumor in the ct image 
# promising images: 004_11 looks good
def main():
    ct_image = np.load('Saved_DCM_Files/R_006/R_006_94.npy')
    ct_image = depth_modification(ct_image, 8)
    ct_image = ct_image.astype(np.uint8)    
    segmented_image = pso(ct_image, num_particles=50, num_iterations=100)
    threshold_value = 50
    ret, binary_img = cv2.threshold(segmented_image, threshold_value, 255, cv2.THRESH_BINARY)
    reversed_img = 255 - binary_img
    for i in range(100):
        reversed_img = cv2.medianBlur(reversed_img, 3)
    
    # Apply morphological operations to remove small objects and fill in holes
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(ct_image, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # Apply edge detection
    edges = cv2.Canny(ct_image, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set minimum area for tumor contour
    min_area = 100

    # Loop through contours and draw bounding box around tumor
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(ct_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    file_name = os.path.basename('006_94')
    fig = plt.figure()
    fig.canvas.manager.window.title = file_name

    plt.subplot(141)
    plt.imshow(ct_image, cmap="gray")
    plt.title(file_name)
    plt.axis("off")
    plt.subplot(142)
    plt.imshow(binary_img, cmap="gray")
    plt.title("Raw Image")
    plt.axis("off")
    plt.subplot(143)
    plt.imshow(reversed_img, cmap="gray")
    plt.title("Tumor Image")
    plt.axis("off")
    plt.subplot(144)
    plt.imshow(ct_image, cmap="gray")
    plt.title("Tumor Detection")
    plt.axis("off")
    plt.show()
main()
