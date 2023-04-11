import cv2
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

def kmeans_segmentation(image):
        # Load grayscale image as a numpy array

    # Reshape the array to 2D to apply clustering
    image_2d = image.reshape(-1, 1)

    # Perform K-means clustering with k=8
    kmeans = KMeans(n_clusters=10, random_state=0).fit(image_2d)

    # Replace each pixel with its corresponding cluster center
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape the compressed image back to the original dimensions
    compressed_image = compressed_image.reshape(image.shape)

    # Create a binary mask based on the tumor label
    tumor_label = np.argmax(kmeans.cluster_centers_)
    mask = (kmeans.labels_ == tumor_label).astype(int)
    mask = mask.reshape(image.shape)  # Add this line to reshape the mask

    # Extract the tumor from the compressed image using the mask
    tumor = compressed_image * mask

    # Display the tumor image
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle("Kmeans Clustering")
    ax[0].imshow(image, cmap = 'gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(compressed_image.astype('uint8'), cmap='gray')
    ax[1].set_title('Clustered Image')
    ax[2].imshow(tumor.astype('uint8'), cmap='gray')
    ax[2].set_title("Tumor Extraction")
    plt.show()

    '''
    # Reshape the array to 2D to apply clustering
    image_2d = image.reshape(-1, 1)

    # Perform K-means clustering with k=8
    kmeans = KMeans(n_clusters=8, random_state=0).fit(image_2d)

    # Replace each pixel with its corresponding cluster center
    compressed_image = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape the compressed image back to the original dimensions
    compressed_image = compressed_image.reshape(image.shape)

    if kmeans.cluster_centers_[0] > kmeans.cluster_centers_[1]:
        tumor_label = 0
    else:
        tumor_label = 1

    # Create a binary mask based on the tumor label
    mask = (kmeans.labels_ == tumor_label).astype(int)

# Extract the tumor from the compressed image using the mask
    tumor = compressed_image * mask
    plt.imshow(tumor.astype('uint8'), cmap='gray')
    plt.show()

    # Display the original and compressed images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image, cmap = 'gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(compressed_image.astype('uint8'), cmap='gray')
    ax[1].set_title('Compressed Image')
    plt.show()
    '''