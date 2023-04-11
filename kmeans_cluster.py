####################################################################################
#   CISC455 Group 4
#   Lung cancer Detection Using Evolutionary Algorithms for Image Segmentation
#   Names: Patrick Bernhard, Pavel-Dumitru Cernelev, Ben Tomkinson
#   Date: 4/11/2023
#####################################################################################
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score

def kmeans_segmentation(image):
    '''
    Applies K-means clustering to a grayscale image to segment a tumor.

    Args:
        image (numpy.ndarray): 2D numpy array, representing the input CT image.

    Returns:
        None
    '''
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
    score = silhouette_score(image_2d, kmeans.predict(image_2d))
    print(f"K Means Silhouette score: {score}")

    # Extract the tumor from the compressed image using the mask
    tumor = compressed_image * mask
    compressed_image = compressed_image.astype(np.uint8)
    
    # Display the tumor image
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle("Kmeans Clustering 004_45")
    ax[0].imshow(image, cmap = 'gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(compressed_image.astype('uint8'), cmap='gray')
    ax[1].set_title('Clustered Image')
    ax[2].imshow(tumor.astype('uint8'), cmap='gray')
    ax[2].set_title("Tumor Extraction")
    plt.show()
