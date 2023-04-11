import numpy as np
import matplotlib.pyplot as plt

def depth_modification(image, depth):
    """
    Question 5.
    The following function changes the depth of an inputed image represented by a numpy
    array to the value represented by the "depth" parameter.

    Parameters:
        image (numpy array) : An array representing an image
        depth (int) : A depth value the image will be changed to
    Returns:
        new_depth_image (numpy array) : The image altered to have the new depth value
    """
    # Get max depth of image
    current_max = np.max(image)
    i = 1
    while 2 ** i < current_max:
        i += 1
    possible_max = 2 ** i - 1
    new_depth_image = np.copy(image)
    for i in range(0, len(image[0])):
        for j in range(0, len(image)):
            new_intensity = round(image[j][i]/possible_max * 2**depth, 0)
            new_depth_image[j][i] = new_intensity
    return new_depth_image

def linear_filters(image, filter):
    padding = len(filter) // 2
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    filtered_image = np.copy(image)
    for i in range(0, len(padded_image[0]) - padding - 1):
        for j in range(0, len(padded_image) - padding - 1):
            matrix_section = padded_image[j:j + len(filter), i:i + len(filter)]
            matrix = np.multiply(matrix_section, filter)
            filtered_image[j][i] = matrix.sum()
    return filtered_image

def pre_process(image, depth):
    # Assume the input matrix A
    #image = np.load(file_path)

    # Construct a matrix with M+2 rows and N+2 columns by appending zeros
    M, N = image.shape
    A_padded = np.zeros((M+2, N+2))
    A_padded[1:-1, 1:-1] = image

    # Take a mask of size 3x3
    mask = np.ones((3, 3))

    # Loop through each element in A and replace it with the median of the surrounding 3x3 elements
    for i in range(1, M+1):
        for j in range(1, N+1):
            submatrix = A_padded[i-1:i+2, j-1:j+2] * mask
            median_val = np.median(submatrix)
            image[i-1, j-1] = median_val

    image = depth_modification(image, depth)
    return image


