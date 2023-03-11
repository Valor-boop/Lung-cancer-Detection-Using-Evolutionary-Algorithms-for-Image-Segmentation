import numpy as np
import matplotlib.pyplot as plt

# Assume the input matrix A
image = np.load("Saved_DCM_Files\R_004\R_004_0.npy")

# Display the input image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Input Image')

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

# Display the processed image
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Processed Image')

# Show the plots
plt.show()