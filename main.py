from KmeansTest import kmeans_segmentation
import numpy as np
import matplotlib.pyplot as plt
def main():
    ct_image = np.load('Saved_DCM_Files\R_006\R_006_94.npy')
    kmeans_segmentation(ct_image)
main()