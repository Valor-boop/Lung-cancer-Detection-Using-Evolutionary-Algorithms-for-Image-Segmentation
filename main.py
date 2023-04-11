from kmeans_cluster import kmeans_segmentation
from pso_cluster import perform_cluster
import numpy as np
import matplotlib.pyplot as plt
def main():
    ct_image = np.load('Saved_DCM_Files\R_006\R_006_94.npy')
    kmeans_segmentation(ct_image)
    perform_cluster(ct_image)
main()
