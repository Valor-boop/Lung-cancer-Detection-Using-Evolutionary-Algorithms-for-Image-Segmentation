####################################################################################
#   CISC455 Group 4
#   Lung cancer Detection Using Evolutionary Algorithms for Image Segmentation
#   Names: Patrick Bernhard, Pavel-Dumitru Cernelev, Ben Tomkinson
#   Date: 4/11/2023
#####################################################################################
from kmeans_cluster import kmeans_segmentation
from pso_cluster import perform_cluster
import numpy as np
def main():
    ct_image_1 = np.load('455_LungCT\R_006_94.npy')
    ct_image_2 = np.load('455_LungCT\R_004_45.npy')
    kmeans_segmentation(ct_image_1)
    perform_cluster(ct_image_1)
    kmeans_segmentation(ct_image_2)
    perform_cluster(ct_image_2)
main()
