import numpy as np

num_clusters = 2
num_iterations = 15
sc = 0
fc = 0
num_success = 0
num_failures = 0

class Particle:
    def init(self, num_clusters, data):
        self.num_clusters = num_clusters
        self.position = np.random.uniform(low=0, high=1, size=(data.shape[1], num_clusters))
        self.velocity = np.random.uniform(low=0, high=1, size=(data.shape[1], num_clusters))
        self.best_position = self.position
        self.best_fitness = float('-inf')

    def update_velocity(self, global_best_position, w=0.729, c1=1.49445, c2=1.49445):
        r1 = np.random.uniform(low=0, high=1, size=self.position.shape)
        r2 = np.random.uniform(low=0, high=1, size=self.position.shape)
        cognitive_component = c1 * r1 * (self.best_position - self.position)
        social_component = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_component + social_component

def update_position(self, data, bounds):
    self.position = self.position + self.velocity
    self.position = np.clip(self.position, bounds[0], bounds[1])
    
def get_clusters(self, data):
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis, :] - self.position.T, axis=2), axis=1)
    clusters = [data[labels == i] for i in range(self.num_clusters)]
    centroids = self.position.T
    return clusters, centroids

def fitness_function(position, data):
    """
    Calculates the fitness value of a set of clusters and their centroids.
    The fitness is defined as the ratio of the sum of the intercluster distances
    to the sum of the intracluster distances, where intercluster distances are
    the distances between different clusters and intracluster distances are the
    distances between points within the same cluster.

    Args:
        position (array): An array of shape (n_features, n_clusters) representing the centroids
                        of each cluster.
        data (array): An array of shape (n_samples, n_features) representing the data points.

    Returns:
        float: The fitness value.
    """
    n_clusters = position.shape[1]
    intercluster_distances = np.zeros((n_clusters, n_clusters))
    intracluster_distances = np.zeros(n_clusters)

    # Calculate intercluster distances
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            intercluster_distances[i, j] = np.linalg.norm(position[:, i] - position[:, j])
            intercluster_distances[j, i] = intercluster_distances[i, j]

    # Calculate intracluster distances
    for i in range(n_clusters):
        if len(data[i]) > 1:
            centroid_distances = np.linalg.norm(data[i] - position[:, i], axis=1)
            intracluster_distances[i] = np.sum(centroid_distances) / (len(data[i]) - 1)

    # Calculate fitness value
    fitness = np.sum(intercluster_distances) / np.sum(intracluster_distances)
    return fitness

data = np.random.rand(100, 5)
bounds = (0, 1)

global_best_position = np.random.uniform(low=0, high=1, size=(data.shape[1], num_clusters))
global_best_fitness = float('-inf')

particles = [Particle(num_clusters, data) for _ in range(10)]

for i in range (num_iterations):
    for p