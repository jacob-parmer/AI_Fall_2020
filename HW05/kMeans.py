"""
### Author: Jacob Parmer
###
### Last Updated: October 13, 2020
"""

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import sys
from pudb import set_trace
from copy import deepcopy


class kMean:

    """
    Initializes kMean object. This generates random center values to be clustered around in the
    kMeans algorithm.

    PARAMS: k (int) - Number of cluster centers desired
            data (list of tuples) - List of data points in format (x, y)

    """
    def __init__(self, k, data):
        self.k = k
        self.data = data
        
        self.centers = []

        for i in range(k):

            # Generates a cluster center at a random position, with the upper limit set to the
            # max value of the data point's x and y values.
            center_x = random.uniform(0, max(data[0]))
            center_y = random.uniform(0, max(data[1]))        
            center = (center_x, center_y)
            self.centers.append(center)

    """
    Determines Euclidean Distance between two points.

    PARAMS: point1 (tuple) - Datapoint in the form of (x, y)
            point2 (tuple) - Second datapoint in the form of (x, y)

    RETURNS: eucl_dist (float) - Determined distance between the two points

    """
    def distance(self, point1, point2):
        eucl_dist = (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
        eucl_dist = math.sqrt(eucl_dist)

        return eucl_dist

    """
    Determines clusters given the object's current cluster center values. Does this by computing 
    distances between each point and each respective cluster center, and  outputting to a table 
    of clusters.

    RETURNS: clusters (list of list of tuples) - Each data point separated into its own separate
                                                 list of tuples

    """
    def cluster(self):
        
        clusters = [[] for x in range(self.k)]
        
        for point in self.data:
            min_dist = np.inf
            closest_center = -1
            for k, center in enumerate(self.centers):
                dist = self.distance(point, center)
                if (dist < min_dist):
                    min_dist = dist
                    closest_center = k
                    
            clusters[closest_center].append(point)        

        return clusters     

    """
    Moves the cluster centers to the mean value of the datapoints that are contained inside its
    cluster.

    PARAMS: clusters (list of list of tuples) - Each data point separated into its own separate
                                                list of tuples

    """
    def iterate(self, clusters):

        for k, cluster in enumerate(clusters):

            if (len(cluster) == 0):
                print("Unlucky random positions led to cluster of size 0.")
                print("Solution cannot be found in this case. Exiting...")
                sys.exit()

            mean_x = 0.0
            mean_y = 0.0
            sum_x = 0
            sum_y = 0
            for point in cluster:
                sum_x = sum_x + point[0]
                sum_y = sum_y + point[1]

            mean_x =  sum_x / len(cluster)  
            mean_y = sum_y / len(cluster)

            self.centers[k] = (mean_x, mean_y)


    """
    Implements the actual kMeans algorithm. Determines clusters given the current cluster centers,
    iterates to find new cluster centers, and then repeats until the centers are no longer moving.

    RETURNS: clusters (list of list of tuples) - Each data point separated into its own separate
                                                 list of tuples

    """
    def find(self):

        clusters = []

        center_distance = 1 # Set this to greater than 0.0 initially so that the loop will run 

        # Detects when the cluster centers stop moving
        while (center_distance != 0.0):

            centers_temp = deepcopy(self.centers)
            center_distance = 0
            
            clusters = self.cluster()
            self.iterate(clusters)
            
            for k, center in enumerate(self.centers):
                center_distance += self.distance(self.centers[k], centers_temp[k])
                

        return clusters 

    """
    Displays a matplotlib plot with each cluster separated by color.

    PARAMS: clusters (list of list of tuples) - Each data point separated into its own separate
                                                list of tuples

    """
    def display(self, clusters):

        fig, ax = plt.subplots()
        for k, cluster in enumerate(clusters):
            cluster.append(self.centers[k])
            ax.scatter(*zip(*cluster))

        plt.title("Clusters determined by kMeans")    
    
        plt.show()

    """
    Determines the value of the distortion function given the current clusters in the K-Means
    algorithm. 

    PARAMS: clusters (list of list of tuples) - Each data point separated into its own separate
                                                list of tuples

    RETURNS: total_distort (float) - Result of distortion function                                            
    """
    def distortion_function(self, clusters):
        S = 0

        total_distort = 0
        for k, cluster in enumerate(clusters):
            cluster_distort = 0
            for point in cluster:
                
                distort = abs(point[0] - self.centers[k][0]) + abs(point[1] - self.centers[k][1])
                distort = distort**2

                cluster_distort += distort

            total_distort += cluster_distort
    
        return total_distort


def main():

    k = 3
    data = []

    with open('A.txt', 'r') as reader:
        for line in reader.readlines():
            line = line.replace('\n', '').split(' ')
            line[0] = float(line[0])
            line[1] = float(line[1])
            line = tuple(line)
            data.append(line)

   
    result = kMean(k, data)
    clusters = result.find()
    print(result.distortion_function(clusters))
    result.display(clusters)

if __name__ == "__main__":
    main()

