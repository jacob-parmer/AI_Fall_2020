"""
### Author: Jacob Parmer, Auburn University
###
### Last Updated: September 22, 2020
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from pudb import set_trace


class Data:

    """
    Constructor for Data object. Creates a list of tuples called "points" that contains all the 
    points in a float(x,y) fashion. These points are defined in the file listed as "filename".

    """
    def __init__(self, filename):
        with open(filename, 'r') as reader:
            self.points = [] 
            for line in reader.readlines():
                line = line.replace('\n', '').split(' ')
                line[0] = float(line[0])
                line[1] = float(line[1])
                line = tuple(line)
                self.points.append(line)

    """
    Will return datapoints in the object.

    """
    def getData(self):
        return self.points
    
    """
    Calculates and returns the Euclidian distance between two points, specified by index1 and
    index2. 

    """
    def distance(self, index1, index2):
        distance = (self.points[index1][0] - self.points[index2][0])**2 + (self.points[index1][1] - self.points[index2][1])**2

        distance = math.sqrt(distance)

        return distance


class HeirarchicalClustering:

    """
    Initialies a HeirarchicalClustering object. This object takes in a Data object, and uses this
    to keep track of clusters, and the number of clusters currently in the object.
    
    """
    def __init__(self, data):
        self.data = data
        self.clusters = [[data.points[i]] for i in range(len(data.points))]
        self.numOfClusters = len(self.clusters)

    """
    This function returns a numpy array of size (len(datapoints), len(datapoints)). Each position
    in this array tells us the distance between two points in our data. Distance self-comparisons
    are set to have an infinite distance for the purpose of finding the minimums in the array.

    """
    def getAllDistances(self):

        distances = np.zeros((len(self.data.points), len(self.data.points)), dtype=float)

        for i, point1 in enumerate(self.data.points):
            for j, point2 in enumerate(self.data.points):
                distance = self.data.distance(i, j)
                if (i == j): # Returns distance inf if testing distance between a point and itself
                    distance = np.inf

                distances[i][j] = distance

        return distances

    """
    Takes two points at index1 and index2 of the original dataset, finds them in the object's 
    denoted clusters, and merges the two clusters where the two points were found.

    """
    def cluster(self, index1, index2):

        point1 = self.data.points[index1]
        point2 = self.data.points[index2]

        first_cluster_index = 0
        second_cluster_index = 0
        for i, cluster in enumerate(self.clusters):
            for item in cluster:
                
                if (item == point1 and first_cluster_index == 0):
                    first_cluster_index = i
                elif (item == point2):
                    second_cluster_index = i

        # If both points are in the same cluster, no need to merge them.
        if (first_cluster_index == second_cluster_index):
            return    

        # Pops out second cluster, joins it to first cluster for one bigger cluster
        # If statement is necessary in case .pop screws up the indexing
        second_cluster = self.clusters.pop(second_cluster_index)
        if (first_cluster_index < second_cluster_index):
            self.clusters[first_cluster_index] = self.clusters[first_cluster_index] + second_cluster
        else:
            self.clusters[first_cluster_index - 1] = self.clusters[first_cluster_index - 1] + second_cluster

        self.numOfClusters = self.numOfClusters - 1
        

    """
    Implements HAC algorithm to cluster together similar points. This takes an array of distances
    that can be gotten by calling .getAllDistances(). The algorithm searches for the minimum 
    distance in our array, grabs the indexes of the datapoints that created that minimum, and
    clusters those two datapoints. It also sets the used index of distances to infinity so that it
    does not grab the same two points over and over again.

    """
    def HAC(self, distances, finalClusterNum):
            
        while (self.numOfClusters != finalClusterNum):
            index = np.argmin(distances)
            index = np.unravel_index(index, distances.shape)
            distances[index[0], index[1]] = np.inf
            distances[index[1], index[0]] = np.inf

            self.cluster(index[0], index[1])

    """
    Returns a list of clusters currently found in the object.

    """
    def getClusters(self):
        return self.clusters


class Display:

    """
    Initializes a display object to be used to display the clusters found in the program.
    Takes in two clusters on the assumption that this will be used for HW04. This 
    isn't generalizable in any way and is generally bad. But this whole thing is generally bad.
    >:(

    """
    def __init__(self, first_cluster, second_cluster):
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(*zip(*first_cluster))
        self.ax.scatter(*zip(*second_cluster))
        plt.title("Clusters determined by HAC")

    """
    Displays built plot.

    """
    def show(self):
        plt.show()

def main():
    
    #set_trace()
    
    data = Data("B.txt")
    hc = HeirarchicalClustering(data)

    dist = hc.getAllDistances()
    hc.HAC(dist, 2)

    clst = hc.getClusters()
    dsp = Display(clst[0], clst[1])
    dsp.show()

if __name__ == "__main__":
    main()
