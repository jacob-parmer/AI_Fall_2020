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

    def __init__(self, filename):
        with open(filename, 'r') as reader:
            self.points = [] 
            for line in reader.readlines():
                line = line.replace('\n', '').split(' ')
                line[0] = float(line[0])
                line[1] = float(line[1])
                line = tuple(line)
                self.points.append(line)

    def getData(self):
        return self.points
    
    def distance(self, index1, index2):
        distance = (self.points[index1][0] - self.points[index2][0])**2 + (self.points[index1][1] - self.points[index2][1])**2

        distance = math.sqrt(distance)

        return distance

class HeirarchicalClustering:

    def __init__(self, data):
        self.data = data
        self.clusters = [[data.points[i]] for i in range(len(data.points))]
        self.numOfClusters = len(self.clusters)

    def getAllDistances(self):

        distances = np.zeros((len(self.data.points), len(self.data.points)), dtype=float)

        for i, point1 in enumerate(self.data.points):
            for j, point2 in enumerate(self.data.points):
                distance = self.data.distance(i, j)
                if (i == j): # Returns distance inf if testing distance between a point and itself
                    distance = np.inf

                distances[i][j] = distance

        print(distances)
        return distances

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

        # Pops out second cluster, joins it to first cluster for one bigger cluster
        second_cluster = self.clusters.pop(second_cluster_index)
        self.clusters[first_cluster_index] = self.clusters[first_cluster_index] + second_cluster
        self.numOfClusters = self.numOfClusters - 1
        

    def HAC(self, distances, finalClusterNum):
            
        while (self.numOfClusters != finalClusterNum):
            index = np.argmin(distances)
            index = np.unravel_index(index, distances.shape)
            distances[index[0], index[1]] = np.inf
            distances[index[1], index[0]] = np.inf

            self.cluster(index[0], index[1])

    def getClusters(self):
        return self.clusters

class Display:

    def __init__(self, first_cluster, second_cluster):
        self.fig, self.ax = plt.subplots()
        self.ax.scatter(*zip(*first_cluster))
        self.ax.scatter(*zip(*second_cluster))

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
