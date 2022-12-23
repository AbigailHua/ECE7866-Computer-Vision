import PIL.Image as Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import maxflow
from skimage import color, io

class GraphImage:
    def __init__(self, imgPath, strokePath, k=64) -> None:
        self.startTime = time.time()
        self.k = k
        self.img = self.openImage(imgPath)
        self.strokeImg = self.openImage(strokePath)
        self.h, self.w = self.img.shape[:2]
        self.imgSize = self.h * self.w
    
    def run(self):
        # K-means + GMM -> data cost
        self.frontStrokeCoord, self.backStrokeCoord = self.getCoordinates()
        self.clusterWeightFront, self.centerValueFront = self.kMeans(foreground=True)
        self.clusterWeightBack, self.centerValueBack = self.kMeans(foreground=False)
        self.probFront, self.probBack = self.getProb()
        # smooth cost
        self.smoothCost = self.getSmoothCost() # (down, right)
        # Graph cut
        self.graph = maxflow.Graph[float]()
        self.foregroundMask, self.backgroundMask = self.graphCut()
        return self.foregroundMask*self.img + self.backgroundMask*255

    def openImage(self, fn):
        print("[Image Read] {}".format(fn))
        img = Image.open(fn)
        return np.asarray(img, dtype=int)[:, :, :3]
    
    def calcDist(self, a, b):
        return np.sqrt(np.sum(np.square(a - b), axis=-1))
    
    def getCoordinates(self):
        front = np.where(self.strokeImg[:, :, 0] == 255)
        back = np.where(self.strokeImg[:, :, 2] == 255)
        return front, back
    
    def getCluster(self, colored_stroke, center_value, num_points):
        # distance[i][j]: the distance between i-th cluster center & j-th point
        distance = np.zeros((self.k, num_points))
        for (i, center) in enumerate(center_value):
            distance[i] = self.calcDist(colored_stroke, center)
        return np.argmin(distance, axis=0)
    
    def kMeans(self, foreground=True):
        coord = self.frontStrokeCoord if foreground else self.backStrokeCoord
        colored_stroke = self.img[coord] # (num, 3)
        num_points = coord[0].size # num
        # Randomly pick k points from all stroke points
        all_idx = [i for i in range(num_points)]
        random.shuffle(all_idx)
        center_idx = all_idx[:self.k] # (k,)
        center_value = colored_stroke[center_idx] # (k, 3)
        p_central_value = np.empty_like(center_value)
        itr = 0
        while not np.array_equal(center_value, p_central_value):
            if itr <= 200:
                itr += 1
                p_central_value = center_value
                index = self.getCluster(colored_stroke, center_value, num_points)
                center_value = np.zeros((self.k, 3))
                cnt = np.zeros(self.k)
                for i in range(num_points):
                    cur_cluster = index[i]
                    center_value[cur_cluster] += colored_stroke[i]
                    cnt[cur_cluster] += 1
                cnt = cnt[:, np.newaxis]
                center_value = np.divide(center_value, cnt, out=np.zeros_like(center_value), where=cnt!=0)
            else:
                p_central_value = center_value
        print("[K-Means] Iteration {}: {}ground converged!".format(itr, "fore" if foreground else "back"))
        return self.getClusterWeight(index), center_value
    
    def getClusterWeight(self, clusters):
        weight = np.zeros(self.k)
        for c in clusters:
            weight[c] += 1
        return weight / clusters.size

    def getProb(self):
        prob_front, prob_back = np.zeros((self.h, self.w)), np.zeros((self.h, self.w))
        for i in range(self.k):
            exp_dist = np.exp(-self.calcDist(self.img, self.centerValueFront[i]))
            prob_front += self.clusterWeightFront[i] * exp_dist
            exp_dist = np.exp(-self.calcDist(self.img, self.centerValueBack[i]))
            prob_back += self.clusterWeightBack[i] * exp_dist
        return prob_front, prob_back
    
    def indexTrans(self, i, j):
        return i*self.w + j
    
    def isInBound(self, i, j):
        return 0 <= i < self.h and 0 <= j < self.w

    def getSmoothDist(self, a, b):
        c = 1/np.square(a-b) # difference between a & b
        return 0.3*c[:, :, 0] + 0.59*c[:, :, 1] + 0.11*c[:, :, 2] # convert rgb diff to grayscale

    def getSmoothCost(self):
        # Calculate the distance between current pixel to its right/bottom neighbours
        # by shifting the image left/up
        right_dist = self.getSmoothDist(self.img, np.roll(self.img, -1, axis=1))
        down_dist = self.getSmoothDist(self.img, np.roll(self.img, -1, axis=0))
        print("[Smooth cost] Smooth cost computed!")
        return down_dist, right_dist

    def graphCut(self):
        nodes = self.graph.add_grid_nodes(self.img.shape[:2])
        dir = [(1, 0), (0, 1)]
        # Add edges between nodes
        for i in range(self.h):
            for j in range(self.w):
                for (idx, (di, dj)) in enumerate(dir):
                    ni, nj = i+di, j+dj
                    if self.isInBound(ni, nj):
                        weight = self.smoothCost[idx][i, j]
                        self.graph.add_edge(nodes[i, j], nodes[ni, nj], weight, weight)
        # Add edges connected to the source and terminal
        self.graph.add_grid_tedges(nodes, self.probFront, self.probBack)
        # Find the max flow
        self.graph.maxflow()
        print("[Graph cut] Max flow computed! Time cost: {}s".format(time.time()-self.startTime))
        segments = self.graph.get_grid_segments(nodes)
        fg = np.int_(np.logical_not(segments))[:, :, None]
        return fg, 1-fg

if __name__ == '__main__':
    random.seed(0)
    root = os.getcwd()
    prefix = "data/Lazysnapping_data/"
    img_dict = {
        "tableball.jpg": ["ballstrokes.png"],
        "van Gogh.PNG": ["van Gogh strokes.png"],
        "Mona-lisa.PNG": ["Mona-lisa stroke 1.png", "Mona-lisa stroke 2.png"],
        "lady.PNG": ["lady strokes1.png", "lady stroke 2.png"],
        "dog.PNG": ["dog stroke.png"],
        "dance.PNG": ["dance stroke 1.png", "dance stroke 2.png"]
    }
    for k in img_dict.keys():
        v = img_dict[k]
        for (idx, v0) in enumerate(v):
            ig = GraphImage(
                imgPath=os.path.join(root, prefix, k),
                strokePath=os.path.join(root, prefix, v0)
            )
            res = ig.run()
            res = Image.fromarray(np.asarray(res, dtype=np.uint8))
            res.save(os.path.join(root, prefix, "{}_result{}.png".format(k[:-4], "_"+str(idx+1) if len(v) > 1 else "")))