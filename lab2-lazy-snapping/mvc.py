import numpy as np
import PIL.Image as Image
import os
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class MVC:
    def __init__(self, srcPath, tgtPath, maskPath, resPath) -> None:
        self.srcImage = self.openImage(srcPath)
        self.tgtImage = self.openImage(tgtPath)
        self.maskImage = self.openImage(maskPath)
        self.borderPoints = self.getBorders()
        self.diff = self.getDiff()
        self.result = self.interp()
        resImage = Image.fromarray(self.result)
        resImage.save(resPath)
    
    def openImage(self, fn):
        return np.array(Image.open(fn), dtype=int)
    
    def getBorders(self):
        self.maskImage = np.asarray(self.maskImage[:, :, 0]/255, dtype=np.uint8)
        white = np.nonzero(self.maskImage)
        up, down = np.min(white[0]), np.max(white[0])
        left, right = np.min(white[1]), np.max(white[1])
        self.bounds = [up, down, left, right]
        return [(r, left) for r in range(up, down)] + [(down, c) for c in range(left, right)] \
            + [(r, right) for r in range(down, up, -1)] + [(up, c) for c in range(right, left, -1)]
        
    def getHalfTan(self, v1, v2, mag1, mag2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        cos = dot / mag1 / mag2
        assert 0 <= cos <= 1
        return math.sqrt((1-cos)/(1+cos))
        
    def getMVC(self, r, c, mags):
        weights = np.zeros(len(self.borderPoints))
        for i in range(len(self.borderPoints)):
            if i == 0:
                p0, p1, p2 = self.borderPoints[-1], *self.borderPoints[:2]
                mag0, mag1, mag2 = mags[-1], *mags[:2]
            elif i == len(self.borderPoints)-1:
                p0, p1, p2 = *self.borderPoints[-2:], self.borderPoints[0]
                mag0, mag1, mag2 = *mags[-2:], mags[0]
            else:
                p0, p1, p2 = self.borderPoints[i-1:i+2]
                mag0, mag1, mag2 = mags[i-1:i+2]
            x_p0 = (p0[0]-r, p0[1]-c)
            x_p1 = (p1[0]-r, p1[1]-c)
            x_p2 = (p2[0]-r, p2[1]-c)
            tg1, tg2 = self.getHalfTan(x_p0, x_p1, mag0, mag1), self.getHalfTan(x_p1, x_p2, mag1, mag2)
            weights[i] = (tg1+tg2) / mag1
        return weights / np.sum(weights)
    
    def getMag(self, r, c):
        bp = np.array(self.borderPoints)
        bp = bp - np.array([r, c])
        mags = np.sqrt(bp[:, 0] * bp[:, 0] + bp[:, 1] * bp[:, 1])
        return mags

    def getDiff(self):
        innerMask = np.copy(self.maskImage)
        innerMask[self.bounds[0]+1:self.bounds[1], self.bounds[2]+1:self.bounds[3]] = 0
        return (self.tgtImage - self.srcImage) * innerMask[:, :, None]
    
    def interp(self):
        result = np.copy(self.tgtImage)
        for r in tqdm(range(self.bounds[0]+1, self.bounds[1])):
            for c in range(self.bounds[2]+1, self.bounds[3]):
                MVC_Coords = self.getMVC(r, c, self.getMag(r, c))
                rx = 0
                for (i, b) in enumerate(self.borderPoints):
                    rx += self.diff[b[0], b[1]] * MVC_Coords[i]
                result[r, c] = self.srcImage[r, c] + rx
        return np.asarray(result, dtype=np.uint8)


if __name__ == '__main__':
    root = os.getcwd()
    prefix = "data/Poisson_editing/data2"
    result = MVC(
        srcPath=os.path.join(root, prefix, "source.png"),
        tgtPath=os.path.join(root, prefix, "background.jpg"),
        maskPath=os.path.join(root, prefix, "mask.png"),
        resPath=os.path.join(root, prefix, "MVC.png")
    )