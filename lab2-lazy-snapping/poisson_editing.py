import numpy as np
from scipy.sparse import linalg, identity
import PIL.Image as Image
import os

class Poisson:
    def __init__(self, srcPath, tgtPath, maskPath):
        self.srcImage = self.openImage(srcPath)
        self.tgtImage = self.openImage(tgtPath)
        self.maskImage = self.openImage(maskPath)
        assert self.srcImage.shape == self.tgtImage.shape == self.maskImage.shape
        self.imgShape = self.maskImage.shape[:2]
    
    def openImage(self, fn):
        return np.array(Image.open(fn))
    
    def prepare_mask(self, mask):
        return np.asarray(mask[:, :, 0]/255, dtype=np.uint8)
    
    def blend(self):
        self.maskImage = self.prepare_mask(self.maskImage)
        self.maskImage[self.maskImage==0] = False
        self.maskImage[self.maskImage!=False] = True

        A = identity(np.prod(self.imgShape), format='lil')
        for y in range(self.imgShape[0]):
            for x in range(self.imgShape[1]):
                if self.maskImage[y,x]:
                    index = x+y*self.imgShape[1]
                    A[index, index] = 4
                    if index+1 < np.prod(self.imgShape):
                        A[index, index+1] = -1
                    if index-1 >= 0:
                        A[index, index-1] = -1
                    if index+self.imgShape[1] < np.prod(self.imgShape):
                        A[index, index+self.imgShape[1]] = -1
                    if index-self.imgShape[1] >= 0:
                        A[index, index-self.imgShape[1]] = -1
        A = A.tocsr()
        
        P = 4*identity(self.maskImage.shape[0]*self.maskImage.shape[1], dtype='float', format='lil')
        for r in range(self.maskImage.shape[0]):
            for c in range(self.maskImage.shape[1]):
                i = r * self.maskImage.shape[1]+c
                for (r0, c0) in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                    if 0<=r0<self.maskImage.shape[0] and 0<=c0<self.maskImage.shape[1]:
                        j = r0 * self.maskImage.shape[1]+c0
                        P[i, j] = -1
        result = np.zeros_like(self.tgtImage)
        
        for ch in range(3):
            t = self.tgtImage[:, :, ch].flatten()
            s = self.srcImage[:, :, ch].flatten()

            b = P * s
            for y in range(self.imgShape[0]):
                for x in range(self.imgShape[1]):
                    if not self.maskImage[y,x]:
                        index = x+y*self.imgShape[1]
                        b[index] = t[index]
            x, _ = linalg.cg(A, b)

            x= np.reshape(x, self.imgShape)
            x[x>255] = 255
            x[x<0] = 0
            x = np.array(x, self.tgtImage.dtype)
            result[:, :,ch] = x
        return Image.fromarray(np.uint8(result))

if __name__ == '__main__':
    root = os.getcwd()
    prefix = "data/Poisson_editing/data2"
    result = Poisson(
        srcPath=os.path.join(root, prefix, "source2.png"),
        tgtPath=os.path.join(root, prefix, "background.jpg"),
        maskPath=os.path.join(root, prefix, "mask2.png")
    ).blend()
    result.save(os.path.join(root, prefix, "blend.png"))
