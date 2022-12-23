import cv2
import numpy as np
import os
import time


def skewFromVec(vec):
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])


def triangulate(P, points):
    num_pts = points.shape[0]
    num_proj_mats = len(P)
    points3D = []
    for p in range(num_pts):
        A = np.zeros((num_proj_mats*3, 4))
        for pp in range(num_proj_mats):
            pt_cur_homo = np.array([
                points[p, pp*2],
                points[p, pp*2+1],
                1])
            A[pp*3:(pp+1)*3, :] = np.matmul(skewFromVec(pt_cur_homo), P[pp])

        _, _, V = np.linalg.svd(A)
        V = V.T
        sol_X = V[:, -1].T
        if sol_X[-1] != 0:
            sol_X = sol_X / sol_X[-1]
            if np.all(np.abs(sol_X[:-1]) < 1):
                points3D.append(sol_X[:-1].T)

    return np.array(points3D)


class SGM:
    def __init__(self, folder, leftPath="img1.png", rightPath="img2.png", max_disparity=64, P1=10, P2=120, pointCloud=False):
        self.folder = folder
        self.dmax = max_disparity
        self.P1, self.P2 = P1, P2
        self.cwSize = 7
        self.bwSize = 3
        subfolder = "disp{}_{}_{}".format(max_disparity, P1, P2)

        if not os.path.exists(os.path.join(folder, subfolder)):
            os.mkdir(os.path.join(folder, subfolder))

        # Load images + Gaussian blur
        self.leftImage, self.rightImage = self.loadImage(
            leftPath), self.loadImage(rightPath)
        print("Image loaded")
        self.imgH, self.imgW = self.leftImage.shape

        # Get census
        if os.path.exists(os.path.join(folder, "census", "left_census.npy")):
            with open(os.path.join(folder, "census", "left_census.npy"), "rb") as f:
                self.leftCensus = np.load(f)
            print("Left census loaded")
        else:
            tik =  time.time()
            self.leftCensus = self.encode(self.leftImage)
            print("Img1 census computed: {:.2f}s".format(time.time()-tik))
            cv2.imwrite(os.path.join(folder, "census", "left_census.png"), self.leftCensus.astype(np.uint8))
            with open(os.path.join(folder, "census", "left_census.npy"), "wb") as f:
                np.save(f, self.leftCensus)

        if os.path.exists(os.path.join(folder, "census", "right_census.npy")):
            with open(os.path.join(folder, "census", "right_census.npy"), "rb") as f:
                self.rightCensus = np.load(f)
            print("Right census loaded")
        else:
            tik = time.time()
            self.rightCensus = self.encode(self.rightImage)
            print("Img2 census computed: {:.2f}s".format(time.time()-tik))
            cv2.imwrite(os.path.join(folder, "census", "right_census.png"), self.rightCensus.astype(np.uint8))
            with open(os.path.join(folder, "census", "right_census.npy"), "wb") as f:
                np.save(f, self.rightCensus)

        # Compute cost volume
        if os.path.exists(os.path.join(folder, subfolder, "left_CV.npy")) and os.path.exists(os.path.join(folder, subfolder, "right_CV.npy")):
            with open(os.path.join(folder, subfolder, "left_CV.npy"), "rb") as f:
                self.leftCV = np.load(f)
            print("LeftCV loaded")
            with open(os.path.join(folder, subfolder, "right_CV.npy"), "rb") as f:
                self.rightCV = np.load(f)
            print("rightCV loaded")
        else:
            tik = time.time()
            self.leftCV, self.rightCV = self.getCostVolumes()
            print("Cost volumes computed: {:.2f}s".format(time.time()-tik))
            # cv2.imwrite(os.path.join(folder, subfolder, 'left_CV.png'), self.getDisparityMap(self.leftCV))
            # cv2.imwrite(os.path.join(folder, subfolder, 'right_CV.png'), self.getDisparityMap(self.rightCV))
            with open(os.path.join(folder, subfolder, "left_CV.npy"), "wb") as f:
                np.save(f, self.leftCV)
            with open(os.path.join(folder, subfolder, "right_CV.npy"), "wb") as f:
                np.save(f, self.rightCV)

        # Aggregate cost volume
        if os.path.exists(os.path.join(folder, subfolder, "left_ACV.npy")):
            with open(os.path.join(folder, subfolder, "left_ACV.npy"), "rb") as f:
                self.leftACV = np.load(f)
            print("leftACV loaded")
        else:
            print("Start computing aggregated left cost volume...\t", end="")
            self.leftACV = self.getAggregatedCostVolume(self.leftCV)
            with open(os.path.join(folder, subfolder, "left_ACV.npy"), "wb") as f:
                np.save(f, self.leftACV)
            print("Computed")
        if os.path.exists(os.path.join(folder, subfolder, "right_ACV.npy")):
            with open(os.path.join(folder, subfolder, "right_ACV.npy"), "rb") as f:
                self.rightACV = np.load(f)
            print("rightACV loaded")
        else:
            print("Start computing aggregated right cost volume...\t", end="")
            self.rightACV = self.getAggregatedCostVolume(self.rightCV)
            with open(os.path.join(folder, subfolder, "right_ACV.npy"), "wb") as f:
                np.save(f, self.rightACV)
            print("Computed")

        # Disparity refine
        self.leftDisparity = np.argmin(self.leftACV, axis=2)
        cv2.imwrite(os.path.join(folder, subfolder, 'left_disp_map.png'), self.normalize(self.leftDisparity))
        with open(os.path.join(folder, subfolder, "left_disparity_map.npy"), "wb") as f:
            np.save(f, self.leftDisparity)

        self.rightDisparity = np.argmin(self.rightACV, axis=2)
        cv2.imwrite(os.path.join(folder, subfolder, 'right_disp_map.png'), self.normalize(self.rightDisparity))
        with open(os.path.join(folder, subfolder, "right_disparity_map.npy"), "wb") as f:
            np.save(f, self.rightDisparity)

        with open(os.path.join(folder, subfolder, "left_disparity_map.npy"), "rb") as f:
            self.leftDisparity = np.load(f)
        with open(os.path.join(folder, subfolder, "right_disparity_map.npy"), "rb") as f:
            self.rightDisparity = np.load(f)
        self.disparity = self.getDisparity()

        cv2.imwrite(os.path.join(folder, subfolder, 'result.png'), cv2.medianBlur(self.normalize(self.disparity), self.bwSize))

        if pointCloud:
        # Get 3D point cloud
            print("Getting 3D point cloud...\t", end="")
            self.disparity = self.getDisparity()
            self.leftProjMat = self.loadProjectionMat("P1.txt")
            self.rightProjMat = self.loadProjectionMat("P2.txt")
            self.imgPoints = self.getImagePoints()
            self.point3D = triangulate(
                [self.leftProjMat, self.rightProjMat], self.imgPoints)
            rows = ["{},{},{},255,255,255".format(i, j, k) for i, j, k in self.point3D]
            with open(os.path.join(folder, subfolder, 'point3D.csv'), 'w') as f:
                f.write("\n".join(rows))
            print("Computed")
        print("Finished")

    def getDisparity(self):
        res = np.copy(self.leftDisparity)
        for r in range(self.imgH):
            for c in range(self.imgW):
                displ = self.leftDisparity[r, c]
                cr = c - displ
                if 0 <= cr < self.imgW:
                    dispr = self.rightDisparity[r, cr]
                    if abs(displ - dispr) <= 1:
                        continue
                res[r, c] = 0
        return res

    def getImagePoints(self, step=2):
        pts = []
        for i in range(0, self.imgH, step):
            for j in range(self.imgW//10, self.imgW, step):
                pts.append([i, j, i-self.leftDisparity[i, j], j])
        return np.array(pts, dtype=np.float)

    def loadProjectionMat(self, fn):
        with open(os.path.join(self.folder, fn), 'r') as f:
            proj_mat = np.zeros((3, 4), dtype=np.float)
            for (i, line) in enumerate(f.readlines()):
                line = line.strip().split(' ')
                for (j, val) in enumerate(line):
                    proj_mat[i, j] = float(val)
        return proj_mat

    def loadImage(self, fn):
        img = cv2.imread(os.path.join(self.folder, fn), 0)
        img = cv2.GaussianBlur(img, (self.bwSize, self.bwSize), 0, 0)
        return img

    def encode(self, img):
        offset = self.cwSize // 2
        census = np.zeros((self.imgH, self.imgW), dtype=np.uint64)

        for r in range(offset, self.imgH-offset):
            for c in range(offset, self.imgW-offset):
                bin = np.int64(0)
                center_val = img[r, c]
                ref = np.full((self.cwSize, self.cwSize),
                              fill_value=center_val, dtype=np.int64)
                crop = img[r-offset:r+offset+1, c-offset:c+offset+1]
                diff = crop - ref
                for j in range(diff.shape[0]):
                    for i in range(diff.shape[1]):
                        if (i, j) != (offset, offset):
                            bin = bin << 1
                            bit = 1 if diff[j, i] < 0 else 0
                            bin = bin | bit
                census[r, c] = bin
        return census

    def getCostVolumes(self):
        offset = self.cwSize // 2
        left_CV = np.zeros((self.imgH, self.imgW, self.dmax), dtype=np.uint32)
        right_CV = np.zeros((self.imgH, self.imgW, self.dmax), dtype=np.uint32)
        lcensus = np.zeros((self.imgH, self.imgW), dtype=np.int64)
        rcensus = np.zeros((self.imgH, self.imgW), dtype=np.int64)
        for d in range(self.dmax):
            rcensus[:, (offset+d):(self.imgW-offset)
                    ] = self.rightCensus[:, offset:(self.imgW-d-offset)]
            left_xor = np.int64(np.bitwise_xor(
                np.int64(self.leftCensus), rcensus))
            left_distance = np.zeros((self.imgH, self.imgW), dtype=np.uint32)
            while not np.all(left_xor == 0):
                tmp = left_xor - 1
                mask = left_xor != 0
                left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
                left_distance[mask] = left_distance[mask] + 1
            left_CV[:, :, d] = left_distance

            lcensus[:, offset:(self.imgW-d-offset)] = self.leftCensus[:,
                                                                      (offset + d):(self.imgW-offset)]
            right_xor = np.int64(np.bitwise_xor(
                np.int64(self.rightCensus), lcensus))
            right_distance = np.zeros((self.imgH, self.imgW), dtype=np.uint32)
            while not np.all(right_xor == 0):
                tmp = right_xor - 1
                mask = right_xor != 0
                right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
                right_distance[mask] = right_distance[mask] + 1
            right_CV[:, :, d] = right_distance

        return left_CV, right_CV

    def getDisparityMap(self, arr):
        return self.normalize(np.argmin(arr, axis=2))

    def normalize(self, arr):
        return np.uint8(255.0 * arr / self.dmax)

    def getPathCost(self, slice):
        dim1, dim2 = slice.shape

        disparity_mask = [d for d in range(dim2)] * dim2
        disparity_mask = np.array(disparity_mask).reshape(dim2, dim2)

        Pmat = np.zeros((dim2, dim2), dtype=slice.dtype)
        Pmat[np.abs(disparity_mask-disparity_mask.T) == 1] = self.P1
        Pmat[np.abs(disparity_mask-disparity_mask.T) > 1] = self.P2

        res = np.copy(slice)
        for i in range(1, dim1):
            prev_cost = res[i-1, :]
            cur_cost = slice[i, :]
            costMat = np.repeat(prev_cost, repeats=dim2,
                                axis=0).reshape(dim2, dim2)
            costMat = np.amin(costMat+Pmat, axis=0)
            res[i, :] = cur_cost + costMat - np.amin(prev_cost)
        return res

    def getAggregatedCostVolume(self, CV):
        ACV = np.zeros((self.imgH, self.imgW, self.dmax, 8), dtype=np.uint32)
        paths = [
            [(1, 0), (-1, 0)],
            [(0, 1), (0, -1)],
            [(1, 1), (-1, -1)],
            [(-1, 1), (1, -1)]
        ]

        for (i, path) in enumerate(paths):
            print("Aggregating direction {} and {}...".format(
                path[0], path[1]))
            tik = time.time()
            agg1 = np.zeros((self.imgH, self.imgW, self.dmax), dtype=np.uint32)
            agg2 = np.copy(agg1)

            if i == 0:
                for r in range(self.imgH):
                    a = CV[r, :, :]
                    b = np.flip(a, axis=0)
                    agg1[r, :, :] = self.getPathCost(a)
                    agg2[r, :, :] = np.flip(self.getPathCost(b))
            elif i == 1:
                for c in range(self.imgW):
                    a = CV[:, c, :]
                    b = np.flip(a, axis=0)
                    agg1[:, c, :] = self.getPathCost(a)
                    agg2[:, c, :] = np.flip(self.getPathCost(b))
            elif i == 2:
                for x in range(1-self.imgH, self.imgW-1):
                    a = CV.diagonal(offset=x).T
                    b = np.flip(a, axis=0)
                    dim = a.shape[0]
                    rList1 = np.array(
                        [-x+j if x < 0 else j for j in range(dim)])
                    cList1 = np.array(
                        [j if x < 0 else x+j for j in range(dim)])
                    rList2, cList2 = np.flip(rList1, axis=0), np.flip(cList1, axis=0)
                    agg1[rList1, cList1, :] = self.getPathCost(a)
                    agg2[rList2, cList2, :] = self.getPathCost(b)
            elif i == 3:
                for x in range(1-self.imgH, self.imgW-1):
                    a = np.flipud(CV).diagonal(offset=x).T
                    b = np.flip(a, axis=0)
                    dim = a.shape[0]
                    rList1 = np.array(
                        [self.imgH-1+x-j if x < 0 else self.imgH-1-j for j in range(dim)])
                    cList1 = np.array(
                        [j if x < 0 else x+j for j in range(dim)])
                    rList2, cList2 = np.flip(rList1, axis=0), np.flip(cList1, axis=0)
                    agg1[rList1, cList1, :] = self.getPathCost(a)
                    agg2[rList2, cList2, :] = self.getPathCost(b)
            ACV[:, :, :, i*2] = agg1
            ACV[:, :, :, i*2+1] = agg2
            print("Done in {:.2f}s".format(time.time() - tik))
        return np.sum(ACV, axis=3)


if __name__ == '__main__':
    root = os.getcwd()
    prefix = "dataset/data1"
    sgm = SGM(
        folder=os.path.join(root, prefix),
        max_disparity=64,
        leftPath="img1.png",
        rightPath="img2.png"
        P1=10,
        P2=120)
