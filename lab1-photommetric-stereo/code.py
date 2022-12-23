from pathlib import Path
import numpy as np
import struct
import matplotlib.pyplot as plt
import os
from PIL import Image
import math
import time
import scipy.sparse
import scipy.sparse.linalg

root = os.getcwd()

def rgb2grey(rgb):
    r, g, b = rgb
    return 0.3*r + 0.59*g + 0.11*b

def normalize(vec):
    return vec / np.linalg.norm(vec)

class Solution:
    def __init__(self):
        self.image, self.grayscale = None, None
        self.height, self.width = 0, 0
        self.c1, self.c2, self.cm = None, None, None
        self.r1, self.r2, self.rm = 0, 0, 0
        self.highlight1, self.highlight2 = None, None
        self.L = None

    def get_full_path_name(self, fn):
        return os.path.join(root, "Assignment_1", fn)
    
    def img_coord_shift_to_camera(self, coord):
        return np.array([self.width/2 - coord[0], self.height/2 - coord[1], coord[2]])
    
    def camera_coord_shift_to_img(self, coord):
        return np.array([int(self.width/2 - coord[0]), int(self.height/2 - coord[1]), 0], dtype=int)

    def read_pfm(self, fn):
        """
        A revision of memorial.pfm from http://www.pauldebevec.com/Research/HDR/PFM/
        """
        with Path(self.get_full_path_name(fn)).open('rb') as pfm_file:
            mode, width, height, endian = (pfm_file.readline().decode('latin-1').strip() for _ in range(4))
            assert mode in ('PF', 'Pf')
            width, height, endian = int(width), int(height), float(endian)
            
            channels = 3 if "PF" in mode else 1
            bigendian = endian > 0
            scale = abs(endian)

            buffer = pfm_file.read()
            samples = width * height * channels
            assert len(buffer) == samples * 4
            
            fmt = f'{"<>"[bigendian]}{samples}f'
            decoded = struct.unpack(fmt, buffer)
            shape = (height, width, 3) if channels == 3 else (height, width)
            self.image = np.reshape(decoded, shape, order='C') * scale
            self.height = height
            self.width = width
            self.get_grayscale()
            return self.image

    def get_grayscale(self):
        if self.image is None:
            return
        if self.grayscale is None:
            self.grayscale = 0.3*self.image[:, :, 0] + 0.59*self.image[:, :, 1] + 0.11*self.image[:, :, 2]
        return self.grayscale
    
    def get_boundary(self, fn):
        mask = np.array(Image.open(self.get_full_path_name(fn)))[:, :, :-1]
        mask = ~(mask == np.ones(mask.shape)*255).all(axis=2) # True for black and False for white
        height, width = mask.shape
        assert self.height == height and self.width == width
        # Find the left, right, up and down boundary of the mask
        minX, minY, maxX, maxY = width, height, 0, 0
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    minX, minY = min(minX, col), min(minY, row)
                    maxX, maxY = max(maxX, col), max(maxY, row)
        return minX, minY, maxX, maxY
    
    def get_sphere(self, fn, matte=False):
        minX, minY, maxX, maxY = self.get_boundary(fn)
        # Coordinates of the ball center (camera-centered coord sys)
        center = self.img_coord_shift_to_camera([(maxX+minX)/2, (maxY+minY)/2, 0])
        radius = (maxX-minX+maxY-minY)/4
        if matte:
            return center, radius
        # Finally get the position of highlight of the sphere
        highlight_pos = [0, 0, 0]
        highlight_val = 0
        for x in range(minX, maxX+1):
            for y in range(minY, maxY+1):
                if self.grayscale[y, x] > highlight_val:
                    highlight_val = self.grayscale[y, x]
                    highlight_pos = [x, y, 0]
        return center, radius, self.img_coord_shift_to_camera(highlight_pos)

    def set_metal_spheres(self, c, r, h):
        self.c1, self.c2 = c
        self.r1, self.r2 = r
        self.highlight1, self.highlight2 = h
    
    def set_matte_spheres(self, cm, rm):
        self.cm = cm
        self.rm = rm
    
    def get_light_dir(self):
        if self.L is not None:
            return self.L
        assert self.r1 and self.r2
        N1 = self.get_normal_on_sphere(self.c1, self.r1, self.highlight1)
        N2 = self.get_normal_on_sphere(self.c2, self.r2, self.highlight2)
        self.L = normalize((N1+N2) / 2)
        return self.L
    
    def get_normal_on_sphere(self, c, r, h):
        x = h[0] - c[0]
        y = h[1] - c[1]
        z = -math.sqrt(r**2 - x**2 - y**2)
        return normalize(np.array([x, y, z]))
    
    def get_matte_highlight_intensity(self):
        loc = self.camera_coord_shift_to_img(self.rm * self.L + self.cm)
        print("Matte highlight location: {}".format(loc))
        return self.image[loc[1], loc[0], :]

if __name__=="__main__":
    display_image = False
    display_normal = False
    display_rerender = False
    ########################################################
    ########## Part I ######################################
    ########################################################
    start_time = time.time()
    folder = "Elephant"
    solutionList = []
    step = 5
    for i in range(1, 22, step):
        fn = "{0}/image{1:03d}.pbm".format(folder, i)
        solution = Solution()
        solution.read_pfm(fn)
        print("{} loaded!".format(fn))
        c1, r1, h1 = solution.get_sphere("{}/mask_dir_1.png".format(folder))
        print("Metal sphere1: centered at {}, radius={}".format(c1, r1))
        c2, r2, h2 = solution.get_sphere("{}/mask_dir_2.png".format(folder))
        print("Metal sphere2: centered at {}, radius={}".format(c2, r2))
        solution.set_metal_spheres([c1, c2], [r1, r2], [h1, h2])
        light_dir = solution.get_light_dir()
        print("Light vector:", light_dir)
        cm, rm = solution.get_sphere("{}/mask_I.png".format(folder), matte=True)
        print("Matte sphere: centered at {}, radius={}".format(cm, rm))
        solution.set_matte_spheres(cm, rm)
        intensity = solution.get_matte_highlight_intensity()
        print("Matte intensity: {}".format(intensity))
        if display_image:
            fig = plt.gcf()
            fig.canvas.set_window_title(fn)
            plt.imshow(solution.grayscale*40, cmap='gray', vmin=0, vmax=1)
            plt.show()
        print("="*80)
        solutionList.append(solution)
    print("Part I total time: {:.3f}s".format(time.time() - start_time))

    #########################################################
    ########## Part II ######################################
    #########################################################
    start_time = time.time()
    n = len(solutionList)
    fn = "{}/{}mask.png".format(folder, "" if folder == "Elephant" else folder.lower())
    mask = np.array(Image.open(solutionList[0].get_full_path_name(fn)))[:, :, :-1]
    mask = ~(mask == np.ones(mask.shape)*255).all(axis=2) # True for black and False for white
    # boundary = [minX, minY, maxX, maxY]
    obj_boundary = solutionList[0].get_boundary(fn)
    obj_width = obj_boundary[2] - obj_boundary[0] + 1
    obj_height = obj_boundary[3] - obj_boundary[1] + 1
    normal_map = np.zeros((obj_height, obj_width, 3))
    Kd_map = np.zeros((obj_height, obj_width))
    # Iterate through all pixels on obj surface
    for y in range(obj_boundary[1], obj_boundary[3]+1):
        for x in range(obj_boundary[0], obj_boundary[2]+1):
            if mask[y, x]:
                continue
            # Stack [I1*L1, I2*L2, ..., In*Ln]
            IL = np.zeros((3, n))
            Isqr = np.zeros((n,))
            for (i, solution) in enumerate(solutionList):
                # I = solution.image[y, x, 2]
                I = solution.grayscale[y, x]
                L = solution.L
                IL[:, i] = I * L
                Isqr[i] = I * I
            A = np.matmul(IL, IL.T)
            Imat = np.matmul(IL, Isqr)
            G = np.matmul(np.linalg.inv(A), Imat)
            Kd = np.linalg.norm(G)
            N = G / Kd
            # Update normal map and albedo map
            normal_map[y-obj_boundary[1], x-obj_boundary[0], :] = N
            Kd_map[y-obj_boundary[1], x-obj_boundary[0]] = Kd
    print("Part II total time: {:.3f}s".format(time.time() - start_time))
    if display_normal:
        fig = plt.gcf()
        fig.canvas.set_window_title("Normal map")
        plt.imshow((normal_map+1)/2)
        plt.show()
        fig = plt.gcf()
        fig.canvas.set_window_title("Albedo")
        plt.imshow(Kd_map, cmap='gray')
        plt.show()
    V = np.array([0.0, 0.0, -1.0])
    rerender = np.multiply(Kd_map, np.sum(np.multiply(normal_map, V), axis=2))
    if display_rerender:
        fig = plt.gcf()
        fig.canvas.set_window_title("Rerendered")
        plt.imshow(rerender, cmap="gray", vmin=0, vmax=np.max(rerender))
        plt.show()
    
    ##########################################################
    ########## Part III ######################################
    ##########################################################
    start_time = time.time()
    mask_cropped = mask[obj_boundary[1]:obj_boundary[3]+1, obj_boundary[0]:obj_boundary[2]+1]
    num_pixels = obj_height * obj_width
    A = scipy.sparse.csr_matrix((num_pixels*2+1, num_pixels))
    B = np.zeros((num_pixels*2+1,))
    for x in range(obj_width-1):
        for y in range(obj_height-1):
            idx = y*obj_width+x
            # if it is background
            if mask_cropped[y, x]:
                A[2*idx, idx] = A[2*idx+1, idx] = 1
                continue
            # if (x+1, y) is not background
            if not mask_cropped[y, x+1]:
                A[2*idx, idx] = normal_map[y, x, 2]
                A[2*idx, idx+1] = -normal_map[y, x, 2]
                B[2*idx] = normal_map[y, x, 0]
            # if (x, y+1) is not background
            if not mask_cropped[y+1, x]:
                A[2*idx+1, idx] = normal_map[y, x, 2]
                A[2*idx+1, idx+obj_width] = -normal_map[y, x, 2]
                B[2*idx+1] = normal_map[y, x, 1]
        y = obj_height-1
        idx = y*obj_width+x
        # if it is background
        if mask_cropped[y, x]:
            A[2*idx, idx] = A[2*idx+1, idx] = 1
            continue
        # if (x+1, y) is not background
        if not mask_cropped[y, x+1]:
            A[2*idx, idx] = normal_map[y, x, 2]
            A[2*idx, idx+1] = -normal_map[y, x, 2]
            B[2*idx] = normal_map[y, x, 0]
        # if (x, y-1) is not background
        if not mask_cropped[y-1, x]:
            A[2*idx+1, idx] = normal_map[y, x, 2]
            A[2*idx+1, idx-obj_width] = -normal_map[y, x, 2]
            B[2*idx+1] = -normal_map[y, x, 1]
    
    x = obj_width-1
    for y in range(obj_height-1):
        idx = y*obj_width+x
        # if it is background
        if mask_cropped[y, x]:
            A[2*idx, idx] = A[2*idx+1, idx] = 1
            continue
        # if (x-1, y) is not background
        if not mask_cropped[y, x-1]:
            A[2*idx, idx] = normal_map[y, x, 2]
            A[2*idx, idx-1] = -normal_map[y, x, 2]
            B[2*idx] = -normal_map[y, x, 0]
        # if (x, y+1) is not background
        if not mask_cropped[y+1, x]:
            A[2*idx+1, idx] = normal_map[y, x, 2]
            A[2*idx+1, idx+obj_width] = -normal_map[y, x, 2]
            B[2*idx+1] = normal_map[y, x, 1]
    
    y = obj_height-1
    idx = y*obj_width+x
    # if it is background
    if mask_cropped[y, x]:
        A[2*idx, idx] = A[2*idx+1, idx] = 1
    else:
        # if (x-1, y) is not background
        if not mask_cropped[y, x-1]:
            A[2*idx, idx] = normal_map[y, x, 2]
            A[2*idx, idx-1] = -normal_map[y, x, 2]
            B[2*idx] = -normal_map[y, x, 0]
        # if (x, y-1) is not background
        if not mask_cropped[y-1, x]:
            A[2*idx+1, idx] = normal_map[y, x, 2]
            A[2*idx+1, idx-obj_width] = -normal_map[y, x, 2]
            B[2*idx+1] = -normal_map[y, x, 1]

    # Boundary condition
    A[-1, num_pixels//2] = 1
    B[-1] = 1
    print("Part III: Matrix built! Start solving...")
    AT = A.transpose()
    Z = scipy.sparse.linalg.spsolve(AT.dot(A), AT.dot(B))
    Z = np.reshape(Z, (-1, obj_width))
    print("Part III total time: {:.3f}s".format(time.time() - start_time))
    fig = plt.gcf()
    fig.canvas.set_window_title("Surface recovery")
    plt.imshow(Z, cmap='gray')
    plt.show()
    