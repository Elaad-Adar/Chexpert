import random

import PIL.Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.transform import resize
import torchvision.transforms as T

from Wavelets.matmMP import matmMP
from Wavelets.WP_an_2dMP import WP_an_2dMP
import Wavelets.wpt_utils as wpt_utils


def sort_by_energy(data):
    s = np.empty(shape=(1, data.shape[0]), dtype=np.int32)
    for c in range(data.shape[0]):
        s[0, c] = np.sum(np.power(np.abs(data[c, :, :]), 2))

    return data[np.flip(np.argsort(s))][0]


class qWPT(object):

    def __init__(self, DeTr=4, dc=2, Par=5, N=256, printout=False,
                 vis_channels=False, energy_sort=False, nfreq=None, norm=False, expand=None, use_originals=False):
        self.DeTr = DeTr
        self.dc = dc
        self.Par = Par
        self.N = N
        self.printout = printout
        self.energy_sort = energy_sort
        self.nfreq = nfreq
        self.norm = norm
        self.expand = expand
        self.mat_p, self.mat_m = self.compute_wavelet_transform()
        self.diagonal = wpt_utils.get_diagonal(level=self.DeTr)
        self.vis = vis_channels
        self.use_originals = use_originals

    def __call__(self, image):
        #check dimension
        if image.ndim == 3:
            image = image[0, :, :]
            if image.shape != (self.N, self.N):
                image = resize(image, (self.N, self.N), 3)
        qwp_mat, _ = self.transform_image(image)
        # sort the matrix into stacked order
        data = self.stack_sub_bands(qwp_mat)
        if self.energy_sort:
            data = sort_by_energy(data)
        if self.nfreq is not None:
            data = data[:self.nfreq, :, :]
        if self.use_originals:
            image = np.expand_dims(image, axis=0)
            data = np.vstack([image, data])
            return data
        return data

    def compute_wavelet_transform(self):
        # TODO add description here
        matrm_p = matmMP(self.N, self.Par, self.DeTr, 3, self.dc)
        matrm_m = matmMP(self.N, self.Par, self.DeTr, 4, self.dc)
        if self.printout:
            print("done calculating transformations array")

        return matrm_p, matrm_m

    def transform_image(self, image):
        # TODO add description here
        ptran, mtran = WP_an_2dMP(image, self.DeTr, self.mat_p, self.mat_p, self.mat_m)

        if self.printout:
            print("done transforming image")
        if self.vis:
            self.show_channels(ptran)

        return ptran, mtran

    def stack_sub_bands(self, mat):
        # TODO add description here
        r, c, num_of_graph = mat.shape
        graph_row = 2 ** self.DeTr
        div = self.DeTr - 1
        split = r // graph_row
        numOfChannels = len(self.diagonal)
        new_tensor = np.zeros((numOfChannels, split, split), dtype='float')
        for idx, (i, j) in enumerate(self.diagonal):
            if self.norm:
                new_tensor[idx, :, :] = wpt_utils.NormalizeData(
                    np.real(mat[i * split:i * split + split, j * split:j * split + split, div]))
            else:
                new_tensor[idx, :, :] = np.real(mat[i * split:i * split + split, j * split:j * split + split, div])
        image_array = new_tensor[:, :, :]

        if self.expand is not None:
            new_tensor2 = resize(image_array, (numOfChannels, self.expand, self.expand), order=3)
            return (new_tensor2).astype(np.uint8)
        return (image_array).astype(np.uint8)

    def show_image(self, image, title=None):
        # x = wpt_utils.resize_image(image, self.N, self.N)
        if len(image.shape) > 2:
            image = image[0, :, :]
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()

    def show_channels(self, tran):
        r, c, num_of_graph = tran.shape
        graph_row = 2 ** self.DeTr
        div = self.DeTr - 1
        split = r // graph_row
        diag = wpt_utils.get_diagonal(level=self.DeTr)
        fig, axs = plt.subplots(graph_row, graph_row)
        for idx, (i, j) in enumerate(diag):
            axs[i, j].axis("off")
            axs[i, j].imshow(np.real(tran[i * split:i * split + split, j * split:j * split + split, div]),
                             cmap="gray",
                             aspect="auto")
            plt.sca(axs[i, j])
        fig.tight_layout()
        fig.suptitle(f'sub matrices for level {self.DeTr}', fontsize=16)
        plt.show()


if __name__ == "__main__":
    ti = qWPT(DeTr=3, vis_channels=True)
    image2add = PIL.Image.open("E:\Thesis\Images\HD samples\Atelectasis_frontal.jpg")
    print(f'W={image2add.width}, H={image2add.height}')
    plt.imshow(image2add, cmap="gray")
    plt.title("original image")
    plt.show()
    transformations = [T.Resize(1024),
        T.CenterCrop(1024),
                       lambda x: torch.from_numpy(np.array(x, copy=True)).float().div(255).unsqueeze(0),
                       T.Normalize(mean=[0.5330], std=[0.0349]), qWPT(DeTr=3, N=1024, expand=80, vis_channels=True)
                       ]
    transforms = T.Compose(transformations)
    x = np.asarray(transforms(image2add))
    print(x.shape)
    ti.show_image(image=x, title="image after transformation")
    for i in range(5):
        id = random.randrange(0, 64)
        plt.imshow(x[id, :, :], cmap="gray")
        plt.title(f"sub band {id} after transformations")
        plt.show()
    # ti.transform_image(image=image2add, vis_channels=True)
    # ti(image2add)
