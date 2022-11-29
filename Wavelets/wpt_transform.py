import PIL.Image
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

import Wavelets.wpt_utils as wpt_utils


def sort_by_energy(data):
    s = np.empty(shape=(1, data.shape[0]), dtype=np.int32)
    for c in range(data.shape[0]):
        s[0, c] = np.sum(np.power(np.abs(data[c, :, :]), 2))

    return data[np.flip(np.argsort(s))][0]


class qWPT(object):

    def __init__(self, DeTr=4, dc=2, Par=5, N=256, printout=False, energy_sort=False, nfreq=None, norm=False, expand=None):
        self.DeTr = DeTr
        self.dc = dc
        self.Par = Par
        self.N = N
        self.printout = printout
        self.energy_sort = energy_sort
        self.nfreq = nfreq
        self.norm = norm
        self.expand = expand
        mat_p, self.mat_m = self.compute_wavelet_transform()
        self.diagonal = wpt_utils.get_diagonal(level=self.DeTr)

    def __call__(self, image):
        qwp_mat, _ = self.transform_image(image)
        # sort the matrix into stacked order
        data = self.stack_sub_bands(qwp_mat, norm=self.norm, expand=self.expand)
        if self.energy_sort:
            data = sort_by_energy(data)
        if self.nfreq is not None:
            data = data[:self.nfreq, :, :]

        return data

    def compute_wavelet_transform(self):
        # TODO add description here
        from matmMP import matmMP

        matrm_p = matmMP(self.N, self.Par, self.DeTr, 3, self.dc)
        matrm_m = matmMP(self.N, self.Par, self.DeTr, 4, self.dc)
        if self.printout:
            print("done calculating transformations array")

        return matrm_p, matrm_m

    def transform_image(self, image, vis_channels=False):
        # TODO add description here
        from WP_an_2dMP import WP_an_2dMP
        image = wpt_utils.resize_image(image, self.N, self.N)
        ptran, mtran = WP_an_2dMP(image, self.DeTr, self.mat_p, self.mat_p, self.mat_m)

        if self.printout:
            print("done transforming image")
        if vis_channels:
            self.show_channels(ptran)

        return ptran, mtran

    def stack_sub_bands(self, mat):
        # TODO add description here
        r, c, num_of_graph = mat.shape
        graph_row = 2 ** self.DeTr
        div = self.DeTr - 1
        split = r // graph_row
        # diag = wpt_utils.get_diagonal(level=self.DeTr)
        numOfChannels = len(self.diagonal)
        new_tensor = np.zeros((numOfChannels, split, split), dtype='float')
        for idx, (i, j) in enumerate(self.diagonal):
            if self.norm:
                new_tensor[idx, :, :] = wpt_utils.NormalizeData(
                    np.real(mat[i * split:i * split + split, j * split:j * split + split, div]))
            else:
                new_tensor[idx, :, :] = np.real(mat[i * split:i * split + split, j * split:j * split + split, div])
        image_array = (new_tensor[:, :, :]).astype(np.uint8)

        if self.expand != None:
            new_tensor2 = np.zeros((numOfChannels, self.expand, self.expand), dtype='float')
            for c in range(numOfChannels):
                new_tensor2[c, :, :] = resize(image_array[c, :, :], (self.expand, self.expand), order=3)
            return new_tensor2
        return image_array

    def show_image(self, image):
        x = wpt_utils.resize_image(image, self.N, self.N)
        plt.imshow(x, cmap="gray")
        plt.title('original image')
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
    ti = qWPT(DeTr=3, N=256)
    image2add = PIL.Image.open("view1_frontal.jpg")
    # ti.show_image(image=image2add)
    # ti.transform_image(image=image2add, vis_channels=True)
    ti(image2add, energy_sort=True, nfreq=10)
