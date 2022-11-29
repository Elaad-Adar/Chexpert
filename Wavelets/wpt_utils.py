import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import torchvision.transforms as T


def ceildiv(a, b):
    """
    ceil(a/b)
    """
    return -(-a // b)

def sort_diagonal(mat, upper):
    R, C = mat.shape
    diag = []

    def isValid(i, j):
        if (i < 0 or i >= R or j >= C or j < 0):
            return False
        return True

    for k in range(0, R):
        diag.append(mat[k][0])
        # set row index for next point in diagonal
        i = k - 1
        # set column index for next point in diagonal
        j = 1
        #Diagonally upward
        while isValid(i, j):
            diag.append(mat[i][j])
            i -= 1
            j += 1  # move in upright direction
    if not upper:
        for k in range(1, C):
            diag.append(mat[R - 1][k])
            # set row index for next point in diagonal
            i = R - 2
            # set column index for next point in diagonal
            j = k + 1
            #Diagonally upward
            while isValid(i, j):
                diag.append(mat[i][j])
                i -= 1
                j += 1  # move in upright direction
    return diag


def NormalizeData(data):
    return ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype('uint8')


def dlevel_channels(tran, level, diag, norm=False, expand=True, newSize=256):
    r, c, num_of_graph = tran.shape
    graph_row = 2 ** level
    div = level - 1
    split = r // graph_row
    numOfChhannels = len(diag)
    new_tensor = np.zeros((numOfChhannels, split, split), dtype='float')
    for idx, (i, j) in enumerate(diag):
        if norm:
            new_tensor[idx, :, :] = NormalizeData(np.real(tran[i * split:i * split + split, j * split:j * split + split, div]))
        else:
            new_tensor[idx, :, :] = np.real(tran[i * split:i * split + split, j * split:j * split + split, div])
        plt.imshow(new_tensor[idx, :, :], cmap="gray", aspect="auto")
        plt.title(f'sub matrix for index {idx} at ({i}, {j}) level {level}')
        plt.show()



    # np.save('numpy_image', new_tensor)  # .npy extension is added if not given
    if expand:
        new_tensor2 = np.zeros((numOfChhannels, r, c), dtype='float')
        for c in range(numOfChhannels):
            new_tensor2[c, :, :] = resize(image_array[c, :, :], (newSize, newSize), order=3)
            # plt.imshow(new_tensor2[c, :, :], cmap="gray", aspect="auto")
            # plt.title(f'sub matrix for index {c} after resize')
            # plt.show()
        return new_tensor2
    return image_array

def run_function_in_parallel(function, args):
    """
    run function with number of workers as number of cores there are
    """
    from multiprocessing import Pool

    # get number of cores
    num_cores = 2
    args2 = [x for x in args if x.size > 0]
    '''for arg in tqdm(args2):
    yield function(arg)
    return'''


    # run function in parallel
    with Pool(num_cores) as p:
        return p.map(function, tqdm(args2))

def get_diagonal(level, upper=False):
    idx_mat = np.empty((2 ** level, 2 ** level), object)
    for id in np.ndindex(2 ** level, 2 ** level):
        idx_mat[id] = id
    diag_idx = sort_diagonal(idx_mat.T, upper)
    return diag_idx

def resize_image(image, width, height):
    transforms = T.Compose([
        T.Resize((width, height))])
    x = np.asarray(transforms(image))

    return x
