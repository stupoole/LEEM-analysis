import os
import numpy as np
import dask.array.image as daim
from scipy.signal import convolve2d
from tkinter import filedialog
from skimage.io import imsave as sk_imsave


class NormalisationImageProcessor:

    def __init__(self):
        self.stack = None
        self.norm_image = None
        self.is_norm = True
        self.bad_pixel_image = daim.imread('badPixelImage*.tif').compute()[0]

    def set_stack(self, stack):
        self.original = stack
        self.num, self.width, self.height = stack.shape
        self.dE = 4

    def apply_corrections(self):
        if self.original is not None:
            self.__fix_overlap()
        self.__fix_bad_pixels()
        return self.__make_norm()

    def __fix_overlap(self):
        self.stack = np.zeros((self.num, self.width + 2, self.height + 2))
        temp = self.original.copy()
        temp[:, self.width // 2 - 1:self.width // 2 + 1, :] = temp[:, self.width // 2 - 1:self.width // 2 + 1, :] / 2
        temp[:, :, self.height // 2 - 1:self.height // 2 + 1] = temp[:, :,
                                                                self.height // 2 - 1:self.height // 2 + 1] / 2
        self.stack[:, :self.width // 2, :self.height // 2] = temp[:, :self.width // 2, :self.height // 2]
        self.stack[:, self.width // 2 + 2:, self.height // 2 + 2:] = temp[:, self.width // 2:, self.height // 2:]
        self.stack[:, :self.width // 2, self.height // 2 + 2:] = temp[:, :self.width // 2, self.height // 2:]
        self.stack[:, self.width // 2 + 2:, :self.height // 2] = temp[:, self.width // 2:, :self.height // 2]

        # left vert top
        self.stack[:, :self.width // 2, self.height // 2] = temp[:, :self.width // 2, self.height // 2 - 1]
        # right vert top
        self.stack[:, :self.width // 2, self.height // 2 + 1] = temp[:, :self.width // 2, self.height // 2]
        # left horiz bot
        self.stack[:, self.width // 2 + 1, :self.height // 2] = temp[:, self.width // 2, :self.height // 2]
        # left horiz top
        self.stack[:, self.width // 2, :self.height // 2] = temp[:, self.width // 2 - 1, :self.height // 2]
        # left vert bot
        self.stack[:, self.width // 2 + 2:, self.height // 2] = temp[:, self.width // 2:, self.height // 2 - 1]
        # right vert bot
        self.stack[:, self.width // 2 + 2:, self.height // 2 + 1] = temp[:, self.width // 2:, self.height // 2]
        # right horiz top
        self.stack[:, self.width // 2, self.height // 2 + 2:] = temp[:, self.width // 2 - 1, self.height // 2:]
        # right horiz bot
        self.stack[:, self.width // 2 + 1, self.height // 2 + 2:] = temp[:, self.width // 2, self.width // 2:]

        # central squares
        self.stack[:, self.width // 2, self.height // 2] = temp[:, self.width // 2 - 1, self.height // 2 - 1]
        self.stack[:, self.width // 2 + 1, self.height // 2] = temp[:, self.width // 2, self.height // 2 - 1]
        self.stack[:, self.width // 2 + 1, self.height // 2 + 1] = temp[:, self.width // 2, self.height // 2]
        self.stack[:, self.width // 2, self.height // 2 + 1] = temp[:, self.width // 2 - 1, self.height // 2]

        # def __apply_normalisation(self):

    def __fix_bad_pixels(self):
        for i in range(self.stack.shape[0]):
            image = self.stack[i]
            meaner = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
            meaned = convolve2d(image, meaner, mode='same')
            image[self.bad_pixel_image == 1] = meaned[self.bad_pixel_image == 1]
            image[image > 3000] = meaned[image > 3000]
            image[image < 0.1] = meaned[image < 0.1]
            self.stack[i] = image

    def __make_norm(self):
        image = np.mean(self.stack, axis=0)
        return image / np.mean(image)


if __name__ == '__main__':
    corrector = NormalisationImageProcessor()
    load_dir = filedialog.askdirectory(title='Select a folder containing Normalisation image file(s)')
    stack = daim.imread(os.path.join(load_dir, '*.tif')).compute()
    corrector.set_stack(stack)

    normImage = corrector.apply_corrections()
    save_name = filedialog.asksaveasfilename(title='Specify save name for norm image in a new folder',
                                             defaultextension='.tif')
    normImage = normImage.astype('float32')
    if save_name:
        print(f'Norm image saved as {save_name}')
        sk_imsave(save_name, normImage)
