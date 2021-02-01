import numpy as np
import os
import numba
import time

import dask
import dask.array as da
import dask.array.image as daim
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster

from scipy.optimize import least_squares
import scipy.ndimage as ndi
import scipy.sparse as sp
from scipy.interpolate import interp1d
from scipy.signal import convolve2d

from skimage import filters
from skimage.io import imsave as sk_imsave
import tkinter as tk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure


class ScrollBarImagePlot(object):
    def __init__(self, ax, X):
        self.ax = ax

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0

        self.im = self.ax.imshow(X[self.ind, :, :].T, cmap='gray', vmin=self.X.min(), vmax=self.X.max())
        self.update()

    def onscroll(self, new_val):
        self.ind = int(new_val)
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :].T)
        self.ax.set_ylabel('slice %s' % self.ind)
        # self.ax.figure.canvas.draw()

    def replace(self, X):
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0
        self.im = self.ax.imshow(X[self.ind, :, :].T, cmap='gray', vmax=self.X.max())
        self.update()


class medipix_corrector:

    def __init__(self):
        self.stack = None
        self.norm_image = None
        self.is_norm = False
        self.bad_pixel_image = daim.imread('badPixelImage*.tif').compute()[0]

    def set_stack(self, stack, norm_image=None, is_norm=False):
        self.original = stack
        self.num, self.width, self.height = stack.shape
        self.norm_image = norm_image
        self.dE = 4
        self.is_norm = is_norm

    def apply_corrections(self):
        if self.original is not None:
            self.__fix_overlap()

        self.__fix_bad_pixels()
        if self.is_norm == True:
            return self.__make_norm()
        if self.norm_image is not None:
            self.__apply_normalisation()
        return self.stack

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
            image[image > 4000] = meaned[image > 4000]
            self.stack[i] = image

    def __make_norm(self):
        image = np.mean(self.stack, axis=0)
        return image / np.mean(image)

    def __apply_normalisation(self):
        self.stack = self.stack / np.repeat(self.norm_image, self.stack.shape[0], axis=0)


class DichroismProcessor:

    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Original Images")
        self.closed = False
        self.corrector = medipix_corrector()
        self.load_norm()
        self.load_images()
        self.corrector.set_stack(self.original, is_norm=False)
        results = self.corrector.apply_corrections()
        plt.imshow(results[0])
        plt.show()
        self.save_stack(results)

    #     todo(stupoole) do the dichroism calculations

    def load_norm(self):
        self.load_directory = filedialog.askdirectory(title='Select a folder containing Normalisation image file(s)')

        if not self.load_directory == "":
            self.norm = daim.imread(os.path.join(self.load_directory, '*.tif')).compute()
            plt.imshow(self.norm[0, :, :])
            plt.show()
        else:
            self.closed = True

        if self.norm.shape[0] != 1:
            self.corrector.set_stack(self.norm, is_norm=True)
            self.norm = self.corrector.apply_corrections()
            self.save_single(self.norm)

    def load_images(self):
        self.load_directory = filedialog.askdirectory(title='Select a folder containing image files')
        print('loading from: ', self.load_directory)
        if not self.load_directory == "":
            self.original = daim.imread(os.path.join(self.load_directory, '*.tif')).compute()
            plt.imshow(self.original[0, :, :])
            plt.show()
        else:
            self.closed = True

    def save_single(self, image):
        save_name = filedialog.asksaveasfilename(title='Specify save name for norm image in a new folder',
                                                 defaultextension='.tif')
        image = image.astype('float32')
        if save_name:
            print(f'Norm image saved as {save_name}')
            sk_imsave(save_name, image)

    def save_stack(self, stack):
        save_directory = filedialog.askdirectory(title='Select save directory')
        if save_directory:  # if a folder was selected, don't save otherwise
            # TODO(STU) combine these operations into one or 2
            file_list = os.listdir(self.load_directory)
            file_list = [file for file in file_list if file[-4] == "."]
            file_list = [file.replace('.tif', '_corrected.tif') for file in file_list]
            for filename, image in zip(file_list, stack):
                target = os.path.join(save_directory, filename)
                sk_imsave(target, image)
                print(f'Image saved as {target}')
        else:
            print('Data not saved')

        self.root.destroy()


if __name__ == '__main__':
    corrector = DichroismProcessor()
