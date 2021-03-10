import numpy as np
import os
import numba
import time
import sys
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

from pathlib import Path

from PyQt5.QtWidgets import QFileDialog, QApplication, QListView, QTreeView, QAbstractItemView

from Correctors import DriftCorrector, StackPlotter, MediPixCorrector


class MultiDirectoryFileDialog(QFileDialog):
    def __init__(self, *args):
        QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        self.tree = self.findChild(QTreeView)
        self.tree.setSelectionMode(QAbstractItemView.MultiSelection)

        self.list = self.findChild(QListView)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)


class RapidXMLD:

    def __init__(self, root,  norm_path, dir_paths, plotting=False, xmcd=False):
        """
        This class applies multiple corrections to XMLD/XMCD images taken at i10 at diamond. It loads a normalisation
        image to use with all images that are processed and has a built in bad pixel image which much be changed
        manually. The dir_paths variable is a list of folders of raw images from the detector. These folders will be
        loaded and processed sedquentially so the images in the first folder are loaded, corrected and aligned before
        dichroism calculations are made.
        The aligned images are saved in a folder "RAW_Aligned" one directory up from the raw data folder of folders.
        The XMLD images are saved to make it convenient to load a batch of results. To do this, the name of the first
        folder of images is used as the folder of results (call it folder_1). The resulting XMLD/XMCD images from
        folder_n are saved up one folder from RAW and in XMLD/folder_1/folder_n.tif
        :param norm_path: Normalisation image path
        :param dir_paths: Folders to load images from
        :param plotting: Do you want
        :param xmcd:
        """
        self.closed = False
        self.corrector = MediPixCorrector()
        self.driftCorrector = DriftCorrector(0, -1, 1, 1, 250, 3, 0.5)
        self.load_norm(norm_path)
        self.plotter = StackPlotter(root)
        self.dichroism_images = []
        self.xmcd = xmcd

        for folder in dir_paths:
            self.load_images(folder)
            if plotting:
                self.plotter.plot_stack(self.original, 'Original ' + folder.split(os.sep)[-1])
            self.corrector.set_stack(self.original, self.norm)
            self.results = self.corrector.apply_corrections()
            if plotting:
                self.plotter.plot_stack(self.results, 'Corrected ' + folder.split(os.sep)[-1])
            self.results = self.driftCorrector.apply_corrections(self.results)
            if plotting:
                self.plotter.plot_stack(self.results, 'Aligned ' + folder.split(os.sep)[-1])

            if xmcd:
                dichroism_image = self.apply_xmcd()
            else:
                dichroism_image = self.apply_xmld()
            if plotting:
                self.plot_single(dichroism_image)
            self.save_stack(self.results, folder)
            self.dichroism_images.append(dichroism_image)
        self.padding_solver()
        self.save_dichroisms(self.dichroism_images, dir_paths)
        self.plotter.plot_stack(np.array(self.dichroism_images), "Dichroism Images", "Save Results")

    def load_norm(self, filepath):
        # Just loads the normalisation image
        if not filepath == "":
            self.norm = daim.imread(filepath)
        else:
            self.closed = True

    def load_images(self, load_directory):
        # Load all images in a given directory
        print('loading from: ', load_directory)
        self.original = daim.imread(os.path.join(load_directory, '*.tif')).compute()

    def save_dichroisms(self, stack, dir_paths):
        # Saves a single dichroism result. If you load 20 images, they are all saved in a folder
        # /../XMLD/<dir_paths[0]>/<dir_paths[i]>_xmld.tif"
        if self.xmcd:
            mode = 'XMCD'
        else:
            mode = 'XMLD'

        root = Path(dir_paths[0])
        root = str(root.parents[1].joinpath(mode).joinpath(root.parts[-1])).replace('_medipixImage', '_batch')
        if not os.path.isdir(root):
            os.mkdir(root)

        for filename, image in zip(dir_paths, stack):
            filename = Path(filename).parts[-1]
            target = os.path.join(root, str(filename).replace('_medipixImage', '_XMLD.tif'))
            image = image.astype('float32')
            sk_imsave(target, image)
            print(f'Image saved as {target}')

    def save_stack(self, stack, dirname):
        # Saves a stack of images. Currently used to save the stack of aligned images.
        path = Path(dirname)
        path = path.parents[1].joinpath('RAW_aligned').joinpath(path.parts[-1])
        if not os.path.isdir(path):
            os.mkdir(path)
        file_list = os.listdir(dirname)
        file_list = [file for file in file_list if file[-4] == "."]
        file_list = [file.replace('.tif', '_corrected.tif') for file in file_list]
        for filename, image in zip(file_list, stack):
            target = os.path.join(path, filename)
            sk_imsave(target, image)
            print(f'Image saved as {target}')

    def apply_xmld(self):
        # Calculates xmld as mean(a - b ) / (a + b)
        # Fixes zero values as average of surrounds
        n, x, y = self.results.shape
        first_half = np.mean(self.results[0:n // 2, :, :], axis=0)
        second_half = np.mean(self.results[n // 2:, :, :], axis=0)

        meaner = np.array([[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]])
        meaned = convolve2d(first_half, meaner, mode='same')
        first_half[first_half <= 0.1] = meaned[first_half <= 0.1]

        meaned = convolve2d(second_half, meaner, mode='same')
        second_half[second_half <= 0.1] = meaned[second_half <= 0.1]

        result = (first_half - second_half) / (first_half + second_half)
        return result

    def apply_xmcd(self):
        # Same as for xmld but the images are normalised with an on edge image. mean(a/b - c/d) / (a/b + c/d)
        # Fixes zero values as average of surrounds
        n, x, y = self.results.shape
        first_half = self.results[0:n // 4, :, :] / self.results[n // 4:n // 2, :, :]
        second_half = self.results[n // 2:3 * n // 4, :, :] / self.results[3 * n // 4:, :, :]

        meaner = np.array([[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]])
        meaned = convolve2d(first_half, meaner, mode='same')
        first_half[first_half <= 0.1] = meaned[first_half <= 0.1]

        meaned = convolve2d(second_half, meaner, mode='same')
        second_half[second_half <= 0.1] = meaned[second_half <= 0.1]

        result = np.mean(first_half - second_half, axis=0) / (first_half + second_half)
        plt.figure('XMCD Image Result')
        plt.imshow(result)
        plt.pause(0.5)
        return result

    def plot_single(self, image):
        # Just plot an image.
        plt.figure('XMLD Image Result')
        plt.imshow(image)
        plt.pause(0.5)

    def padding_solver(self):
        # Makes all arrays same size to make stacking easier. The padded images are not the images that are saved.
        new_size = max([array.shape for array in self.dichroism_images])
        for i in range(0, len(self.dichroism_images)):
            array = self.dichroism_images[i]
            pads = np.array(new_size) - np.array(array.shape)
            if pads.sum() > 0:
                array = np.pad(array, (((pads[0] + 1) // 2, pads[0] // 2), ((pads[1] + 1) // 2, pads[1] // 2)),
                               'constant', constant_values=0)
                self.dichroism_images[i] = array


if __name__ == '__main__':
    # Choose a normalisation image to use
    root = tk.Tk()
    norm_path = os.path.abspath(filedialog.askopenfilename(
        filetypes=[('Tiff Image', '.tif'), ('All Files)', '*')],
        title='Select the Normalisation image file processed with NormalisationImageProcessor'))

    # Selects folders (Have to do this outside the app because of QApplication technicalities (it errors when
    # created inside an object)
    app = QApplication(sys.argv)
    ex = MultiDirectoryFileDialog()
    ex.show()
    app.exec_()
    directories = [os.path.abspath(path) for path in ex.selectedFiles()[1:]]
    app.quit()
    # Specify options. By default XMCD and Plotting are both false
    plotting = True
    XMCD = False
    processor = RapidXMLD(root, norm_path, directories, plotting, XMCD)
