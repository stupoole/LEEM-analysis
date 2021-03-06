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

from Correctors import DriftCorrector, StackPlotter, MediPixCorrector, padding_solver


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

    def __init__(self, root, norm_path, dir_paths, plotting=False, xmcd=False, PCO=False):
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
        :param root:
        :param norm_path: Normalisation image path
        :param dir_paths: Folders to load images from
        :param plotting: Do you want to plot intermediate previews for each step?
        :param xmcd: Calculate XMCD? if false, calculates XMLD
        """
        self.closed = False
        self.original = None
        self.norm = None
        self.corrector = MediPixCorrector()
        self.driftCorrector = DriftCorrector(0, -1, 1, 1, 250, 3, 0.5)
        self.load_norm(norm_path)
        self.plotter = StackPlotter(root)
        self.dichroism_images = []
        self.intensity_images = []
        self.xmcd = xmcd
        self.PCO = PCO

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
                dichroism_image, intensity_image = self.apply_xmcd()
            else:
                dichroism_image, intensity_image = self.apply_xmld()
            if plotting:
                self.plot_single(dichroism_image)
            self.save_stack(self.results, folder)
            self.dichroism_images.append(dichroism_image)
            self.intensity_images.append(intensity_image)
        self.dichroism_images = padding_solver(self.dichroism_images)
        self.save_dichroisms(self.dichroism_images, dir_paths)
        self.save_intensities(self.intensity_images, dir_paths)
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
        # Saves the stack of dichroism results. If you load 20 images, they are all saved in a folder
        # /../XMLD/<dir_paths[0]>/<dir_paths[i]>_xmld.tif"
        if self.xmcd:
            mode = 'XMCD'
        else:
            mode = 'XMLD'

        if self.PCO == True:
            searchString = '_PCOImage'
        else:
            searchString = '_medipixImage'

        root_dir = Path(dir_paths[0])
        root_dir = str(root_dir.parents[1].joinpath(mode).joinpath(root_dir.parts[-1])).replace(searchString,
                                                                                                '_batch')
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        for filename, image in zip(dir_paths, stack):
            filename = Path(filename).parts[-1]
            target = os.path.join(root_dir, str(filename).replace(searchString, '_' + mode + '.tif'))
            image = image.astype('float32')
            sk_imsave(target, image)
            print(f'Image saved as {target}')

    def save_intensities(self, stack, dir_paths):
        # Saves a single dichroism result. If you load 20 images, they are all saved in a folder
        # /../XMLD/<dir_paths[0]>/<dir_paths[i]>_xmld.tif"
        if self.PCO == True:
            searchString = '_PCOImage'
        else:
            searchString = '_medipixImage'
        mode = 'Intensity'
        target_dir = Path(dir_paths[0])
        target_dir = str(target_dir.parents[1].joinpath(mode).joinpath(target_dir.parts[-1])).replace('_medipixImage',
                                                                                                      '_batch')
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        for filename, image in zip(dir_paths, stack):
            filename = Path(filename).parts[-1]
            target = os.path.join(target_dir, str(filename).replace('_medipixImage', '_Intensity.tif'))
            image = image.astype('float32')
            sk_imsave(target, image)
            print(f'Image saved as {target}')

    def save_stack(self, stack, dirname):
        # Saves a stack of images. Currently used to save the stack of aligned images.
        path = Path(dirname)
        path = path.parents[1].joinpath('RAW_aligned').joinpath(path.parts[-1])
        if not os.path.isdir(path):
            os.makedirs(path)
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
        while np.sum(first_half == 0) > 0:
            print(np.sum(first_half == 0))
            meaned = convolve2d(first_half, meaner, mode='same')
            first_half[first_half == 0] = meaned[first_half == 0]

        while np.sum(second_half == 0) > 0:
            print(np.sum(second_half == 0))
            meaned = convolve2d(second_half, meaner, mode='same')
            second_half[second_half == 0] = meaned[second_half == 0]

        xmld = (first_half - second_half) / (first_half + second_half)
        intensity = (first_half + second_half)
        return xmld, intensity

    def apply_xmcd(self):
        threshold = 0.1
        n, x, y = self.results.shape
        temp_stack = np.abs(self.results) + 0.01 * (self.results == 0)  # remove negatives
        first_half = (np.mean(temp_stack[0:n // 4, :, :], axis=0) +
                      np.mean(temp_stack[n // 4:n // 2, :, :], axis=0)) / 2
        second_half = (np.mean(temp_stack[n // 2:(3 * n) // 4, :, :], axis=0) +
                       np.mean(temp_stack[3 * n // 4:, :, :], axis=0)) / 2
        intensity = (first_half + second_half)

        threshold_mask = intensity > threshold  # try to ignore regions outside sample edges

        xmcd = (first_half - second_half) / (first_half + second_half + 10000 * threshold_mask)

        return xmcd, intensity

    def plot_single(self, image):
        # Just plot an image.
        plt.figure('XMLD Image Result')
        plt.imshow(image)
        plt.pause(0.5)


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
    ex.setDirectory(str(Path(norm_path).parent))
    ex.show()
    app.exec_()
    directories = [os.path.abspath(path) for path in ex.selectedFiles() if
                   (path.find('medipixImage') != -1 or path.find('PCOImage') != -1)]

    app.quit()
    # Specify options. By default XMCD and Plotting are both false
    plotting = False
    XMCD = False
    PCO = False
    if directories[0].find('PCOImage') != -1:
        PCO = True
    for arg in sys.argv:
        if arg == '-xmcd' or arg == '-XMCD':
            XMCD = True
        if arg == '-PCO' or arg == '-pco':
            PCO = True
    processor = RapidXMLD(root, norm_path, directories, plotting, XMCD, PCO)
