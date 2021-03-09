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

class FileDialog(QFileDialog):
    def __init__(self, *args):
        QFileDialog.__init__(self, *args)
        self.setOption(self.DontUseNativeDialog, True)
        self.setFileMode(self.DirectoryOnly)

        self.tree = self.findChild(QTreeView)
        self.tree.setSelectionMode(QAbstractItemView.MultiSelection)

        self.list = self.findChild(QListView)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)


class RapidXMLD:

    def __init__(self, norm_path, dir_paths, plotting, xmcd=False):
        self.root = tk.Tk()
        self.root.wm_title('Root Window')
        self.closed = False
        self.corrector = MediPixCorrector()
        self.driftCorrector = DriftCorrector(0, -1, 1, 1, 250, 3, 0.5)
        self.load_norm(norm_path)
        if plotting:
            self.plotter = StackPlotter(self.root)


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
            self.save_stack(self.results, folder)
            if xmcd:
                dichroism_image = self.apply_xmcd()
            else:
                dichroism_image = self.apply_xmld()
            if plotting:
                self.plot_single(dichroism_image)
            self.save_single(dichroism_image, folder)


    def load_norm(self, filepath):
        if not filepath == "":
            self.norm = daim.imread(filepath)
            plt.figure('Normalisation Image Preview')
            plt.imshow(self.norm[0, :, :])
            plt.show()
        else:
            self.closed = True

    def load_images(self, load_directory):
        print('loading from: ', load_directory)
        self.original = daim.imread(os.path.join(load_directory, '*.tif')).compute()

    def save_single(self, image, filename):
        path = Path(filename)
        path = path.parents[1].joinpath('XMLD').joinpath(path.parts[-1] + '_xmld.tif')
        image = image.astype('float32')
        if path:
            sk_imsave(path, image)
            print(f'XMLD image saved as {path}')

    def save_stack(self, stack, dirname):
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
        n, x, y = self.results.shape
        first_half = self.results[0:n//2,: ,:]
        second_half = self.results[n//2:, :, :]
        result = np.mean(first_half - second_half, axis=0)/np.sum(self.results, axis=0)
        return result

    def apply_xmcd(self):
        n, x, y = self.results.shape
        first_half = self.results[0:n//4,: ,:] / self.results[n//4:n//2, :, :]
        second_half = self.results[n//2:3*n//4, :, :] / self.results[3*n//4:, :, :]
        result = np.mean(first_half - second_half, axis=0)/(first_half + second_half)
        plt.figure('XMCD Image Result')
        plt.imshow(result)
        plt.pause(0.5)
        return result
    def plot_single(self, image):
        plt.figure('XMLD Image Result')
        plt.imshow(image)
        plt.pause(0.5)


if __name__ == '__main__':
    norm_path = os.path.abspath(filedialog.askopenfilename(
        filetypes=[('Tiff Image', '.tif'), ('All Files)', '*')],
        title='Select the Normalisation image file processed with NormalisationImageProcessor'))

    app = QApplication(sys.argv)
    ex = FileDialog()
    ex.show()
    app.exec_()
    directories = [os.path.abspath(path) for path in ex.selectedFiles()[1:]]
    # print(directories)

    # norm_path = os.path.abspath('D:\Diamond Data Processing Oct 2020\Data\Processed\M_\M_norm_264021.tif')
    # directories = ['D:/Diamond Data Processing Oct 2020/Data/RAW/265040_medipixImage', 'D:/Diamond Data Processing Oct 2020/Data/RAW/265041_medipixImage', 'D:/Diamond Data Processing Oct 2020/Data/RAW/265042_medipixImage']
    # directories = [os.path.abspath(path) for path in directories]
    processor = RapidXMLD(norm_path, directories, False, False)
