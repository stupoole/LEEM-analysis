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

from skimage import filters
from skimage.io import imsave as sk_imsave
from skimage.io import imread as sk_imread

import tkinter as tk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

import Registration
from Correctors import ScrollBarImagePlot, padding_solver


# Does not have access to self to define this dtype out and fails at infer dtype.
@da.as_gufunc(signature="(i,j),(2)->(i,j)", output_dtypes=np.float32, vectorize=True)
def shift_images(image, shift):
    """Shift `image` by `shift` pixels."""
    return ndi.shift(image, shift=shift, order=1)


class DriftCorrector:
    def __init__(self, start, stop, stride, dE, fftsize, sigma=3, threshold=0.15):
        self.start = start
        self.stop = stop
        self.stride = stride
        self.dE = dE
        self.fftsize = fftsize
        self.savefig = True
        self.Eslice = slice(start, stop, stride)
        self.z_factor = 1
        self.root = tk.Tk()
        self.root.wm_title("Original Images")
        self.closed = False
        self.sigma = sigma
        self.threshold = threshold
        # self.root.withdraw()

    def apply_corrections(self):
        self.load_images()

        if self.closed:
            return

        self.calculate_sobel()

        self.calculate_cross_correlations()

        self.weights, self.argmax = Registration.max_and_argmax(self.correlations)

        self.calculate_half_matrices()

        self.normalise_maximum_weights()

        self.calculate_thresholding()

        self.calculate_shift_vectors()

        self.apply_shifts()
        self.corrected_images = self.corrected.compute()
        self.plot_corrected()

        if not self.closed:
            self.save_stack()

    def load_images(self):
        self.load_directory = filedialog.askdirectory(title='Select a folder containing image files')
        file_list = os.listdir(self.load_directory)
        file_list = [file for file in file_list if file[-4:] == ".tif"]
        self.original = list()
        if not self.load_directory == "":
            for i in range(0, len(file_list)):
                print("Loading " + os.path.join(self.load_directory, file_list[i]))
                self.original.append(sk_imread(os.path.join(self.load_directory, file_list[i]), as_gray=True))
            self.original = padding_solver(self.original)
            #
            # self.original = daim.imread(os.path.join(self.load_directory, '*.tif'))
            self.plot_original()
        else:
            self.closed = True

    def plot_original(self):
        self.root.wm_title("Original Images")
        self.root.protocol("WM_DELETE_WINDOW", self.GUI_close_action)

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.subplots(1, 1)
        self.tracker = ScrollBarImagePlot(fig, ax, self.original.compute())
        canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        slider = tk.Scale(self.root, from_=0, to=self.original.shape[0] - 1, orient=tk.HORIZONTAL,
                          command=self.tracker.onscroll)
        slider.pack(fill=tk.X)
        self.cont_button_text = tk.StringVar()
        cont_button = tk.Button(self.root, command=self.GUI_pause_action, textvariable=self.cont_button_text)
        self.cont_button_text.set("Continue")
        cont_button.pack()
        self.isRunning = False
        self.closed = False
        self.GUI_start_action()

    def calculate_sobel(self):
        sobel = Registration.crop_and_filter(self.original.rechunk({0: 1, 1: -1, 2: -1}), sigma=self.sigma,
                                             finalsize=self.fftsize * 2)
        self.sobel = (sobel - sobel.mean(axis=(1, 2), keepdims=True))

    def calculate_cross_correlations(self):
        self.correlations = Registration.dask_cross_corr(self.sobel)

    def calculate_half_matrices(self):
        t = time.monotonic()
        self.W, self.DX_DY = Registration.calculate_halfmatrices(self.weights, self.argmax, fftsize=self.fftsize)
        print("Computation Time: " + str(time.monotonic() - t))

    def normalise_maximum_weights(self):
        self.w_diag = np.atleast_2d(np.diag(self.W))
        self.W_n = self.W / np.sqrt(self.w_diag.T * self.w_diag)

    def calculate_thresholding(self):
        # TODO(Stu) Implement GUI elements for thresholding
        min_norm = self.threshold
        nr = np.arange(self.W.shape[0]) * self.stride + self.start

        self.coords, self.weightmatrix, self.DX, self.DY, self.row_mask = \
            Registration.threshold_and_mask(min_norm,
                                            self.W,
                                            self.DX_DY,
                                            nr)
        return 0.15

    def calculate_shift_vectors(self):
        dx, dy = Registration.calc_shift_vectors(self.DX, self.DY, self.weightmatrix)
        shifts = np.stack(Registration.interp_shifts(self.coords, [dx, dy], n=self.original.shape[0]),
                          axis=1)
        self.neededMargins = np.ceil(shifts.max(axis=0)).astype(int)
        self.shifts = da.from_array(shifts, chunks=(self.dE, -1))

    def apply_shifts(self):
        padded = da.pad(self.original.rechunk({0: self.dE}),
                        ((0, 0),
                         (0, self.neededMargins[0]),
                         (0, self.neededMargins[1])
                         ),
                        mode='constant'
                        )
        self.corrected = shift_images(padded.rechunk({1: -1, 2: -1}), self.shifts)

    def plot_corrected(self):
        self.root.wm_title("Corrected Images")
        self.tracker.replace(self.corrected_images)
        self.cont_button_text.set("Save")
        self.GUI_start_action()

    def save_stack(self):
        save_directory = filedialog.askdirectory(title='Select save directory')
        if save_directory:  # if a folder was selected, don't save otherwise
            # TODO(STU) combine these operations into one or 2
            file_list = os.listdir(self.load_directory)
            file_list = [file for file in file_list if "." in file]
            file_list = [file.replace('.tif', '_shifted.tif') for file in file_list]
            for filename, image in zip(file_list, self.corrected_images):
                target = os.path.join(save_directory, filename)
                sk_imsave(target, image)
                print(f'Image saved as {target}')
        else:
            print('Data not saved')

        self.root.destroy()

    def GUI_start_action(self):
        self.root.deiconify()
        if not self.closed:
            self.isRunning = True
            while self.isRunning:
                self.root.update()
            if self.closed == True:
                self.root.destroy()

    def GUI_close_action(self):
        self.isRunning = False
        self.closed = True

    def GUI_pause_action(self):
        self.root.withdraw()
        self.isRunning = False

    def plot_corr(self, i, j):
        # fig = plt.figure(figsize=(8.2, 3.5), constrained_layout=True)
        fig = plt.figure(figsize=(4, 7), constrained_layout=True)
        fig.set_constrained_layout_pads(hspace=0.0, wspace=0.06)
        # gs = mpl.gridspec.GridSpec(2, 3,
        #                   width_ratios=[1, 1, 2.9],
        #                   #height_ratios=[4, 1]
        #                   )

        gs = mpl.gridspec.GridSpec(3, 2,
                                   height_ratios=[1, 1, 1.8],
                                   figure=fig,
                                   )

        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[1, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax3 = plt.subplot(gs[1, 1])
        ax4 = plt.subplot(gs[2, :])  # 2grid((2, 4), (0, 2), rowspan=2, colspan=2)
        ax0.imshow(self.original[i * self.stride + self.start, (640 - self.fftsize):(640 + self.fftsize),
                   (512 - self.fftsize):(512 + self.fftsize)].T,
                   cmap='gray', interpolation='none')
        ax0.set_title(f'i={i * self.stride + self.start}')
        ax1.imshow(self.sobel[i, ...].T, cmap='gray')
        ax2.imshow(self.original[j * self.stride + self.start, (640 - self.fftsize):(640 + self.fftsize),
                   (512 - self.fftsize):(512 + self.fftsize)].T,
                   cmap='gray', interpolation='none')
        ax2.set_title(f'j={j * self.stride + self.start}')
        ax3.imshow(self.sobel[j, ...].T,
                   cmap='gray', interpolation='none')
        im = ax4.imshow(self.Corr[i, j, ...].compute().T,
                        extent=[-self.fftsize, self.fftsize, -self.fftsize, self.fftsize],
                        interpolation='none')
        ax4.axhline(0, color='white', alpha=0.5)
        ax4.axvline(0, color='white', alpha=0.5)
        for ax in [ax2, ax3]:
            ax.yaxis.set_label_position("right")
            ax.tick_params(axis='y', labelright=True, labelleft=False)
        plt.colorbar(im, ax=ax4)
        if self.savefig:
            # Saving Figure for paper.
            plt.savefig('autocorrelation.pdf', dpi=300)
        plt.show()
        return fig

    def plot_masking(self, min_normed_weight):
        extent = [self.start, self.stop, self.stop, self.start]
        fig, axs = plt.subplots(1, 3, figsize=(8, 2.5), constrained_layout=True)
        im = {}
        im[0] = axs[0].imshow(self.DX_DY[0], cmap='seismic', extent=extent, interpolation='none')
        im[1] = axs[1].imshow(self.DX_DY[1], cmap='seismic', extent=extent, interpolation='none')
        im[2] = axs[2].imshow(self.W_n - np.diag(np.diag(self.W_n)), cmap='inferno',
                              extent=extent, clim=(0.0, None), interpolation='none')
        axs[0].set_ylabel('$j$')
        fig.colorbar(im[0], ax=axs[:2], shrink=0.82, fraction=0.1)
        axs[0].contourf(self.W_n, [0, min_normed_weight],
                        colors='black', alpha=0.6,
                        extent=extent, origin='upper')
        axs[1].contourf(self.W_n, [0, min_normed_weight],
                        colors='black', alpha=0.6,
                        extent=extent, origin='upper')
        CF = axs[2].contourf(self.W_n, [0, min_normed_weight],
                             colors='white', alpha=0.2,
                             extent=extent, origin='upper')
        cbar = fig.colorbar(im[2], ax=axs[2], shrink=0.82, fraction=0.1)
        cbar.ax.fill_between([0, 1], 0, min_normed_weight, color='white', alpha=0.2)
        for i in range(3):
            axs[i].set_xlabel('$i$')
            axs[i].tick_params(labelbottom=False, labelleft=False)
        axs[0].set_title('$DX_{ij}$')
        axs[1].set_title('$DY_{ij}$')
        axs[2].set_title('$W_{ij}$')
        if self.savefig:
            plt.savefig('shiftsandweights.pdf', dpi=300)
        plt.show()
        return min_normed_weight

    # def padding_solver(self):
    #     # Makes all arrays same size to make stacking easier. The padded images are not the images that are saved.
    #     shapes = list(zip(*[list(array.shape) for array in self.original]))
    #     new_size = (max(shapes[0]), max(shapes[1]))
    #     for i in range(0, len(self.original)):
    #         array = self.original[i]
    #         pads = np.array(new_size) - np.array(array.shape)
    #         if pads.sum() > 0:
    #             array = np.pad(array, (((pads[0] + 1) // 2, pads[0] // 2), ((pads[1] + 1) // 2, pads[1] // 2)),
    #                            'constant', constant_values=0)
    #             self.original[i] = array
    #     self.original = da.from_array(np.array(self.original))


if __name__ == '__main__':
    cluster = LocalCluster(n_workers=1, threads_per_worker=4)
    client = Client(cluster)
    client.upload_file('Registration.py')
    dc = DriftCorrector(0, -1, 1, 1, 250, 3, 0.5)
    dc.apply_corrections()
