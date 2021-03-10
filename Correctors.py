import numpy as np

import time

import dask.array as da
import dask.array.image as daim

import scipy.ndimage as ndi
from scipy.signal import convolve2d

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
import Registration


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
        self.sigma = sigma
        self.threshold = threshold

    def apply_corrections(self, stack):
        self.original = stack
        self.calculate_sobel()

        self.calculate_cross_correlations()

        self.weights, self.argmax = Registration.max_and_argmax(self.correlations)

        self.calculate_half_matrices()

        self.normalise_maximum_weights()

        self.calculate_thresholding()

        self.calculate_shift_vectors()

        self.apply_shifts()
        return self.corrected.compute()

    def calculate_sobel(self):
        sobel = Registration.crop_and_filter(self.original.rechunk({0: self.dE}), sigma=self.sigma,
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


class MediPixCorrector:

    def __init__(self):
        self.stack = None
        self.norm_image = None
        self.is_norm = False
        self.bad_pixel_image = daim.imread('badPixelImage12KV.tif').compute()[0]

    def set_stack(self, stack, norm_image):
        self.original = stack
        self.num, self.width, self.height = stack.shape
        self.norm_image = norm_image
        self.dE = 4

    def apply_corrections(self):
        if self.original is not None:
            self.__fix_overlap()

        self.__fix_bad_pixels()

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
            meaner = np.array([[0.125, 0.125, 0.125], [0.125, 0, 0.125], [0.125, 0.125, 0.125]])
            meaned = convolve2d(image, meaner, mode='same')
            image[self.bad_pixel_image == 1] = meaned[self.bad_pixel_image == 1]
            image[image > 3000] = meaned[image > 3000]
            image[image <= 0.1] = meaned[image <= 0.1]
            self.stack[i] = image

    def __make_norm(self):
        image = np.mean(self.stack, axis=0)
        return image / np.mean(image)

    def __apply_normalisation(self):
        self.stack = self.stack / np.repeat(self.norm_image, self.stack.shape[0], axis=0)


class ScrollBarImagePlot(object):
    def __init__(self, fig, ax, X):
        self.fig = fig
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
        self.fig.canvas.draw()

    def replace(self, X):
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = 0

        self.im = self.ax.imshow(X[self.ind, :, :].T, cmap='gray', vmax=self.X.max())
        self.update()


class StackPlotter:

    def __init__(self, root=None):
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        self.closed = False
        self.cont_button_text = tk.StringVar()
        self.root.protocol("WM_DELETE_WINDOW", self.GUI_close_action)
        self.tracker = None
        self.isRunning = False

    def GUI_start_action(self):
        self.root.deiconify()
        if not self.closed:
            self.isRunning = True
            while self.isRunning:
                self.root.update()
            if self.closed:
                self.root.destroy()

    def GUI_close_action(self):
        self.isRunning = False
        self.closed = True

    def GUI_pause_action(self):
        self.root.withdraw()
        self.isRunning = False

    def plot_stack(self, stack, title="", button_txt="Continue"):
        self.root.wm_title(title)
        if self.tracker is None:
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.subplots(1, 1)
            self.tracker = ScrollBarImagePlot(fig, ax, stack)
            canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.slider = tk.Scale(self.root, from_=0, to=stack.shape[0] - 1, orient=tk.HORIZONTAL,
                              command=self.tracker.onscroll)
            self.slider.pack(fill=tk.X)

            cont_button = tk.Button(self.root, command=self.GUI_pause_action, textvariable=self.cont_button_text)
            self.cont_button_text.set(button_txt)
            cont_button.pack()
        else:
            self.tracker.replace(stack)
            self.cont_button_text.set(button_txt)
            self.slider.set(0)

        self.isRunning = False
        self.closed = False
        self.GUI_start_action()
