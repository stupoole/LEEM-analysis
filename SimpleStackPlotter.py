from Correctors import  StackPlotter
import tkinter as tk
from tkinter import filedialog
import os
from skimage.io import imread as sk_imread
import numpy as np

class Plotter:

    def __init__(self):
        self.root = tk.Tk()
        self.plotter = StackPlotter(self.root)

    def load_images(self):
        self.load_directory = filedialog.askdirectory(title='Select a folder containing image files')
        file_list = os.listdir(self.load_directory)
        file_list = [file for file in file_list if file[-4:] == ".tif"]
        self.stack = list()

        for i in range(0, len(file_list)):
            print("Loading " + os.path.join(self.load_directory, file_list[i]))
            self.stack.append(sk_imread(os.path.join(self.load_directory, file_list[i]), as_gray=True))
        self.padding_solver()
        # self.original = daim.imread(os.path.join(self.load_directory, '*.tif'))
        self.plotter.plot_stack(self.stack, 'Original ' + self.load_directory.split(os.sep)[-1])

    def padding_solver(self):
        # Makes all arrays same size to make stacking easier. The padded images are not the images that are saved.
        shapes = list(zip(*[list(array.shape) for array in self.stack]))
        new_size = (max(shapes[0]), max(shapes[1]))
        for i in range(0, len(self.stack)):
            array = self.stack[i]
            pads = np.array(new_size) - np.array(array.shape)
            if pads.sum() > 0:
                array = np.pad(array, (((pads[0] + 1) // 2, pads[0] // 2), ((pads[1] + 1) // 2, pads[1] // 2)),
                               'constant', constant_values=0)
                self.stack[i] = array
        self.stack = np.array(self.stack)


if __name__ == '__main__':
    plotter = Plotter()
    plotter.load_images()