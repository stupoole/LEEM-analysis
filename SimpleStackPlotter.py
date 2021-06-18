from Correctors import StackPlotter, padding_solver
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
        self.stack = padding_solver(self.stack)
        # self.original = daim.imread(os.path.join(self.load_directory, '*.tif'))
        self.plotter.plot_stack(self.stack, 'Original ' + self.load_directory.split(os.sep)[-1])


if __name__ == '__main__':
    plotter = Plotter()
    plotter.load_images()
