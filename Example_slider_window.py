import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

import dask.array.image as daim
from Correctors import ScrollBarImagePlot



class GUI:

    def __init__(self):
        self.root = tk.Tk()
        self.root.wm_title("Slider Test")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        load_directory = filedialog.askdirectory(title='Select a folder containing image files')
        original = daim.imread(load_directory + '\\*.tif')

        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.subplots(1, 1)
        tracker = ScrollBarImagePlot(ax, original)
        canvas = FigureCanvasTkAgg(fig, master=self.root)  # A tk.DrawingArea.
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        slider = tk.Scale(self.root, from_=0, to=original.shape[0] - 1, orient=tk.HORIZONTAL, command=tracker.onscroll)
        slider.pack(fill=tk.X)
        self.cont_button = tk.Button(self.root, command=self.pause, text="Continue")
        self.cont_button.pack()

        self.isRunning = False
        self.closed = False

    def start(self):
        if not self.closed:
            self.isRunning = True
            while self.isRunning:
                self.root.update()
            if self.closed == True:
                self.root.destroy()

    def on_closing(self):
        self.isRunning = False
        self.closed = True

    def pause(self):
        self.isRunning = False
        self.cont_button.setvar("text", "Quit")


if __name__ == "__main__":
    mygui = GUI()
    mygui.start()
    print("paused")
    mygui.start()
