import os
import csv
import time
import tkinter as tk
from tkinter import filedialog, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from analysis import PolarizationProcessor


class PolarizationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polarization Analysis GUI")
        self.default_geometry = "1200x850"
        self.geometry(self.default_geometry)
        plt.rcParams["figure.autolayout"] = True

        self.processor = PolarizationProcessor()
        self.batch_processor1 = PolarizationProcessor()
        self.batch_processor2 = PolarizationProcessor()
        self.roi_start = None
        self.roi_rect = None
        self.batch_roi_start = None
        self.batch_roi_rect = None
        self.compare_roi_start = None
        self.compare_roi_rect = None
        self.compare_processor_image = None
        self.compare_results = []

        self.page_buttons_frame = tk.Frame(self)
        self.page_buttons_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(self.page_buttons_frame, text="Single Image", command=self.show_single_page).pack(side=tk.LEFT, padx=4)
        tk.Button(self.page_buttons_frame, text="Batch Folder", command=self.show_batch_page).pack(side=tk.LEFT, padx=4)
        tk.Button(self.page_buttons_frame, text="Batch Compare", command=self.show_compare_page).pack(side=tk.LEFT, padx=4)

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.page1_frame = tk.Frame(self.main_frame)
        self.page2_frame = tk.Frame(self.main_frame)
        self.page3_frame = tk.Frame(self.main_frame)

        self.status_label = tk.Label(self, text="Load an image to start", anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=4)

        self._create_single_page()
        self._create_batch_page()
        self._create_compare_page()

        self.show_single_page()

    def _create_single_page(self):
        self.left_frame = tk.Frame(self.page1_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.grid_frame = tk.Frame(self.page1_frame)
        self.grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)

        self.load_btn = tk.Button(self.left_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(fill=tk.X, pady=4)

        self.display_demosaic_btn = tk.Button(self.left_frame, text="Display Demosaiced", command=self.display_demosaic)
        self.display_demosaic_btn.pack(fill=tk.X, pady=4)

        self.display_saturation_histogram_btn = tk.Button(self.left_frame, text="Display Saturation Histogram", command=self.display_saturation_histogram)
        self.display_saturation_histogram_btn.pack(fill=tk.X, pady=4)

        self.filter_label = tk.Label(self.left_frame, text="Select Filter:")
        self.filter_label.pack(pady=(12, 2))
        self.filter_var = tk.StringVar(value="None")
        self.filter_menu = tk.OptionMenu(self.left_frame, self.filter_var, "None", "Gaussian Blur", command=self.update_filter_params)
        self.filter_menu.pack(fill=tk.X, pady=2)

        self.params_frame = tk.Frame(self.left_frame)
        self.params_frame.pack(fill=tk.X, pady=4)

        self.kernel_label = tk.Label(self.params_frame, text="Kernel Size (odd):")
        self.kernel_label.grid(row=0, column=0, sticky="w")
        self.kernel_var = tk.IntVar(value=5)
        self.kernel_entry = tk.Entry(self.params_frame, textvariable=self.kernel_var)
        self.kernel_entry.grid(row=0, column=1, sticky="ew", padx=4)

        self.sigma_label = tk.Label(self.params_frame, text="Sigma:")
        self.sigma_label.grid(row=1, column=0, sticky="w")
        self.sigma_var = tk.DoubleVar(value=1.0)
        self.sigma_entry = tk.Entry(self.params_frame, textvariable=self.sigma_var)
        self.sigma_entry.grid(row=1, column=1, sticky="ew", padx=4)
        self.params_frame.grid_columnconfigure(1, weight=1)
        self.params_frame.pack_forget()

        self.apply_filter_btn = tk.Button(self.left_frame, text="Apply Filter", command=self.apply_filter)
        self.apply_filter_btn.pack(fill=tk.X, pady=4)

        self.preview_btn = tk.Button(self.left_frame, text="Preview Filtered Demosaiced", command=self.preview_filter)
        self.preview_btn.pack(fill=tk.X, pady=4)

        self.pol_var = tk.StringVar(value="linear")
        tk.Label(self.left_frame, text="Polarization Type:").pack(pady=(12, 2), anchor="w")
        tk.Radiobutton(self.left_frame, text="Linear", variable=self.pol_var, value="linear").pack(anchor="w")
        tk.Radiobutton(self.left_frame, text="Circular", variable=self.pol_var, value="circular").pack(anchor="w")

        self.calculate_btn = tk.Button(self.left_frame, text="Calculate Polarization", command=self.calculate)
        self.calculate_btn.pack(fill=tk.X, pady=12)

        self.export_csv_btn = tk.Button(self.left_frame, text="Export to CSV", command=self.export_csv)
        self.export_csv_btn.pack(fill=tk.X, pady=4)

        self.export_png_btn = tk.Button(self.left_frame, text="Export Graphs as PNG", command=self.export_single_png)
        self.export_png_btn.pack(fill=tk.X, pady=4)

        self.canvas = tk.Canvas(self.grid_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        self.canvas.bind('<Configure>', lambda event: self.resize_image())
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)

        self.dolp_canvas = None
        self.aolp_canvas = None
        self.hist_canvas = None
        self.single_plot_figures = {}

    def _create_batch_page(self):
        self.batch_left_frame = tk.Frame(self.page2_frame)
        self.batch_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.batch_grid_frame = tk.Frame(self.page2_frame)
        self.batch_grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.batch_grid_frame.grid_rowconfigure(0, weight=1)
        self.batch_grid_frame.grid_rowconfigure(1, weight=1)
        self.batch_grid_frame.grid_rowconfigure(2, weight=1)
        self.batch_grid_frame.grid_columnconfigure(0, weight=1)
        self.batch_grid_frame.grid_columnconfigure(1, weight=1)
        self.batch_grid_frame.grid_columnconfigure(2, weight=1)

        self.batch_load_btn = tk.Button(self.batch_left_frame, text="Load Folder", command=self.load_folder)
        self.batch_load_btn.pack(fill=tk.X, pady=4)

        self.batch_folder_label = tk.Label(self.batch_left_frame, text="No folder loaded", anchor="w", justify="left", wraplength=200)
        self.batch_folder_label.pack(fill=tk.X, pady=4)

        self.batch_mode_var = tk.StringVar(value="mean")
        tk.Label(self.batch_left_frame, text="Batch monolithic mode:").pack(pady=(12, 2), anchor="w")
        self.batch_mode_menu = tk.OptionMenu(self.batch_left_frame, self.batch_mode_var, "mean", "median", "stokes_mean")
        self.batch_mode_menu.pack(fill=tk.X, pady=2)

        self.batch_pol_var = tk.StringVar(value="linear")
        tk.Label(self.batch_left_frame, text="Polarization Type:").pack(pady=(12, 2), anchor="w")
        tk.Radiobutton(self.batch_left_frame, text="Linear", variable=self.batch_pol_var, value="linear").pack(anchor="w")
        tk.Radiobutton(self.batch_left_frame, text="Circular", variable=self.batch_pol_var, value="circular").pack(anchor="w")

        self.batch_calculate_btn = tk.Button(self.batch_left_frame, text="Calculate Batch", command=self.calculate_batch)
        self.batch_calculate_btn.pack(fill=tk.X, pady=12)

        self.batch_export_csv_btn = tk.Button(self.batch_left_frame, text="Export Batch CSV", command=self.export_batch_csv)
        self.batch_export_csv_btn.pack(fill=tk.X, pady=4)

        self.batch_export_png_btn = tk.Button(self.batch_left_frame, text="Export Graphs as PNG", command=self.export_batch_png)
        self.batch_export_png_btn.pack(fill=tk.X, pady=4)

        self.batch_canvas = tk.Canvas(self.batch_grid_frame, bg="black")
        self.batch_canvas.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.batch_canvas.bind('<Configure>', lambda event: self.resize_batch_image())
        self.batch_canvas.bind('<ButtonPress-1>', self.batch_on_mouse_press)
        self.batch_canvas.bind('<B1-Motion>', self.batch_on_mouse_drag)
        self.batch_canvas.bind('<ButtonRelease-1>', self.batch_on_mouse_release)

        self.batch_plot_canvases = {}
        self.batch_plot_figures = {}
        self.batch_plot_data = {}

    def _create_compare_page(self):
        self.compare_left_frame = tk.Frame(self.page3_frame)
        self.compare_left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.compare_grid_frame = tk.Frame(self.page3_frame)
        self.compare_grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.compare_grid_frame.grid_rowconfigure(0, weight=1)
        self.compare_grid_frame.grid_rowconfigure(1, weight=1)
        self.compare_grid_frame.grid_rowconfigure(2, weight=1)
        self.compare_grid_frame.grid_columnconfigure(0, weight=1)
        self.compare_grid_frame.grid_columnconfigure(1, weight=1)
        self.compare_grid_frame.grid_columnconfigure(2, weight=1)

        self.compare_load_btn1 = tk.Button(self.compare_left_frame, text="Load Folder 1", command=self.load_folder1)
        self.compare_load_btn1.pack(fill=tk.X, pady=4)
        self.compare_folder_label1 = tk.Label(self.compare_left_frame, text="No folder 1 loaded", anchor="w", justify="left", wraplength=200)
        self.compare_folder_label1.pack(fill=tk.X, pady=4)

        self.compare_load_btn2 = tk.Button(self.compare_left_frame, text="Load Folder 2", command=self.load_folder2)
        self.compare_load_btn2.pack(fill=tk.X, pady=4)
        self.compare_folder_label2 = tk.Label(self.compare_left_frame, text="No folder 2 loaded", anchor="w", justify="left", wraplength=200)
        self.compare_folder_label2.pack(fill=tk.X, pady=4)

        self.compare_mode_var = tk.StringVar(value="mean")
        tk.Label(self.compare_left_frame, text="Comparison mode:").pack(pady=(12, 2), anchor="w")
        self.compare_mode_menu = tk.OptionMenu(self.compare_left_frame, self.compare_mode_var, "mean", "median", "stokes_mean")
        self.compare_mode_menu.pack(fill=tk.X, pady=2)

        self.compare_pol_var = tk.StringVar(value="linear")
        tk.Label(self.compare_left_frame, text="Polarization Type:").pack(pady=(12, 2), anchor="w")
        tk.Radiobutton(self.compare_left_frame, text="Linear", variable=self.compare_pol_var, value="linear").pack(anchor="w")
        tk.Radiobutton(self.compare_left_frame, text="Circular", variable=self.compare_pol_var, value="circular").pack(anchor="w")

        self.compare_calculate_btn = tk.Button(self.compare_left_frame, text="Calculate Compare", command=self.calculate_compare)
        self.compare_calculate_btn.pack(fill=tk.X, pady=12)

        self.compare_export_csv_btn = tk.Button(self.compare_left_frame, text="Export Compare CSV", command=self.export_compare_csv)
        self.compare_export_csv_btn.pack(fill=tk.X, pady=4)

        self.compare_export_png_btn = tk.Button(self.compare_left_frame, text="Export Graphs as PNG", command=self.export_compare_png)
        self.compare_export_png_btn.pack(fill=tk.X, pady=4)

        self.compare_canvas = tk.Canvas(self.compare_grid_frame, bg="black")
        self.compare_canvas.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.compare_canvas.bind('<Configure>', lambda event: self.resize_compare_image())
        self.compare_canvas.bind('<ButtonPress-1>', self.compare_on_mouse_press)
        self.compare_canvas.bind('<B1-Motion>', self.compare_on_mouse_drag)
        self.compare_canvas.bind('<ButtonRelease-1>', self.compare_on_mouse_release)

        self.compare_plot_canvases = {}
        self.compare_plot_figures = {}
        self.compare_plot_data = {}

        # state for interactive plot isolation
        self._isolated_windows = {}

    def show_single_page(self):
        self.page_active = 'single'
        self.page2_frame.pack_forget()
        self.page1_frame.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Single-image analysis page active")

    def show_batch_page(self):
        self.page_active = 'batch'
        self.page1_frame.pack_forget()
        self.page3_frame.pack_forget()
        self.page2_frame.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Batch folder analysis page active")

    def show_compare_page(self):
        self.page_active = 'compare'
        self.page1_frame.pack_forget()
        self.page2_frame.pack_forget()
        self.page3_frame.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Batch comparison page active")

    def update_filter_params(self, value):
        if value == "Gaussian Blur":
            self.params_frame.pack(fill=tk.X, pady=4)
        else:
            self.params_frame.pack_forget()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if not file_path:
            return
        success = self.processor.load_image(file_path)
        if success:
            self.status_label.config(text="Image loaded. Drag on the image to select ROI.")
            self.roi_start = None
            self.roi_rect = None
            self.processor.roi = None
            self.resize_image()
            self.clear_plots()
        else:
            self.status_label.config(text="Failed to load image.")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        success = self.processor.load_folder(folder_path)
        if success:
            self.batch_folder_label.config(text=f"Loaded {len(self.processor.batch_files)} images\n{folder_path}")
            self.batch_roi_start = None
            self.batch_roi_rect = None
            self.processor.batch_roi = None
            self.processor.roi = None
            self.resize_batch_image()
            self.clear_batch_plots()
            self.status_label.config(text="Folder loaded. Drag on the batch image to select ROI.")
        else:
            self.status_label.config(text="Failed to load folder or no valid images found.")

    def load_folder1(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        success = self.batch_processor1.load_folder(folder_path)
        if success:
            self.compare_folder_label1.config(text=f"Loaded {len(self.batch_processor1.batch_files)} images\n{folder_path}")
            self.compare_processor_image = self.batch_processor1.batch_first_img.copy()
            self.compare_roi_start = None
            self.compare_roi_rect = None
            self.batch_processor1.batch_roi = None
            self.batch_processor1.roi = None
            self.compare_compare_image_display = self.compare_processor_image
            self.resize_compare_image()
            self.clear_compare_plots()
            self.status_label.config(text="Folder 1 loaded. Drag on the compare image to select ROI.")
        else:
            self.status_label.config(text="Failed to load folder 1 or no valid images found.")

    def load_folder2(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
        success = self.batch_processor2.load_folder(folder_path)
        if success:
            self.compare_folder_label2.config(text=f"Loaded {len(self.batch_processor2.batch_files)} images\n{folder_path}")
            if not hasattr(self, 'compare_processor_image') or self.compare_processor_image is None:
                self.compare_processor_image = self.batch_processor2.batch_first_img.copy()
                self.resize_compare_image()
            self.batch_processor2.batch_roi = None
            self.batch_processor2.roi = None
            self.clear_compare_plots()
            self.status_label.config(text="Folder 2 loaded. Drag on the compare image to select ROI.")
        else:
            self.status_label.config(text="Failed to load folder 2 or no valid images found.")

    def on_mouse_press(self, event):
        if self.processor.original_img is None:
            return
        self.roi_start = (event.x, event.y)
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = None

    def on_mouse_drag(self, event):
        if self.roi_start is None:
            return
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y
        self.roi_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def on_mouse_release(self, event):
        if self.roi_start is None or self.processor.original_img is None:
            return
        x1, y1 = self.roi_start
        x2, y2 = event.x, event.y
        self.roi_start = None
        self.roi_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        image_roi = self._canvas_to_image_roi(x1, y1, x2, y2)
        self.processor.roi = image_roi
        self.status_label.config(text=f"ROI selected: {tuple(map(int, image_roi))}")

    def batch_on_mouse_press(self, event):
        if self.processor.original_img is None:
            return
        self.batch_roi_start = (event.x, event.y)
        if self.batch_roi_rect:
            self.batch_canvas.delete(self.batch_roi_rect)
            self.batch_roi_rect = None

    def batch_on_mouse_drag(self, event):
        if self.batch_roi_start is None:
            return
        if self.batch_roi_rect:
            self.batch_canvas.delete(self.batch_roi_rect)
        x1, y1 = self.batch_roi_start
        x2, y2 = event.x, event.y
        self.batch_roi_rect = self.batch_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def batch_on_mouse_release(self, event):
        if self.batch_roi_start is None or self.processor.original_img is None:
            return
        x1, y1 = self.batch_roi_start
        x2, y2 = event.x, event.y
        self.batch_roi_start = None
        self.batch_roi_rect = self.batch_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        image_roi = self._batch_canvas_to_image_roi(x1, y1, x2, y2)
        self.processor.batch_set_roi(image_roi)
        self.status_label.config(text=f"Batch ROI selected: {tuple(map(int, image_roi))}")

    def compare_on_mouse_press(self, event):
        if self.compare_processor_image is None:
            return
        self.compare_roi_start = (event.x, event.y)
        if self.compare_roi_rect:
            self.compare_canvas.delete(self.compare_roi_rect)
            self.compare_roi_rect = None

    def compare_on_mouse_drag(self, event):
        if self.compare_roi_start is None:
            return
        if self.compare_roi_rect:
            self.compare_canvas.delete(self.compare_roi_rect)
        x1, y1 = self.compare_roi_start
        x2, y2 = event.x, event.y
        self.compare_roi_rect = self.compare_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def compare_on_mouse_release(self, event):
        if self.compare_roi_start is None or not hasattr(self, 'compare_processor_image'):
            return
        x1, y1 = self.compare_roi_start
        x2, y2 = event.x, event.y
        self.compare_roi_start = None
        self.compare_roi_rect = self.compare_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        image_roi = self._compare_canvas_to_image_roi(x1, y1, x2, y2)
        self.batch_processor1.batch_set_roi(image_roi)
        self.batch_processor2.batch_set_roi(image_roi)
        self.status_label.config(text=f"Comparison ROI selected: {tuple(map(int, image_roi))}")

    def _canvas_to_image_roi(self, x1, y1, x2, y2):
        img_width, img_height = self.processor.original_img.size
        return self._canvas_to_image_roi_generic(self.canvas, img_width, img_height, x1, y1, x2, y2)

    def _batch_canvas_to_image_roi(self, x1, y1, x2, y2):
        img_height, img_width = self.processor.batch_first_img.shape[:2]
        return self._canvas_to_image_roi_generic(self.batch_canvas, img_width, img_height, x1, y1, x2, y2)

    def _compare_canvas_to_image_roi(self, x1, y1, x2, y2):
        img_height, img_width = self.compare_processor_image.shape[:2]
        return self._canvas_to_image_roi_generic(self.compare_canvas, img_width, img_height, x1, y1, x2, y2)

    def resize_image(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        if self.processor.original_img is None:
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            return
        img_width, img_height = self.processor.original_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = self.processor.original_img.resize((new_width, new_height), Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(resized_img)
        self.canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=self.img_tk)
        self.canvas.image = self.img_tk
        self._redraw_roi()

    def resize_batch_image(self):
        self.batch_canvas.delete("all")
        self.batch_canvas.update_idletasks()
        if not hasattr(self.processor, 'batch_first_img') or self.processor.batch_first_img is None:
            return
        canvas_width = self.batch_canvas.winfo_width()
        canvas_height = self.batch_canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            return
        img_height, img_width = self.processor.batch_first_img.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = Image.fromarray(self.processor.batch_first_img).resize((new_width, new_height), Image.LANCZOS)
        self.batch_img_tk = ImageTk.PhotoImage(resized_img)
        self.batch_canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=self.batch_img_tk)
        self.batch_canvas.image = self.batch_img_tk
        self._batch_redraw_roi()

    def resize_compare_image(self):
        self.compare_canvas.delete("all")
        self.compare_canvas.update_idletasks()
        if not hasattr(self, 'compare_processor_image') or self.compare_processor_image is None:
            return
        canvas_width = self.compare_canvas.winfo_width()
        canvas_height = self.compare_canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2:
            return
        img_height, img_width = self.compare_processor_image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        resized_img = Image.fromarray(self.compare_processor_image).resize((new_width, new_height), Image.LANCZOS)
        self.compare_img_tk = ImageTk.PhotoImage(resized_img)
        self.compare_canvas.create_image((canvas_width - new_width) // 2, (canvas_height - new_height) // 2, anchor=tk.NW, image=self.compare_img_tk)
        self.compare_canvas.image = self.compare_img_tk
        self._compare_redraw_roi()

    def _redraw_roi(self):
        img_width, img_height = self.processor.original_img.size
        self._redraw_roi_generic(self.canvas, self.processor.roi, img_width, img_height, 'roi_rect')

    def _batch_redraw_roi(self):
        img_height, img_width = self.processor.batch_first_img.shape[:2]
        self._redraw_roi_generic(self.batch_canvas, getattr(self.processor, 'batch_roi', None), img_width, img_height, 'batch_roi_rect')

    def _compare_redraw_roi(self):
        if not hasattr(self, 'compare_processor_image') or self.compare_processor_image is None or self.batch_processor1.batch_roi is None:
            return
        img_height, img_width = self.compare_processor_image.shape[:2]
        self._redraw_roi_generic(self.compare_canvas, self.batch_processor1.batch_roi, img_width, img_height, 'compare_roi_rect')

    # --- Generic helpers to reduce duplication ---
    def _canvas_to_image_roi_generic(self, canvas, img_width, img_height, x1, y1, x2, y2):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        scale = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 1.0
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        roi_x1 = max(0, min(img_width, (x1 - x_offset) / scale))
        roi_y1 = max(0, min(img_height, (y1 - y_offset) / scale))
        roi_x2 = max(0, min(img_width, (x2 - x_offset) / scale))
        roi_y2 = max(0, min(img_height, (y2 - y_offset) / scale))
        return (min(roi_x1, roi_x2), min(roi_y1, roi_y2), abs(roi_x2 - roi_x1), abs(roi_y2 - roi_y1))

    def _redraw_roi_generic(self, canvas, roi, img_width, img_height, rect_attr_name):
        if roi is None:
            return
        x, y, w, h = roi
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        scale = min(canvas_width / img_width, canvas_height / img_height) if img_width > 0 and img_height > 0 else 1.0
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        x1 = int(x_offset + x * scale)
        y1 = int(y_offset + y * scale)
        x2 = int(x_offset + (x + w) * scale)
        y2 = int(y_offset + (y + h) * scale)
        existing = getattr(self, rect_attr_name, None)
        if existing:
            canvas.delete(existing)
        new_rect = canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        setattr(self, rect_attr_name, new_rect)

    def _draw_plot_generic(self, fig, master_frame, canvas_store, figures_store, attr_name, row, column, adjust_kwargs=None):
        if adjust_kwargs is None:
            adjust_kwargs = {'left': 0.08, 'right': 0.96, 'top': 0.90, 'bottom': 0.12, 'pad': 1.5}
        fig.tight_layout(pad=adjust_kwargs.get('pad', 1.5))
        fig.subplots_adjust(left=adjust_kwargs.get('left', 0.08), right=adjust_kwargs.get('right', 0.96), top=adjust_kwargs.get('top', 0.90), bottom=adjust_kwargs.get('bottom', 0.12))
        # Determine existing canvas
        canvas = None
        if canvas_store is not None:
            canvas = canvas_store.get(attr_name)
        else:
            canvas = getattr(self, attr_name, None)

        if canvas is not None:
            try:
                canvas.get_tk_widget().destroy()
            except Exception:
                pass

        canvas = FigureCanvasTkAgg(fig, master=master_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)

        if canvas_store is not None:
            canvas_store[attr_name] = canvas
            if figures_store is not None:
                figures_store[attr_name] = fig
        else:
            setattr(self, attr_name, canvas)
            if figures_store is not None:
                self.single_plot_figures[attr_name] = fig

        # Bind click event for interactive isolation when canvas_store is used (batch/compare)
        try:
            if canvas_store is not None:
                widget = canvas.get_tk_widget()
                widget.bind('<Button-1>', lambda event, name=attr_name, store=canvas_store: self.on_plot_click(name, store))
        except Exception:
            pass

        self._redraw_canvas(canvas)
        plt.close(fig)

    def _make_polar_scatter_fig(self, theta, radii, values, cmap, title, vmin=None, vmax=None, size=20):
        fig, ax = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
        # Use scatter with explicit vmin/vmax when provided
        scatter = ax.scatter(theta, radii, c=values, cmap=cmap, s=size, edgecolors='black', linewidth=0.1, vmin=vmin, vmax=vmax)
        ax.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
        ax.set_title(title, fontsize=10, pad=10)
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.08)
        return fig

    def _get_plot_colormap(self, cmap_name):
        return plt.get_cmap(cmap_name) if isinstance(cmap_name, str) else cmap_name

    def _make_custom_colormap(self, low_color, middle_color, high_color):
        return mcolors.LinearSegmentedColormap.from_list(
            'custom', [low_color, middle_color, high_color], N=256
        )

    def display_demosaic(self):
        if self.processor.img is None:
            self.status_label.config(text="Load an image first.")
            return
        display = self.processor.preview_filtered_demosaiced()
        cv2.imshow('Demosaiced Images', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def display_saturation_histogram(self):
        if self.processor.img is None:
            self.status_label.config(text="Load an image first.")
            return
        hist = self.processor.calculate_saturation_histogram(roi=self.processor.roi)
        hist.show()
        return

    def apply_filter(self):
        if self.processor.img_org is None:
            self.status_label.config(text="Load an image first.")
            return
        filter_name = self.filter_var.get()
        if filter_name == "None":
            self.processor.reset_image()
            self.status_label.config(text="Reset to original image.")
        else:
            kernel = self.kernel_var.get()
            if kernel % 2 == 0:
                kernel += 1
                self.kernel_var.set(kernel)
            sigma = self.sigma_var.get()
            self.processor.apply_filter(filter_name, kernel=kernel, sigma=sigma)
            self.status_label.config(text=f"Filter applied: {filter_name}.")
        self.resize_image()
        self.clear_plots()

    def preview_filter(self):
        if self.processor.img_org is None:
            self.status_label.config(text="Load an image first.")
            return
        filter_name = self.filter_var.get()
        if filter_name == "None":
            self.status_label.config(text="Select a filter first.")
            return
        kernel = self.kernel_var.get()
        if kernel % 2 == 0:
            kernel += 1
            self.kernel_var.set(kernel)
        sigma = self.sigma_var.get()
        preview_image = self.processor.preview_filtered_demosaiced(filter_name, kernel=kernel, sigma=sigma)
        cv2.imshow('Filtered Demosaiced Preview', preview_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate(self):
        if self.processor.img is None:
            self.status_label.config(text="Load an image first.")
            return
        pol_type = self.pol_var.get()
        self.processor.calculate(pol_type, roi=self.processor.roi)
        self.update_plots()
        self.status_label.config(text=f"Polarization calculated for {pol_type}.")

    def calculate_batch(self):
        if not hasattr(self.processor, 'batch_files') or not self.processor.batch_files:
            self.status_label.config(text="Load a folder first.")
            return
        if self.processor.batch_roi is None:
            self.status_label.config(text="Select a ROI on the batch image first.")
            return
        mode = self.batch_mode_var.get()
        pol_type = self.batch_pol_var.get()
        results = self.processor.batch_calculate(mode=mode, pol_type=pol_type)
        if results:
            self.update_batch_plots()
            self.status_label.config(text=f"Batch calculated for {len(results)} images using {mode}.")
        else:
            self.status_label.config(text="Batch calculation failed.")

    def calculate_compare(self):
        if not hasattr(self.batch_processor1, 'batch_files') or not self.batch_processor1.batch_files:
            self.status_label.config(text="Load folder 1 first.")
            return
        if not hasattr(self.batch_processor2, 'batch_files') or not self.batch_processor2.batch_files:
            self.status_label.config(text="Load folder 2 first.")
            return
        if self.batch_processor1.batch_roi is None:
            self.status_label.config(text="Select a ROI on the compare image first.")
            return

        mode = self.compare_mode_var.get()
        pol_type = self.compare_pol_var.get()
        results1 = self.batch_processor1.batch_calculate(mode=mode, pol_type=pol_type)
        results2 = self.batch_processor2.batch_calculate(mode=mode, pol_type=pol_type)
        if not results1 or not results2:
            self.status_label.config(text="Comparison failed: one dataset did not produce any results.")
            return

        if len(results1) != len(results2):
            self.status_label.config(text=f"Comparison failed: dataset sizes differ (folder1={len(results1)}, folder2={len(results2)}). Ensure both folders contain the same number of matching images.")
            return

        self.compare_results = []
        mismatched_count = 0
        first_mismatch = None
        for r1, r2 in zip(results1, results2):
            # if r1['params'] != r2['params']: #this checks if the photos are the same but in reality they will always be slightly different.
            #     mismatched_count += 1
            #     if first_mismatch is None:
            #         first_mismatch = (r1, r2)
            #     continue
            # 
            if pol_type == 'linear':
                dolp1 = float(r1['pol'][0])*100
                dolp2 = float(r2['pol'][0])*100
                dolp_diff = (dolp1 - dolp2)

                aolp1 = float(r1['pol'][1])
                aolp2 = float(r2['pol'][1])
                aolp_diff = ((aolp1 - aolp2 + np.pi) % (2 * np.pi)) - np.pi
                dolp_std_diff = float(r1['dolp_std']) - float(r2['dolp_std'])
                aolp_std_diff = float(r1['aolp_std']) - float(r2['aolp_std'])
                self.compare_results.append({
                    'params': r1['params'],
                    'dolp_diff': dolp_diff,
                    'dolp_std_diff': dolp_std_diff,
                    'aolp_diff': aolp_diff,
                    'aolp_std_diff': aolp_std_diff,
                    'sat_diff': float(r1['saturation']) - float(r2['saturation'])
                })
            else:
                docp_diff = float(r1['pol'][0]) - float(r2['pol'][0])
                docp_std_diff = float(r1['dolp_std']) - float(r2['dolp_std'])
                self.compare_results.append({
                    'params': r1['params'],
                    'docp_diff': docp_diff,
                    'docp_std_diff': docp_std_diff,
                    'sat_diff': float(r1['saturation']) - float(r2['saturation'])
                })

        if self.compare_results:
            self.update_compare_plots()
            self.status_label.config(text=f"Comparison calculated for {len(self.compare_results)} matched points ({mismatched_count} mismatches skipped).")
        else:
            if mismatched_count == len(results1) and first_mismatch is not None:
                r1, r2 = first_mismatch
                self.status_label.config(text=(
                    f"Comparison failed: same dataset size but no matching image metadata was found. "
                    f"First mismatch: folder1 params={r1['params']} vs folder2 params={r2['params']}.")
                )
            else:
                self.status_label.config(text="Comparison failed: no matching image metadata found. Check that both folders contain corresponding images with matching parameters.")

    def update_plots(self):
        self._clear_plot_canvas('dolp_canvas')
        self._clear_plot_canvas('aolp_canvas')
        self._clear_plot_canvas('hist_canvas')
        if self.processor.pol_params is None:
            return
        S, pol_vect = self.processor.pol_params
        DoLP, AoLP = pol_vect
        pol_type = self.pol_var.get()

        fig1, ax1 = plt.subplots(figsize=(4, 3))
        if pol_type == "circular":
            im1 = ax1.imshow(DoLP, cmap='RdBu', vmin=-1, vmax=1)
            ax1.set_title('DoCP')
        else:
            im1 = ax1.imshow(DoLP, cmap='viridis')
            ax1.set_title('DoLP')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        self._draw_plot(fig1, 'dolp_canvas', row=0, column=1)

        fig2, ax2 = plt.subplots(figsize=(4, 3))
        if pol_type == "linear":
            im2 = ax2.imshow(AoLP, cmap='RdBu')
            ax2.set_title('AoLP')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        else:
            # For circular polarisation show Circular Polarisation Ratio (RH/LH)
            # CPR = (1 + DoCP) / (1 - DoCP) (clip extreme values for display)
            with np.errstate(divide='ignore', invalid='ignore'):
                cpr = (1.0 + DoLP) / (1.0 - DoLP)
            # Replace infinities and NaNs with large number for plotting
            cpr = np.nan_to_num(cpr, nan=0.0, posinf=np.nanmax(cpr[np.isfinite(cpr)]) if np.any(np.isfinite(cpr)) else 0.0, neginf=0.0)
            # Clip for visualisation
            cpr_display = np.clip(cpr, 0.0, 10.0)
            im2 = ax2.imshow(cpr_display, cmap='plasma', vmin=0, vmax=10)
            ax2.set_title('Circular Polarisation Ratio (RH/LH)')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        self._draw_plot(fig2, 'aolp_canvas', row=1, column=0)

        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.hist(DoLP.flatten(), bins=100, alpha=0.7)
        ax3.set_title('Histogram')
        ax3.set_xlabel('DoLP' if pol_type == 'linear' else 'DoCP')
        ax3.set_ylabel('Frequency')
        if pol_type == 'linear':
            ax3.set_xlim(0, 1)
        else:
            ax3.set_xlim(-1, 1)
        self._draw_plot(fig3, 'hist_canvas', row=1, column=1)

    def update_batch_plots(self):
        for canvas in self.batch_plot_canvases.values():
            if canvas is not None:
                canvas.get_tk_widget().destroy()
        self.batch_plot_canvases.clear()

        if not hasattr(self.processor, 'batch_results') or not self.processor.batch_results:
            return

        results = self.processor.batch_results
        pol_type = self.batch_pol_var.get()
        
        azimuths = np.array([float(r['params']['az']) for r in results], dtype=float)
        zeniths = np.array([float(r['params']['ze']) for r in results], dtype=float)
        theta = np.deg2rad(azimuths)
        radii = zeniths

        # Mirror azimuths around 180°
        # Example:
        # 0   -> 180
        # 30  -> 210
        # 90  -> 270
        # 150 -> 330
        mirrored_azimuths = 360 - azimuths

        # Duplicate zenith values for mirrored hemisphere
        mirrored_zeniths = zeniths.copy()

        # Combine original + mirrored data
        azimuths_full = np.concatenate([azimuths, mirrored_azimuths])
        zeniths_full  = np.concatenate([zeniths, mirrored_zeniths])

        # Polar plotting variables
        theta = np.deg2rad(azimuths_full)
        radii = zeniths_full



        if pol_type == 'linear':
            dolp_vals = np.array([(float(r['pol'][0])*100) for r in results], dtype=float)
            dolp_vals = self.mirror_array(dolp_vals)

            aolp_vals = np.array([float(r['pol'][1]) for r in results], dtype=float)
            aolp_vals = self.mirror_array(aolp_vals)

            dolp_stds = np.array([float(r['dolp_std']) for r in results], dtype=float)
            dolp_stds = self.mirror_array(dolp_stds)

            aolp_stds = np.array([float(r['aolp_std']) for r in results], dtype=float)
            aolp_stds = self.mirror_array(aolp_stds)

            sat_vals = np.array([float(r['saturation']) for r in results], dtype=float)
            sat_vals = self.mirror_array(sat_vals)

            aolp_vals_norm = aolp_vals / np.pi
            
            # Row 0, Col 0: DoLP average
            vmin_val = float(np.nanmin(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 0.0
            vmax_val = float(np.nanmax(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 1.0
            cmap_name = 'viridis'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['dolp_avg'] = (theta, radii, dolp_vals, 'DoLP image average', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,0)})
            fig1 = self._make_polar_scatter_fig(theta, radii, dolp_vals, 'viridis', 'DoLP image average', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig1, 'dolp_avg', row=1, column=0)

            # Row 0, Col 1: DoLP std dev
            vmin_val = float(np.nanmin(dolp_stds)) if np.any(np.isfinite(dolp_stds)) else 0.0
            vmax_val = float(np.nanmax(dolp_stds)) if np.any(np.isfinite(dolp_stds)) else 1.0
            cmap_name = 'Purples'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['dolp_std'] = (theta, radii, dolp_stds, 'DoLP image standard deviation', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,1)})
            fig2 = self._make_polar_scatter_fig(theta, radii, dolp_stds, 'Purples', 'DoLP image standard deviation', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig2, 'dolp_std', row=1, column=1)

            # Row 0, Col 2: Saturation %
            vmin_val = float(np.nanmin(sat_vals)) if np.any(np.isfinite(sat_vals)) else 0.0
            vmax_val = float(np.nanmax(sat_vals)) if np.any(np.isfinite(sat_vals)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['saturation'] = (theta, radii, sat_vals, 'Image saturation % (total 0-255)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,2)})
            fig3 = self._make_polar_scatter_fig(theta, radii, sat_vals, 'YlGn', 'Image saturation % (total 0-255)', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig3, 'saturation', row=1, column=2)

            # Row 1, Col 0: AoLP average
            vmin_val = float(np.nanmin(aolp_vals_norm)) if np.any(np.isfinite(aolp_vals_norm)) else 0.0
            vmax_val = float(np.nanmax(aolp_vals_norm)) if np.any(np.isfinite(aolp_vals_norm)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['aolp_avg'] = (theta, radii, aolp_vals_norm, 'AoLP image average', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,0)})
            fig4 = self._make_polar_scatter_fig(theta, radii, aolp_vals_norm, 'YlGn', 'AoLP image average', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig4, 'aolp_avg', row=2, column=0)

            # Row 1, Col 1: AoLP std dev
            vmin_val = float(np.nanmin(aolp_stds)) if np.any(np.isfinite(aolp_stds)) else 0.0
            vmax_val = float(np.nanmax(aolp_stds)) if np.any(np.isfinite(aolp_stds)) else 1.0
            cmap_name = 'Blues'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['aolp_std'] = (theta, radii, aolp_stds, 'AoLP image standard deviation', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,1)})
            fig5 = self._make_polar_scatter_fig(theta, radii, aolp_stds, 'Blues', 'AoLP image standard deviation', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig5, 'aolp_std', row=2, column=1)

            # Row 1, Col 2: DoLP distribution
            vmin_val = float(np.nanmin(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 0.0
            vmax_val = float(np.nanmax(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 1.0
            cmap_name = 'cool'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['dolp_dist'] = (theta, radii, dolp_vals, 'DoLP Distribution', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,2)})
            fig6 = self._make_polar_scatter_fig(theta, radii, dolp_vals, 'cool', 'DoLP Distribution', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig6, 'dolp_dist', row=2, column=2)

        else:
            docp_vals = np.array([float(r['pol'][0]) for r in results], dtype=float)
            docp_vals = self.mirror_array(docp_vals)

            docp_stds = np.array([float(r['dolp_std']) for r in results], dtype=float)
            docp_stds = self.mirror_array(docp_stds)

            sat_vals = np.array([float(r['saturation']) for r in results], dtype=float)
            sat_vals = self.mirror_array(sat_vals)
            
            # Row 0, Col 0: DoCP average
            vmin_val = float(np.nanmin(docp_vals)) if np.any(np.isfinite(docp_vals)) else -1.0
            vmax_val = float(np.nanmax(docp_vals)) if np.any(np.isfinite(docp_vals)) else 1.0
            cmap_name = 'RdBu'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['docp_avg'] = (theta, radii, docp_vals, 'DoCP image average', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,0)})
            fig1 = self._make_polar_scatter_fig(theta, radii, docp_vals, 'RdBu', 'DoCP image average', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig1, 'docp_avg', row=1, column=0)

            # Row 0, Col 1: DoCP std dev
            vmin_val = float(np.nanmin(docp_stds)) if np.any(np.isfinite(docp_stds)) else 0.0
            vmax_val = float(np.nanmax(docp_stds)) if np.any(np.isfinite(docp_stds)) else 1.0
            cmap_name = 'Purples'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['docp_std'] = (theta, radii, docp_stds, 'DoCP image standard deviation', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,1)})
            fig2 = self._make_polar_scatter_fig(theta, radii, docp_stds, 'Purples', 'DoCP image standard deviation', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig2, 'docp_std', row=1, column=1)

            # Row 0, Col 2: Saturation %
            vmin_val = float(np.nanmin(sat_vals)) if np.any(np.isfinite(sat_vals)) else 0.0
            vmax_val = float(np.nanmax(sat_vals)) if np.any(np.isfinite(sat_vals)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['saturation'] = (theta, radii, sat_vals, 'Image saturation % (total 0-255)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,2)})
            fig3 = self._make_polar_scatter_fig(theta, radii, sat_vals, 'YlGn', 'Image saturation % (total 0-255)', vmin=vmin_val, vmax=vmax_val)
            self._draw_batch_plot(fig3, 'saturation', row=1, column=2)

            # Row 1, Col 0: Circular Polarisation Ratio (RH/LH)
            # CPR = (1 + DoCP) / (1 - DoCP)
            with np.errstate(divide='ignore', invalid='ignore'):
                cpr_vals = (1.0 + np.array([float(r['pol'][0]) for r in results], dtype=float)) / (1.0 - np.array([float(r['pol'][0]) for r in results], dtype=float))
            cpr_vals = self.mirror_array(cpr_vals)
            finite_mask = np.isfinite(cpr_vals)
            if np.any(finite_mask):
                cpr_max = np.nanmax(cpr_vals[finite_mask])
            else:
                cpr_max = 1.0
            cpr_plot_vals = np.nan_to_num(cpr_vals, nan=0.0, posinf=cpr_max, neginf=0.0)
            cpr_plot_vals = np.clip(cpr_plot_vals, 0.0, max(10.0, cpr_max))

            fig4 = self._make_polar_scatter_fig(theta, radii, cpr_plot_vals, 'plasma', 'Circular Polarisation Ratio (RH/LH)')
            vmin_val = float(np.nanmin(cpr_plot_vals)) if np.any(np.isfinite(cpr_plot_vals)) else 0.0
            vmax_val = float(np.nanmax(cpr_plot_vals)) if np.any(np.isfinite(cpr_plot_vals)) else 1.0
            cmap_name = 'plasma'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.batch_plot_data['cpr'] = (theta, radii, cpr_plot_vals, 'Circular Polarisation Ratio (RH/LH)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,0)})
            self._draw_batch_plot(fig4, 'cpr', row=2, column=0)

    def _redraw_canvas(self, canvas):
        if canvas is None:
            return
        try:
            canvas.draw_idle()
            canvas.get_tk_widget().update_idletasks()
        except Exception:
            pass
    def _draw_plot(self, fig, attr_name, row, column):
        self._draw_plot_generic(fig, master_frame=self.grid_frame, canvas_store=None, figures_store=self.single_plot_figures, attr_name=attr_name, row=row, column=column)

    def _clear_plot_canvas(self, attr_name):
        canvas = getattr(self, attr_name, None)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            setattr(self, attr_name, None)

    def clear_plots(self):
        self._clear_plot_canvas('dolp_canvas')
        self._clear_plot_canvas('aolp_canvas')
        self._clear_plot_canvas('hist_canvas')

    def _draw_batch_plot(self, fig, attr_name, row, column):
        adjust = {'left': 0.15, 'right': 0.75, 'top': 0.80, 'bottom': 0.12, 'pad': 1.5}
        self._draw_plot_generic(fig, master_frame=self.batch_grid_frame, canvas_store=self.batch_plot_canvases, figures_store=self.batch_plot_figures, attr_name=attr_name, row=row, column=column, adjust_kwargs=adjust)

    def _clear_batch_plot_canvas(self, attr_name):
        canvas = self.batch_plot_canvases.get(attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            del self.batch_plot_canvases[attr_name]

    def clear_batch_plots(self):
        for canvas in self.batch_plot_canvases.values():
            if canvas is not None:
                canvas.get_tk_widget().destroy()
        self.batch_plot_canvases.clear()

    def clear_compare_plots(self):
        for canvas in self.compare_plot_canvases.values():
            if canvas is not None:
                canvas.get_tk_widget().destroy()
        self.compare_plot_canvases.clear()

    def _draw_compare_plot(self, fig, attr_name, row, column):
        adjust = {'left': 0.15, 'right': 0.75, 'top': 0.80, 'bottom': 0.12, 'pad': 1.5}
        self._draw_plot_generic(fig, master_frame=self.compare_grid_frame, canvas_store=self.compare_plot_canvases, figures_store=self.compare_plot_figures, attr_name=attr_name, row=row, column=column, adjust_kwargs=adjust)

    def on_plot_click(self, attr_name, store):
        # Determine whether this is a batch or compare plot
        page = None
        if store is self.batch_plot_canvases:
            page = 'batch'
        elif store is self.compare_plot_canvases:
            page = 'compare'
        else:
            return
        self.open_plot_isolation(attr_name, page)

    def open_plot_isolation(self, attr_name, page):
        # Avoid opening multiple windows for same plot
        key = f"{page}:{attr_name}"
        if key in self._isolated_windows:
            try:
                self._isolated_windows[key].lift()
                return
            except Exception:
                pass

        data_store = self.batch_plot_data if page == 'batch' else self.compare_plot_data
        fig_store = self.batch_plot_figures if page == 'batch' else self.compare_plot_figures

        if attr_name not in data_store:
            return
        theta, radii, values, title, meta = data_store[attr_name]

        # initial vmin/vmax from metadata
        vmin = float(meta.get('vmin', float(np.nanmin(values)) if np.any(np.isfinite(values)) else 0.0))
        vmax = float(meta.get('vmax', float(np.nanmax(values)) if np.any(np.isfinite(values)) else 1.0))
        low_init = meta.get('low_color', '#0000ff')
        high_init = meta.get('high_color', '#ff0000')

        win = tk.Toplevel(self)
        win.title(f"Isolated Plot: {attr_name}")
        win.geometry('800x640')

        # create a frame to hold the plot so we can replace its children cleanly
        plot_frame = tk.Frame(win)
        plot_frame.pack(fill=tk.BOTH, expand=True)

        cmap_name = meta.get('cmap', 'viridis')
        try:
            original_cmap = self._get_plot_colormap(cmap_name)
            middle_init = meta.get('middle_color', mcolors.to_hex(original_cmap(0.5)))
            cmap_obj = original_cmap
        except Exception:
            middle_init = meta.get('middle_color', '#ffffff')
            cmap_obj = self._make_custom_colormap(low_init, middle_init, high_init)

        fig = self._make_polar_scatter_fig(theta, radii, values, cmap=cmap_obj, title=title, vmin=vmin, vmax=vmax)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(win)
        controls_frame.pack(fill=tk.X, padx=6, pady=6)

        # Color pickers
        low_color_var = tk.StringVar(value=low_init)
        middle_color_var = tk.StringVar(value=middle_init)
        high_color_var = tk.StringVar(value=high_init)

        def pick_low():
            c = colorchooser.askcolor(color=low_color_var.get(), parent=win)
            if c and c[1]:
                low_color_var.set(c[1])
                try:
                    range_slider.update_colors(low_color_var.get(), middle_color_var.get(), high_color_var.get())
                except Exception:
                    pass

        def pick_middle():
            c = colorchooser.askcolor(color=middle_color_var.get(), parent=win)
            if c and c[1]:
                middle_color_var.set(c[1])
                try:
                    range_slider.update_colors(low_color_var.get(), middle_color_var.get(), high_color_var.get())
                except Exception:
                    pass

        def pick_high():
            c = colorchooser.askcolor(color=high_color_var.get(), parent=win)
            if c and c[1]:
                high_color_var.set(c[1])
                try:
                    range_slider.update_colors(low_color_var.get(), middle_color_var.get(), high_color_var.get())
                except Exception:
                    pass

        tk.Button(controls_frame, text='Pick Low Color', command=pick_low).grid(row=0, column=0, padx=4)
        tk.Label(controls_frame, textvariable=low_color_var).grid(row=0, column=1, padx=4)
        tk.Button(controls_frame, text='Pick Middle Color', command=pick_middle).grid(row=0, column=2, padx=4)
        tk.Label(controls_frame, textvariable=middle_color_var).grid(row=0, column=3, padx=4)
        tk.Button(controls_frame, text='Pick High Color', command=pick_high).grid(row=0, column=4, padx=4)
        tk.Label(controls_frame, textvariable=high_color_var).grid(row=0, column=5, padx=4)

        # vmin/vmax dual-handle gradient slider (more user-friendly)
        class RangeSlider(tk.Frame):
            def __init__(self, master, data_min, data_max, low_color, middle_color, high_color, width=360, height=28, on_change=None, **kwargs):
                super().__init__(master, **kwargs)
                self.width = width
                self.height = height
                self.data_min = float(data_min)
                self.data_max = float(data_max)
                self.low_color = low_color
                self.middle_color = middle_color
                self.high_color = high_color
                self.on_change = on_change
                self.canvas = tk.Canvas(self, width=self.width, height=self.height)
                self.canvas.pack(side=tk.TOP, fill=tk.X, expand=False)
                self.label_frame = tk.Frame(self)
                self.label_frame.pack(side=tk.TOP, fill=tk.X)
                self.min_label = tk.Label(self.label_frame, text=f"{self.data_min:.3g}")
                self.max_label = tk.Label(self.label_frame, text=f"{self.data_max:.3g}")
                self.min_label.pack(side=tk.LEFT)
                self.max_label.pack(side=tk.RIGHT)

                # slider state: positions in pixels
                self.left_x = 4
                self.right_x = self.width - 4

                self._draw_gradient()
                self.left_handle = self.canvas.create_oval(self.left_x - 6, self.height/2 - 8, self.left_x + 6, self.height/2 + 8, fill='white', outline='black')
                self.right_handle = self.canvas.create_oval(self.right_x - 6, self.height/2 - 8, self.right_x + 6, self.height/2 + 8, fill='white', outline='black')

                self._drag_data = {'which': None}
                self.canvas.tag_bind(self.left_handle, '<ButtonPress-1>', lambda e: self._start_drag('left', e))
                self.canvas.tag_bind(self.right_handle, '<ButtonPress-1>', lambda e: self._start_drag('right', e))
                self.canvas.bind('<B1-Motion>', self._drag)
                self.canvas.bind('<ButtonRelease-1>', lambda e: self._end_drag())

            def _draw_gradient(self):
                self.canvas.delete('grad')
                steps = 180
                low_rgb = np.array(mcolors.to_rgb(self.low_color), dtype=float)
                middle_rgb = np.array(mcolors.to_rgb(self.middle_color), dtype=float)
                high_rgb = np.array(mcolors.to_rgb(self.high_color), dtype=float)
                for i in range(steps):
                    t = i / (steps - 1)
                    if t <= 0.5:
                        local_t = t * 2.0
                        col_rgb = low_rgb * (1.0 - local_t) + middle_rgb * local_t
                    else:
                        local_t = (t - 0.5) * 2.0
                        col_rgb = middle_rgb * (1.0 - local_t) + high_rgb * local_t
                    col = mcolors.to_hex(col_rgb)
                    x1 = int(2 + i * (self.width - 4) / steps)
                    x2 = int(2 + (i + 1) * (self.width - 4) / steps)
                    self.canvas.create_rectangle(x1, 4, x2, self.height - 4, fill=col, outline=col, tags='grad')
                # ensure gradient sits below handles so handles remain visible after redraw
                try:
                    self.canvas.tag_lower('grad')
                except Exception:
                    pass

            def _start_drag(self, which, event):
                self._drag_data['which'] = which

            def _drag(self, event):
                which = self._drag_data.get('which')
                if not which:
                    return
                x = min(max(4, event.x), self.width - 4)
                if which == 'left':
                    # prevent crossing
                    x = min(x, self.right_x - 12)
                    self.left_x = x
                    self.canvas.coords(self.left_handle, x - 6, self.height/2 - 8, x + 6, self.height/2 + 8)
                else:
                    x = max(x, self.left_x + 12)
                    self.right_x = x
                    self.canvas.coords(self.right_handle, x - 6, self.height/2 - 8, x + 6, self.height/2 + 8)
                self._update_labels()

            def _end_drag(self):
                self._drag_data['which'] = None

            def _update_labels(self):
                vmin_val, vmax_val = self.get()
                self.min_label.config(text=f"{vmin_val:.3g}")
                self.max_label.config(text=f"{vmax_val:.3g}")
                if hasattr(self, 'on_change') and callable(self.on_change):
                    try:
                        self.on_change(vmin_val, vmax_val)
                    except Exception:
                        pass

            def set_range(self, vmin_val, vmax_val):
                # clamp
                vmin_val = max(self.data_min, min(self.data_max, float(vmin_val)))
                vmax_val = max(self.data_min, min(self.data_max, float(vmax_val)))
                if self.data_max == self.data_min:
                    self.left_x = self.right_x = (self.width - 1) // 2
                    self.canvas.coords(self.left_handle, self.left_x - 6, self.height/2 - 8, self.left_x + 6, self.height/2 + 8)
                    self.canvas.coords(self.right_handle, self.right_x - 6, self.height/2 - 8, self.right_x + 6, self.height/2 + 8)
                    self._update_labels()
                    return
                if vmax_val <= vmin_val:
                    vmax_val = min(self.data_max, vmin_val + 1e-9)
                left_rel = (vmin_val - self.data_min) / (self.data_max - self.data_min)
                right_rel = (vmax_val - self.data_min) / (self.data_max - self.data_min)
                self.left_x = int(4 + left_rel * (self.width - 8))
                self.right_x = int(4 + right_rel * (self.width - 8))
                self.canvas.coords(self.left_handle, self.left_x - 6, self.height/2 - 8, self.left_x + 6, self.height/2 + 8)
                self.canvas.coords(self.right_handle, self.right_x - 6, self.height/2 - 8, self.right_x + 6, self.height/2 + 8)
                self._update_labels()

            def get(self):
                # map pixel positions to data range
                if self.data_max == self.data_min:
                    return (self.data_min, self.data_max)
                left_rel = (self.left_x - 4) / (self.width - 8)
                right_rel = (self.right_x - 4) / (self.width - 8)
                vmin_val = self.data_min + left_rel * (self.data_max - self.data_min)
                vmax_val = self.data_min + right_rel * (self.data_max - self.data_min)
                return (vmin_val, vmax_val)

            def update_colors(self, low_color, middle_color, high_color):
                self.low_color = low_color
                self.middle_color = middle_color
                self.high_color = high_color
                self._draw_gradient()

        finite_values = values[np.isfinite(values)]
        slider_min = float(np.min(finite_values)) if finite_values.size else 0.0
        slider_max = float(np.max(finite_values)) if finite_values.size else 1.0
        range_slider = RangeSlider(controls_frame, data_min=slider_min, data_max=slider_max, low_color=low_color_var.get(), middle_color=middle_color_var.get(), high_color=high_color_var.get())
        # set initial handles to reflect current vmin/vmax
        try:
            range_slider.set_range(vmin, vmax)
        except Exception:
            pass
        range_slider.grid(row=1, column=0, columnspan=6, pady=6)
        # Numeric entries can intentionally extend beyond the data range.
        span = slider_max - slider_min if slider_max - slider_min != 0 else 1.0
        step = span / 500.0 if span else 1.0
        min_var = tk.DoubleVar(value=vmin)
        max_var = tk.DoubleVar(value=vmax)

        min_box = tk.Entry(controls_frame, textvariable=min_var, width=12)
        max_box = tk.Entry(controls_frame, textvariable=max_var, width=12)
        tk.Label(controls_frame, text='Global min:').grid(row=2, column=0, sticky='e')
        min_box.grid(row=2, column=1, sticky='w', padx=4)
        tk.Label(controls_frame, text='Global max:').grid(row=2, column=2, sticky='e')
        max_box.grid(row=2, column=3, sticky='w', padx=4)

        def boxes_to_slider(event=None):
            try:
                lv = float(min_var.get())
                hv = float(max_var.get())
                if hv <= lv:
                    return
                if slider_min <= lv <= slider_max and slider_min <= hv <= slider_max:
                    range_slider.set_range(lv, hv)
            except Exception:
                pass

        def get_typed_range():
            lv = float(min_var.get())
            hv = float(max_var.get())
            if hv <= lv:
                raise ValueError('Maximum must be greater than minimum.')
            return lv, hv

        def slider_to_boxes(lv, hv):
            try:
                min_var.set(lv)
                max_var.set(hv)
            except Exception:
                pass

        min_box.bind('<KeyRelease>', boxes_to_slider)
        min_box.bind('<FocusOut>', boxes_to_slider)
        max_box.bind('<KeyRelease>', boxes_to_slider)
        max_box.bind('<FocusOut>', boxes_to_slider)
        range_slider.on_change = slider_to_boxes

        # Re-pack the initial canvas after controls are added so layout accounts for control height
        try:
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack_forget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
        except Exception:
            pass

        def apply_changes():
            low = low_color_var.get()
            middle = middle_color_var.get()
            high = high_color_var.get()
            try:
                vmin_new, vmax_new = get_typed_range()
            except ValueError:
                return
            # build colormap
            cmap = self._make_custom_colormap(low, middle, high)
            # update gradient on slider to reflect new colors
            try:
                range_slider.update_colors(low, middle, high)
            except Exception:
                pass

            # create new fig and update isolated canvas in-place
            new_fig = self._make_polar_scatter_fig(theta, radii, values, cmap=cmap, title=title, vmin=vmin_new, vmax=vmax_new)
            try:
                # destroy existing contents of the plot frame to avoid residual images
                for child in plot_frame.winfo_children():
                    child.destroy()
            except Exception:
                pass
            new_canvas = FigureCanvasTkAgg(new_fig, master=plot_frame)
            new_canvas.draw()
            new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            try:
                plot_frame.update_idletasks()
                w = plot_frame.winfo_width()
                h = plot_frame.winfo_height()
                new_canvas.get_tk_widget().config(width=w, height=h)
            except Exception:
                pass

            # persist metadata
            meta_new = {'low_color': low, 'middle_color': middle, 'high_color': high, 'vmin': float(vmin_new), 'vmax': float(vmax_new), 'cmap': 'custom', 'pos': meta.get('pos', (1,0))}
            data_store[attr_name] = (theta, radii, values, title, meta_new)

            # update stored figure and redraw in main UI without stealing focus
            fig_store[attr_name] = new_fig
            if page == 'batch':
                try:
                    self._clear_batch_plot_canvas(attr_name)
                except Exception:
                    pass
                try:
                    r,c = meta_new.get('pos', (1,0))
                    self._draw_batch_plot(new_fig, attr_name, row=r, column=c)
                except Exception:
                    pass
            else:
                try:
                    existing = self.compare_plot_canvases.get(attr_name)
                    if existing is not None:
                        existing.get_tk_widget().destroy()
                        del self.compare_plot_canvases[attr_name]
                except Exception:
                    pass
                try:
                    r,c = meta_new.get('pos', (1,0))
                    self._draw_compare_plot(new_fig, attr_name, row=r, column=c)
                except Exception:
                    pass

            # keep isolation window focused so user can continue adjusting
            try:
                win.focus_force()
                controls_frame.focus_set()
            except Exception:
                pass

        tk.Button(controls_frame, text='Apply', command=apply_changes).grid(row=4, column=0, columnspan=4, pady=6)

        def persist_and_close():
            # persist current slider/colors to main UI even if user didn't click Apply
            try:
                low = low_color_var.get()
                middle = middle_color_var.get()
                high = high_color_var.get()
                vmin_new, vmax_new = get_typed_range()
                cmap = self._make_custom_colormap(low, middle, high)
                new_fig = self._make_polar_scatter_fig(theta, radii, values, cmap=cmap, title=title, vmin=vmin_new, vmax=vmax_new)
                meta_new = {'low_color': low, 'middle_color': middle, 'high_color': high, 'vmin': float(vmin_new), 'vmax': float(vmax_new), 'cmap': 'custom', 'pos': meta.get('pos', (1,0))}
                data_store[attr_name] = (theta, radii, values, title, meta_new)
                fig_store[attr_name] = new_fig
                if page == 'batch':
                    try:
                        self._clear_batch_plot_canvas(attr_name)
                    except Exception:
                        pass
                    try:
                        r,c = meta_new.get('pos', (1,0))
                        self._draw_batch_plot(new_fig, attr_name, row=r, column=c)
                    except Exception:
                        pass
                else:
                    try:
                        existing = self.compare_plot_canvases.get(attr_name)
                        if existing is not None:
                            existing.get_tk_widget().destroy()
                            del self.compare_plot_canvases[attr_name]
                    except Exception:
                        pass
                    try:
                        r,c = meta_new.get('pos', (1,0))
                        self._draw_compare_plot(new_fig, attr_name, row=r, column=c)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                del self._isolated_windows[key]
            except Exception:
                pass
            win.destroy()

        win.protocol('WM_DELETE_WINDOW', persist_and_close)
        self._isolated_windows[key] = win

    def update_compare_plots(self):
        self.clear_compare_plots()
        if not self.compare_results:
            return

        pol_type = self.compare_pol_var.get()
        azimuths = np.array([float(r['params']['az']) for r in self.compare_results], dtype=float)
        zeniths = np.array([float(r['params']['ze']) for r in self.compare_results], dtype=float)
        mirrored_azimuths = 360 - azimuths
        azimuths_full = np.concatenate([azimuths, mirrored_azimuths])
        zeniths_full = np.concatenate([zeniths, zeniths])
        theta = np.deg2rad(azimuths_full)
        radii = zeniths_full

        if pol_type == 'linear':
            dolp_diffs = np.array([r['dolp_diff'] for r in self.compare_results], dtype=float)
            dolp_std_diffs = np.array([r['dolp_std_diff'] for r in self.compare_results], dtype=float)
            aolp_diffs = np.array([r['aolp_diff'] for r in self.compare_results], dtype=float)
            aolp_std_diffs = np.array([r['aolp_std_diff'] for r in self.compare_results], dtype=float)
            sat_diffs = np.array([r['sat_diff'] for r in self.compare_results], dtype=float)

            dolp_vals = self.mirror_array(dolp_diffs)
            dolp_std_vals = self.mirror_array(dolp_std_diffs)
            aolp_vals = self.mirror_array(aolp_diffs)
            aolp_std_vals = self.mirror_array(aolp_std_diffs)
            sat_vals = self.mirror_array(sat_diffs)

            vmin_val = float(np.nanmin(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else -1.0
            vmax_val = float(np.nanmax(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 1.0
            cmap_name = 'RdBu'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['dolp_diff'] = (theta, radii, dolp_vals, 'DoLP difference (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,0)})
            fig1 = self._make_polar_scatter_fig(theta, radii, dolp_vals, 'RdBu', 'DoLP difference (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig1, 'dolp_diff', row=1, column=0)

            vmin_val = float(np.nanmin(dolp_std_vals)) if np.any(np.isfinite(dolp_std_vals)) else 0.0
            vmax_val = float(np.nanmax(dolp_std_vals)) if np.any(np.isfinite(dolp_std_vals)) else 1.0
            cmap_name = 'Purples'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['dolp_std_diff'] = (theta, radii, dolp_std_vals, 'DoLP std diff (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,1)})
            fig2 = self._make_polar_scatter_fig(theta, radii, dolp_std_vals, 'Purples', 'DoLP std diff (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig2, 'dolp_std_diff', row=1, column=1)

            vmin_val = float(np.nanmin(sat_vals)) if np.any(np.isfinite(sat_vals)) else 0.0
            vmax_val = float(np.nanmax(sat_vals)) if np.any(np.isfinite(sat_vals)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['sat_diff'] = (theta, radii, sat_vals, 'Saturation diff (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,2)})
            fig3 = self._make_polar_scatter_fig(theta, radii, sat_vals, 'YlGn', 'Saturation diff (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig3, 'sat_diff', row=1, column=2)

            vmin_val = float(np.nanmin(aolp_vals)) if np.any(np.isfinite(aolp_vals)) else 0.0
            vmax_val = float(np.nanmax(aolp_vals)) if np.any(np.isfinite(aolp_vals)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['aolp_diff'] = (theta, radii, aolp_vals, 'AoLP difference (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,0)})
            fig4 = self._make_polar_scatter_fig(theta, radii, aolp_vals, 'YlGn', 'AoLP difference (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig4, 'aolp_diff', row=2, column=0)

            vmin_val = float(np.nanmin(aolp_std_vals)) if np.any(np.isfinite(aolp_std_vals)) else 0.0
            vmax_val = float(np.nanmax(aolp_std_vals)) if np.any(np.isfinite(aolp_std_vals)) else 1.0
            cmap_name = 'Blues'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['aolp_std_diff'] = (theta, radii, aolp_std_vals, 'AoLP std diff (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,1)})
            fig5 = self._make_polar_scatter_fig(theta, radii, aolp_std_vals, 'Blues', 'AoLP std diff (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig5, 'aolp_std_diff', row=2, column=1)

            vmin_val = float(np.nanmin(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 0.0
            vmax_val = float(np.nanmax(dolp_vals)) if np.any(np.isfinite(dolp_vals)) else 1.0
            cmap_name = 'cool'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['dolp_diff_dist'] = (theta, radii, dolp_vals, 'DoLP diff distribution', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,2)})
            fig6 = self._make_polar_scatter_fig(theta, radii, dolp_vals, 'cool', 'DoLP diff distribution', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig6, 'dolp_diff_dist', row=2, column=2)

        else:
            docp_diffs = np.array([r['docp_diff'] for r in self.compare_results], dtype=float)
            docp_std_diffs = np.array([r['docp_std_diff'] for r in self.compare_results], dtype=float)
            sat_diffs = np.array([r['sat_diff'] for r in self.compare_results], dtype=float)

            docp_vals = self.mirror_array(docp_diffs)
            docp_std_vals = self.mirror_array(docp_std_diffs)
            sat_vals = self.mirror_array(sat_diffs)

            vmin_val = float(np.nanmin(docp_vals)) if np.any(np.isfinite(docp_vals)) else -1.0
            vmax_val = float(np.nanmax(docp_vals)) if np.any(np.isfinite(docp_vals)) else 1.0
            cmap_name = 'RdBu'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['docp_diff'] = (theta, radii, docp_vals, 'DoCP difference (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,0)})
            fig1 = self._make_polar_scatter_fig(theta, radii, docp_vals, 'RdBu', 'DoCP difference (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig1, 'docp_diff', row=1, column=0)

            vmin_val = float(np.nanmin(docp_std_vals)) if np.any(np.isfinite(docp_std_vals)) else 0.0
            vmax_val = float(np.nanmax(docp_std_vals)) if np.any(np.isfinite(docp_std_vals)) else 1.0
            cmap_name = 'Purples'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['docp_std_diff'] = (theta, radii, docp_std_vals, 'DoCP std diff (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,1)})
            fig2 = self._make_polar_scatter_fig(theta, radii, docp_std_vals, 'Purples', 'DoCP std diff (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig2, 'docp_std_diff', row=1, column=1)

            vmin_val = float(np.nanmin(sat_vals)) if np.any(np.isfinite(sat_vals)) else 0.0
            vmax_val = float(np.nanmax(sat_vals)) if np.any(np.isfinite(sat_vals)) else 1.0
            cmap_name = 'YlGn'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['sat_diff'] = (theta, radii, sat_vals, 'Saturation diff (1 - 2)', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (1,2)})
            fig3 = self._make_polar_scatter_fig(theta, radii, sat_vals, 'YlGn', 'Saturation diff (1 - 2)', vmin=vmin_val, vmax=vmax_val)
            self._draw_compare_plot(fig3, 'sat_diff', row=1, column=2)

            # Circular Polarisation Ratio difference for compare
            with np.errstate(divide='ignore', invalid='ignore'):
                cpr1 = (1.0 + np.array([r['docp_diff'] for r in self.compare_results], dtype=float)) / (1.0 - np.array([r['docp_diff'] for r in self.compare_results], dtype=float))
                cpr2 = np.nan_to_num(cpr1, nan=0.0, posinf=np.nanmax(cpr1[np.isfinite(cpr1)]) if np.any(np.isfinite(cpr1)) else 0.0, neginf=0.0)
            cpr_vals = self.mirror_array(cpr2)
            finite_mask = np.isfinite(cpr_vals)
            if np.any(finite_mask):
                cpr_max = np.nanmax(cpr_vals[finite_mask])
            else:
                cpr_max = 1.0
            cpr_plot_vals = np.clip(np.nan_to_num(cpr_vals, nan=0.0, posinf=cpr_max, neginf=0.0), 0.0, max(10.0, cpr_max))

            fig4 = self._make_polar_scatter_fig(theta, radii, cpr_plot_vals, 'plasma', 'Circular Polarisation Ratio diff')
            vmin_val = float(np.nanmin(cpr_plot_vals)) if np.any(np.isfinite(cpr_plot_vals)) else 0.0
            vmax_val = float(np.nanmax(cpr_plot_vals)) if np.any(np.isfinite(cpr_plot_vals)) else 1.0
            cmap_name = 'plasma'
            lowc = mcolors.to_hex(plt.get_cmap(cmap_name)(0.0))
            highc = mcolors.to_hex(plt.get_cmap(cmap_name)(1.0))
            self.compare_plot_data['cpr_diff'] = (theta, radii, cpr_plot_vals, 'Circular Polarisation Ratio diff', {'low_color': lowc, 'high_color': highc, 'vmin': vmin_val, 'vmax': vmax_val, 'cmap': cmap_name, 'pos': (2,0)})
            self._draw_compare_plot(fig4, 'cpr_diff', row=2, column=0)

    def _export_plot_figures_png(self, plot_figures, page_name):
        self.geometry(self.default_geometry)
        self.update_idletasks()

        if not plot_figures:
            self.status_label.config(text=f"No {page_name} graphs to export.")
            return

        folder = filedialog.askdirectory(title=f"Select folder to save {page_name} PNG files")
        if not folder:
            return


        all_canvases = list(self.batch_plot_canvases.values()) + list(self.compare_plot_canvases.values())
        all_canvases += [getattr(self, 'dolp_canvas', None), getattr(self, 'aolp_canvas', None), getattr(self, 'hist_canvas', None)]
        for canvas in all_canvases:
            self._redraw_canvas(canvas)

        try:
            for name, fig in plot_figures.items():
                if fig is None:
                    continue
                fig.tight_layout(pad=1.5)
                fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.12)
                safe_name = name.replace(' ', '_')
                output_path = os.path.join(folder, f"{page_name}_{safe_name}.png")
                fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        finally:
            self.update_idletasks()

        self.status_label.config(text=f"Exported {len(plot_figures)} {page_name} graph(s) to PNG.")

    def export_single_png(self):
        self._export_plot_figures_png(self.single_plot_figures, 'single')

    def export_batch_png(self):
        self._export_plot_figures_png(self.batch_plot_figures, 'batch')

    def export_compare_png(self):
        self._export_plot_figures_png(self.compare_plot_figures, 'compare')

    def export_csv(self):
        if self.processor.pol_params is None:
            self.status_label.config(text="Calculate polarization first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.processor.export_csv(file_path)
        self.status_label.config(text="Exported to CSV.")

    def export_compare_csv(self):
        if not self.compare_results:
            self.status_label.config(text="No comparison results to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        csv_base = file_path[:-4] if file_path.lower().endswith('.csv') else file_path
        with open(f"{csv_base}.csv", mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if self.compare_pol_var.get() == 'linear':
                writer.writerow(['file', 'az', 'ze', 'cze', 'exp', 'ISO', 'obs', 'DoLP_diff', 'AoLP_diff_deg', 'Sat_diff'])
                for row in self.compare_results:
                    params = row['params']
                    writer.writerow([
                        params.get('file', ''), params['az'], params['ze'], params['cze'], params['exp'], params['ISO'], params['obs'],
                        row['dolp_diff'], np.rad2deg(row['aolp_diff']), row['sat_diff']
                    ])
            else:
                writer.writerow(['file', 'az', 'ze', 'cze', 'exp', 'ISO', 'obs', 'DoCP_diff', 'Sat_diff'])
                for row in self.compare_results:
                    params = row['params']
                    writer.writerow([
                        params.get('file', ''), params['az'], params['ze'], params['cze'], params['exp'], params['ISO'], params['obs'],
                        row['docp_diff'], row['sat_diff']
                    ])
        self.status_label.config(text="Comparison results exported to CSV.")

    def export_batch_csv(self):
        if not hasattr(self.processor, 'batch_results') or not self.processor.batch_results:
            self.status_label.config(text="No batch results to export.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        success = self.processor.export_batch_csv(file_path)
        if success:
            self.status_label.config(text="Batch results exported to CSV.")
        else:
            self.status_label.config(text="Failed to export batch CSV.")

    def mirror_array(self, array):
        mirrored_array = array.copy()
        array = np.concatenate([array, mirrored_array])
        return array


def run_gui():
    app = PolarizationGUI()
    app.mainloop()


if __name__ == '__main__':
    run_gui()
