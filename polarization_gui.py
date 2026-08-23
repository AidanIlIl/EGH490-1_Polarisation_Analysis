import os
import csv
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
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
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.processor.original_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        roi_x1 = max(0, min(img_width, (x1 - x_offset) / scale))
        roi_y1 = max(0, min(img_height, (y1 - y_offset) / scale))
        roi_x2 = max(0, min(img_width, (x2 - x_offset) / scale))
        roi_y2 = max(0, min(img_height, (y2 - y_offset) / scale))
        return (min(roi_x1, roi_x2), min(roi_y1, roi_y2), abs(roi_x2 - roi_x1), abs(roi_y2 - roi_y1))

    def _batch_canvas_to_image_roi(self, x1, y1, x2, y2):
        canvas_width = self.batch_canvas.winfo_width()
        canvas_height = self.batch_canvas.winfo_height()
        img_width, img_height = self.processor.original_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        roi_x1 = max(0, min(img_width, (x1 - x_offset) / scale))
        roi_y1 = max(0, min(img_height, (y1 - y_offset) / scale))
        roi_x2 = max(0, min(img_width, (x2 - x_offset) / scale))
        roi_y2 = max(0, min(img_height, (y2 - y_offset) / scale))
        return (min(roi_x1, roi_x2), min(roi_y1, roi_y2), abs(roi_x2 - roi_x1), abs(roi_y2 - roi_y1))

    def _compare_canvas_to_image_roi(self, x1, y1, x2, y2):
        canvas_width = self.compare_canvas.winfo_width()
        canvas_height = self.compare_canvas.winfo_height()
        img_height, img_width = self.compare_processor_image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        roi_x1 = max(0, min(img_width, (x1 - x_offset) / scale))
        roi_y1 = max(0, min(img_height, (y1 - y_offset) / scale))
        roi_x2 = max(0, min(img_width, (x2 - x_offset) / scale))
        roi_y2 = max(0, min(img_height, (y2 - y_offset) / scale))
        return (min(roi_x1, roi_x2), min(roi_y1, roi_y2), abs(roi_x2 - roi_x1), abs(roi_y2 - roi_y1))

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
        if self.processor.roi is None:
            return
        x, y, w, h = self.processor.roi
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.processor.original_img.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        x1 = int(x_offset + x * scale)
        y1 = int(y_offset + y * scale)
        x2 = int(x_offset + (x + w) * scale)
        y2 = int(y_offset + (y + h) * scale)
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
        self.roi_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def _batch_redraw_roi(self):
        if not hasattr(self.processor, 'batch_roi') or self.processor.batch_roi is None:
            return
        x, y, w, h = self.processor.batch_roi
        canvas_width = self.batch_canvas.winfo_width()
        canvas_height = self.batch_canvas.winfo_height()
        img_height, img_width = self.processor.batch_first_img.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        x1 = int(x_offset + x * scale)
        y1 = int(y_offset + y * scale)
        x2 = int(x_offset + (x + w) * scale)
        y2 = int(y_offset + (y + h) * scale)
        if self.batch_roi_rect:
            self.batch_canvas.delete(self.batch_roi_rect)
        self.batch_roi_rect = self.batch_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

    def _compare_redraw_roi(self):
        if not hasattr(self, 'compare_processor_image') or self.compare_processor_image is None or self.batch_processor1.batch_roi is None:
            return
        x, y, w, h = self.batch_processor1.batch_roi
        canvas_width = self.compare_canvas.winfo_width()
        canvas_height = self.compare_canvas.winfo_height()
        img_height, img_width = self.compare_processor_image.shape[:2]
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        x_offset = (canvas_width - new_width) // 2
        y_offset = (canvas_height - new_height) // 2
        x1 = int(x_offset + x * scale)
        y1 = int(y_offset + y * scale)
        x2 = int(x_offset + (x + w) * scale)
        y2 = int(y_offset + (y + h) * scale)
        if self.compare_roi_rect:
            self.compare_canvas.delete(self.compare_roi_rect)
        self.compare_roi_rect = self.compare_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)

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
                dolp_diff = float(r1['pol'][0]) - float(r2['pol'][0])
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
            dolp_vals = np.array([float(r['pol'][0]) for r in results], dtype=float)
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
            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter1 = ax1.scatter(theta, radii, c=dolp_vals, cmap='viridis', s=20, edgecolors='black', linewidth=0.1)
            ax1.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax1.set_title('DoLP image average', fontsize=10, pad=10)
            fig1.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig1, 'dolp_avg', row=1, column=0)

            # Row 0, Col 1: DoLP std dev
            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter2 = ax2.scatter(theta, radii, c=dolp_stds, cmap='Purples', s=20, edgecolors='black', linewidth=0.1)
            ax2.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax2.set_title('DoLP image standard deviation', fontsize=10, pad=10)
            fig2.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig2, 'dolp_std', row=1, column=1)

            # Row 0, Col 2: Saturation %
            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter3 = ax3.scatter(theta, radii, c=sat_vals, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax3.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax3.set_title('Image saturation % (total 0-255)', fontsize=10, pad=10)
            fig3.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig3, 'saturation', row=1, column=2)

            # Row 1, Col 0: AoLP average
            fig4, ax4 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter4 = ax4.scatter(theta, radii, c=aolp_vals_norm, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax4.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax4.set_title('AoLP image average', fontsize=10, pad=10)
            fig4.colorbar(scatter4, ax=ax4, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig4, 'aolp_avg', row=2, column=0)

            # Row 1, Col 1: AoLP std dev
            fig5, ax5 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter5 = ax5.scatter(theta, radii, c=aolp_stds, cmap='Blues', s=20, edgecolors='black', linewidth=0.1)
            ax5.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax5.set_title('AoLP image standard deviation', fontsize=10, pad=10)
            fig5.colorbar(scatter5, ax=ax5, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig5, 'aolp_std', row=2, column=1)

            # Row 1, Col 2: DoLP distribution
            fig6, ax6 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter6 = ax6.scatter(theta, radii, c=dolp_vals, cmap='cool', s=20, edgecolors='black', linewidth=0.1)
            ax6.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax6.set_title('DoLP Distribution', fontsize=10, pad=10)
            fig6.colorbar(scatter6, ax=ax6, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig6, 'dolp_dist', row=2, column=2)

        else:
            docp_vals = np.array([float(r['pol'][0]) for r in results], dtype=float)
            docp_vals = self.mirror_array(docp_vals)

            docp_stds = np.array([float(r['dolp_std']) for r in results], dtype=float)
            docp_stds = self.mirror_array(docp_stds)

            sat_vals = np.array([float(r['saturation']) for r in results], dtype=float)
            sat_vals = self.mirror_array(sat_vals)
            
            # Row 0, Col 0: DoCP average
            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter1 = ax1.scatter(theta, radii, c=docp_vals, cmap='RdBu', s=20, edgecolors='black', linewidth=0.1)
            ax1.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax1.set_title('DoCP image average', fontsize=10, pad=10)
            fig1.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig1, 'docp_avg', row=1, column=0)

            # Row 0, Col 1: DoCP std dev
            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter2 = ax2.scatter(theta, radii, c=docp_stds, cmap='Purples', s=20, edgecolors='black', linewidth=0.1)
            ax2.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax2.set_title('DoCP image standard deviation', fontsize=10, pad=10)
            fig2.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig2, 'docp_std', row=1, column=1)

            # Row 0, Col 2: Saturation %
            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter3 = ax3.scatter(theta, radii, c=sat_vals, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax3.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax3.set_title('Image saturation % (total 0-255)', fontsize=10, pad=10)
            fig3.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.08)
            self._draw_batch_plot(fig3, 'saturation', row=1, column=2)

            # Row 1, Col 0: Circular Polarisation Ratio (RH/LH)
            # CPR = (1 + DoCP) / (1 - DoCP)
            with np.errstate(divide='ignore', invalid='ignore'):
                cpr_vals = (1.0 + np.array([float(r['pol'][0]) for r in results], dtype=float)) / (1.0 - np.array([float(r['pol'][0]) for r in results], dtype=float))
            # mirror the CPR values for hemisphere plotting
            cpr_vals = self.mirror_array(cpr_vals)
            # replace infs/nans for plotting
            finite_mask = np.isfinite(cpr_vals)
            if np.any(finite_mask):
                cpr_max = np.nanmax(cpr_vals[finite_mask])
            else:
                cpr_max = 1.0
            cpr_plot_vals = np.nan_to_num(cpr_vals, nan=0.0, posinf=cpr_max, neginf=0.0)
            cpr_plot_vals = np.clip(cpr_plot_vals, 0.0, max(10.0, cpr_max))

            fig4, ax4 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter4 = ax4.scatter(theta, radii, c=cpr_plot_vals, cmap='plasma', s=20, edgecolors='black', linewidth=0.1)
            ax4.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax4.set_title('Circular Polarisation Ratio (RH/LH)', fontsize=10, pad=10)
            fig4.colorbar(scatter4, ax=ax4, fraction=0.046, pad=0.08)
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
        fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.12)
        canvas = getattr(self, attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)
        setattr(self, attr_name, canvas)
        self.single_plot_figures[attr_name] = fig
        self._redraw_canvas(canvas)
        plt.close(fig)

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
        fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.15, right=0.75, top=0.80, bottom=0.12)
        canvas = self.batch_plot_canvases.get(attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.batch_grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)
        self.batch_plot_canvases[attr_name] = canvas
        self.batch_plot_figures[attr_name] = fig
        self._redraw_canvas(canvas)
        plt.close(fig)

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
        fig.tight_layout(pad=1.5)
        fig.subplots_adjust(left=0.15, right=0.75, top=0.80, bottom=0.12)
        canvas = self.compare_plot_canvases.get(attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.compare_grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)
        self.compare_plot_canvases[attr_name] = canvas
        self.compare_plot_figures[attr_name] = fig
        self._redraw_canvas(canvas)
        plt.close(fig)

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

            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter1 = ax1.scatter(theta, radii, c=dolp_vals, cmap='RdBu', s=20, edgecolors='black', linewidth=0.1)
            ax1.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax1.set_title('DoLP difference (1 - 2)', fontsize=10, pad=10)
            fig1.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig1, 'dolp_diff', row=1, column=0)

            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter2 = ax2.scatter(theta, radii, c=dolp_std_vals, cmap='Purples', s=20, edgecolors='black', linewidth=0.1)
            ax2.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax2.set_title('DoLP std diff (1 - 2)', fontsize=10, pad=10)
            fig2.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig2, 'dolp_std_diff', row=1, column=1)

            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter3 = ax3.scatter(theta, radii, c=sat_vals, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax3.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax3.set_title('Saturation diff (1 - 2)', fontsize=10, pad=10)
            fig3.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig3, 'sat_diff', row=1, column=2)

            fig4, ax4 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter4 = ax4.scatter(theta, radii, c=aolp_vals, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax4.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax4.set_title('AoLP difference (1 - 2)', fontsize=10, pad=10)
            fig4.colorbar(scatter4, ax=ax4, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig4, 'aolp_diff', row=2, column=0)

            fig5, ax5 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter5 = ax5.scatter(theta, radii, c=aolp_std_vals, cmap='Blues', s=20, edgecolors='black', linewidth=0.1)
            ax5.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax5.set_title('AoLP std diff (1 - 2)', fontsize=10, pad=10)
            fig5.colorbar(scatter5, ax=ax5, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig5, 'aolp_std_diff', row=2, column=1)

            fig6, ax6 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter6 = ax6.scatter(theta, radii, c=dolp_vals, cmap='cool', s=20, edgecolors='black', linewidth=0.1)
            ax6.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax6.set_title('DoLP diff distribution', fontsize=10, pad=10)
            fig6.colorbar(scatter6, ax=ax6, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig6, 'dolp_diff_dist', row=2, column=2)

        else:
            docp_diffs = np.array([r['docp_diff'] for r in self.compare_results], dtype=float)
            docp_std_diffs = np.array([r['docp_std_diff'] for r in self.compare_results], dtype=float)
            sat_diffs = np.array([r['sat_diff'] for r in self.compare_results], dtype=float)

            docp_vals = self.mirror_array(docp_diffs)
            docp_std_vals = self.mirror_array(docp_std_diffs)
            sat_vals = self.mirror_array(sat_diffs)

            fig1, ax1 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter1 = ax1.scatter(theta, radii, c=docp_vals, cmap='RdBu', s=20, edgecolors='black', linewidth=0.1)
            ax1.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax1.set_title('DoCP difference (1 - 2)', fontsize=10, pad=10)
            fig1.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig1, 'docp_diff', row=1, column=0)

            fig2, ax2 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter2 = ax2.scatter(theta, radii, c=docp_std_vals, cmap='Purples', s=20, edgecolors='black', linewidth=0.1)
            ax2.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax2.set_title('DoCP std diff (1 - 2)', fontsize=10, pad=10)
            fig2.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.08)
            self._draw_compare_plot(fig2, 'docp_std_diff', row=1, column=1)

            fig3, ax3 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter3 = ax3.scatter(theta, radii, c=sat_vals, cmap='YlGn', s=20, edgecolors='black', linewidth=0.1)
            ax3.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax3.set_title('Saturation diff (1 - 2)', fontsize=10, pad=10)
            fig3.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.08)
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

            fig4, ax4 = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': 'polar'})
            scatter4 = ax4.scatter(theta, radii, c=cpr_plot_vals, cmap='plasma', s=20, edgecolors='black', linewidth=0.1)
            ax4.set_ylim(0, max(radii) * 1.1 if len(radii) > 0 else 90)
            ax4.set_title('Circular Polarisation Ratio diff', fontsize=10, pad=10)
            fig4.colorbar(scatter4, ax=ax4, fraction=0.046, pad=0.08)
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
