import os
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from analysis import PolarizationProcessor


class PolarizationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Polarization Analysis GUI")
        self.geometry("1200x850")

        self.processor = PolarizationProcessor()
        self.roi_start = None
        self.roi_rect = None
        self.batch_roi_start = None
        self.batch_roi_rect = None

        self.page_buttons_frame = tk.Frame(self)
        self.page_buttons_frame.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(self.page_buttons_frame, text="Single Image", command=self.show_single_page).pack(side=tk.LEFT, padx=4)
        tk.Button(self.page_buttons_frame, text="Batch Folder", command=self.show_batch_page).pack(side=tk.LEFT, padx=4)

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.page1_frame = tk.Frame(self.main_frame)
        self.page2_frame = tk.Frame(self.main_frame)

        self.status_label = tk.Label(self, text="Load an image to start", anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=4)

        self._create_single_page()
        self._create_batch_page()

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

        self.canvas = tk.Canvas(self.grid_frame, bg="black")
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        self.canvas.bind('<Configure>', lambda event: self.resize_image())
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)

        self.dolp_canvas = None
        self.aolp_canvas = None
        self.hist_canvas = None

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

        self.batch_mode_var = tk.StringVar(value="raw")
        tk.Label(self.batch_left_frame, text="Batch monolithic mode:").pack(pady=(12, 2), anchor="w")
        self.batch_mode_menu = tk.OptionMenu(self.batch_left_frame, self.batch_mode_var, "raw", "stokes_average")
        self.batch_mode_menu.pack(fill=tk.X, pady=2)

        self.batch_pol_var = tk.StringVar(value="linear")
        tk.Label(self.batch_left_frame, text="Polarization Type:").pack(pady=(12, 2), anchor="w")
        tk.Radiobutton(self.batch_left_frame, text="Linear", variable=self.batch_pol_var, value="linear").pack(anchor="w")
        tk.Radiobutton(self.batch_left_frame, text="Circular", variable=self.batch_pol_var, value="circular").pack(anchor="w")

        self.batch_calculate_btn = tk.Button(self.batch_left_frame, text="Calculate Batch", command=self.calculate_batch)
        self.batch_calculate_btn.pack(fill=tk.X, pady=12)

        self.batch_export_csv_btn = tk.Button(self.batch_left_frame, text="Export Batch CSV", command=self.export_batch_csv)
        self.batch_export_csv_btn.pack(fill=tk.X, pady=4)

        self.batch_canvas = tk.Canvas(self.batch_grid_frame, bg="black")
        self.batch_canvas.grid(row=0, column=0, columnspan=3, sticky='nsew', padx=2, pady=2)
        self.batch_canvas.bind('<Configure>', lambda event: self.resize_batch_image())
        self.batch_canvas.bind('<ButtonPress-1>', self.batch_on_mouse_press)
        self.batch_canvas.bind('<B1-Motion>', self.batch_on_mouse_drag)
        self.batch_canvas.bind('<ButtonRelease-1>', self.batch_on_mouse_release)

        self.batch_plot_canvases = {}

    def show_single_page(self):
        self.page_active = 'single'
        self.page2_frame.pack_forget()
        self.page1_frame.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Single-image analysis page active")

    def show_batch_page(self):
        self.page_active = 'batch'
        self.page1_frame.pack_forget()
        self.page2_frame.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Batch folder analysis page active")

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
            ax2.text(0.5, 0.5, 'AoLP not available for circular polarization', ha='center', va='center')
            ax2.axis('off')
        self._draw_plot(fig2, 'aolp_canvas', row=1, column=0)

        fig3, ax3 = plt.subplots(figsize=(4, 3))
        ax3.hist(DoLP.flatten(), bins=50, alpha=0.7)
        ax3.set_title('Histogram')
        ax3.set_xlabel('DoLP' if pol_type == 'linear' else 'DoCP')
        ax3.set_ylabel('Frequency')
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

    def _draw_plot(self, fig, attr_name, row, column):
        canvas = getattr(self, attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)
        setattr(self, attr_name, canvas)
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
        canvas = self.batch_plot_canvases.get(attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.batch_grid_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, column=column, sticky='nsew', padx=2, pady=2)
        self.batch_plot_canvases[attr_name] = canvas
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

    def export_csv(self):
        if self.processor.pol_params is None:
            self.status_label.config(text="Calculate polarization first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.processor.export_csv(file_path)
        self.status_label.config(text="Exported to CSV.")

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
