import tkinter as tk
from tkinter import filedialog
import cv2
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

        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        self.grid_frame = tk.Frame(self.main_frame)
        self.grid_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.grid_frame.grid_rowconfigure(0, weight=1)
        self.grid_frame.grid_rowconfigure(1, weight=1)
        self.grid_frame.grid_columnconfigure(0, weight=1)
        self.grid_frame.grid_columnconfigure(1, weight=1)

        self.load_btn = tk.Button(self.left_frame, text="Load Image", command=self.load_image)
        self.load_btn.pack(fill=tk.X, pady=4)

        self.display_demosaic_btn = tk.Button(self.left_frame, text="Display Demosaiced", command=self.display_demosaic)
        self.display_demosaic_btn.pack(fill=tk.X, pady=4)

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

        self.status_label = tk.Label(self, text="Load an image to start", anchor="w")
        self.status_label.pack(fill=tk.X, padx=8, pady=4)

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

    def display_demosaic(self):
        if self.processor.img is None:
            self.status_label.config(text="Load an image first.")
            return
        display = self.processor.demosaic_preview()
        cv2.imshow('Demosaiced Images', display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            im2 = ax2.imshow(AoLP, cmap='hsv')
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
        canvas = getattr(self, attr_name)
        if canvas is not None:
            canvas.get_tk_widget().destroy()
            setattr(self, attr_name, None)

    def clear_plots(self):
        self._clear_plot_canvas('dolp_canvas')
        self._clear_plot_canvas('aolp_canvas')
        self._clear_plot_canvas('hist_canvas')

    def export_csv(self):
        if self.processor.pol_params is None:
            self.status_label.config(text="Calculate polarization first.")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        self.processor.export_csv(file_path)
        self.status_label.config(text="Exported to CSV.")


def run_gui():
    app = PolarizationGUI()
    app.mainloop()


if __name__ == '__main__':
    run_gui()
