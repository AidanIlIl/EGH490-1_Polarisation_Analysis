# if the reading is at 255 of any pixzel the entire cluster of 4 pixels is considered saturated (as the DoP calculation will be inaccurate) and masked out (i.e. not included in DoP calculations)
# if the reading is 0 be sus
# there is code to find the saturated pixels.
# saturation disribution is probs good.
# distibution 1 per pixel cluster.
# per pxiel averaging should always be done before the stokes calculation
# justify why the superpixel averaging is done before the stokes calculation (i.e. that the stokes calculation is not linear and therefore the order of operations matters)
# half of report explaing how this stuff gets to the camera errors and shit .
# maybe 2 angles that are zero?
# think how we can calibrate and get known data?
# Gausian bluring
# Denoising methods and comparing the results (big chunk of paper)
# Signal to Noise Ratio

# de mosaic -> gausain blur (other image proccesing shabang) -> remosic -> calcualte polartsiaot. 

# look up how white balence works

#think of ways to calibrate the camera (polarisation shiz)

#%% 
# IMPORTS
import cv2
import os
import sys
import numpy as np
import warnings
import pickle
from time import time
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# try: # if running as a script
#     import helper_functions as hf
#     import polarisation_calcs as pc
#     import gvars as gvars
# except: # if running as a module
#     import src.helper_functions as hf
#     import src.polarisation_calcs as pc
#     import src.gvars as gvars

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helper_functions as hf
import polarisation_calcs as pc
import gvars as gvars

# CONSTANTS
D0, D45, D90, D135 = 0, 1, 2, 3 # polarisation angle indices

# cv2 variables
cv2font = cv2.FONT_HERSHEY_DUPLEX
black = (0, 0, 0)
white = (255, 255, 255)
cv2fontcol = (255, 0, 0) # red # (255, 255, 255) # black
sc_f = 5 # scale factor for images

#%% MAIN FUNCTION (FOR TESTING)
def main():
    
    # find all files in current directory with a .png extension, load the first one in as 'img'
    root = './test'
    if not os.path.exists(root): os.makedirs(root)
    
    # img = cv2.imread(next(filter(lambda x: x.endswith('.png'), os.listdir('./Tests/the_mask/images'))), cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(f"./Tests/{os.listdir('./Tests/')[0]}", cv2.IMREAD_GRAYSCALE)
       
    pol_params = img_to_AoI_to_linear_pol_params_pipeline(img) # also isolates the area of interest for the polarisation analysis
    XoLP_to_HSV_export(pol_params[1], f"{root}/test_hsv")
    # pol_export(f'{root}/test_pol_data', polvect) # one of these was like 57MB
    
    # get saturation percentages
    img_AoI = hf.crop_to_proportions(img, gvars.AoI_bounds) # <== for saturation analysis only (looking at unsplit image)
    oversat, undersat = hf.sat_pct(img_AoI)
    
    # superpixel averages
    DoLP_SP_avg, AoLP_SP_avg = XoLP_superpixel_avg_from_img(img)
    
    # dummy file params
    file_params = {'az': 30, 'ze': 180, 'cze': 60, 'exp': 100, 'ISO': 100, 'obs': False}
    
    pol_summary_export(f'{root}/test_pol_summary', pol_params, file_params, oversat, undersat, DoLP_SP_avg=DoLP_SP_avg, AoLP_SP_avg=AoLP_SP_avg) # dummy values for azimuth, zenith, camera zenith

#%% HIGH LEVEL FUNCTIONS
def linear_polarisation_analysis(test_name, img, testdir, hdir, paramstr):
    '''Full Analysis pipeline for linear polarisation
    input: test_name - name of the test (used for filenames)
           img - image to analyse (un-split, un-cropped, polarsens image)
           testdir - directory to save images and data
           hdir - directory to save hsv images
           paramstr - string to encode into filenames (and from which to extract parameters)
    output: an image showing the degree of linear polarisation per pixel (saved to hdir)
    '''
    # p dict keys: ['im_num': image number, 'az': azimuth, 'ze': zenith, 'cze': camera zenith, 'exp': exposure, 'ISO': gain]
    img_params = hf.param_extract(paramstr) 
    
    '''
    TODO: passing only the area of interest to analysis functions:
      argument for doing this at the camera level: you can crop the image but keep all the exact polarisation values 
      (i.e. it won't cut off any group of 4 pixels)
      otherwise have to split the image into each pol angle and THEN crop each one separately
      (current approach)
    '''
    # get stokes vector and polarisation vector (splits an unaltered image and crops to AoI)
    pol_params_lin = img_to_AoI_to_linear_pol_params_pipeline(img) 
    pol_vect_lin = pol_params_lin[1] # for HSV export
    HSV_AoI = XoLP_to_HSV_export(pol_vect_lin, f"{hdir}/{paramstr}_HSV")
    DoLP_SP_avg, AoLP_SP_avg = XoLP_superpixel_avg_from_img(img) # superpixel averages

    # get the % total saturation and undetected for the cropped image in the area of interest
    img_AoI = hf.crop_to_proportions(img, gvars.AoI_bounds) # <== for saturation analysis only (looking at unsplit image)
    oversat, undersat = hf.sat_pct(img_AoI)

    # export the polarisation summary
    # pol_export(f"{hdir}/POL_{paramstr}", pol_params(img)) # large file size (~57MB)
    pol_summary_export(
        # csv_path=f"{testdir}/{test_name}_POL_SUMMARY_LINEAR", 
        csv_path = os.path.join(testdir, f'{test_name}_POL_SUMMARY_LINEAR'),
        pol_params=pol_params_lin, 
        file_params=img_params,
        oversaturation=oversat,
        undersaturation=undersat,
        DoLP_SP_avg=DoLP_SP_avg, # <== comment out these lines to 
        AoLP_SP_avg=AoLP_SP_avg  # <== record vector averages instead
    )
    
    # return the most recent raw image and HSV image side-by-side
    # bounding box in left image ('img') showing area analysed
    img_preview = hf.gs3c(hf.img_res(img, 30))
    x1, x2 = int(img_preview.shape[1]*gvars.x1()), int(img_preview.shape[1]*gvars.x2())
    y1, y2 = int(img_preview.shape[0]*gvars.y1()), int(img_preview.shape[0]*gvars.y2())
    img_preview = cv2.rectangle(img_preview, (x1, y1), (x2, y2), cv2fontcol, 2)
    HSV_AoI = cv2.resize(hf.hsv2bgr(HSV_AoI), (img_preview.shape[1], img_preview.shape[0])) # resize the HSV image to match the raw image
    return np.hstack([img_preview, HSV_AoI])
    
    # full_hsv_in_bgr = hsv2bgr(DoPVis(XoP(stokes(pol_split(img)))))
    # combined_img = img_res(np.hstack([gs3c(img_res(img,50)), full_hsv_in_bgr]), 50)
    # cv2.imshow('Raw and HSV', combined_img)

def circular_polarisation_analysis(test_name, img, testdir, cdir, paramstr):
    '''Full Analysis pipeline for circular polarisation
    input: test_name - name of the test (used for filenames)
           img - image to analyse (un-split, un-cropped, polarsens image)
           testdir - directory to save images and data
           cdir - directory to save circular polarisation intensity images
           paramstr - string to encode into filenames (and from which to extract parameters)
    output: an image showing the degree of circular polarisation per pixel (saved to cdir)
    '''
    # p dict keys: ['im_num': image number, 'az': azimuth, 'ze': zenith, 'cze': camera zenith, 'exp': exposure, 'ISO': gain]
    img_params = hf.param_extract(paramstr)

    pol_params_circ = img_to_AoI_to_circular_pol_params_pipeline(img)
    DoCP = pol_params_circ[1][0] # for exporting
    DoCP_SP = XoCP_superpixel_avg_from_img(img)[0] # superpixel average
    CPI_AoI = DoCP_to_circ_pol_intensity_img_export(DoCP, DoCP_SP, f"{cdir}/{paramstr}_CPI")
    # plot_DoCP_with_colorbar(DoCP)


    DoCP_SP_avg = XoCP_superpixel_avg_from_img(img) # superpixel average
    
    img_AoI = hf.crop_to_proportions(img, gvars.get_AoI_bounds()) # <== for saturation analysis only
    oversat, undersat = hf.sat_pct(img_AoI)
    
    # save the circular polarisation summary
    pol_summary_export(
        # csv_path=f"{testdir}/{test_name}_POL_SUMMARY_CIRCULAR",
        csv_path = os.path.join(testdir, f'{test_name}_POL_SUMMARY_CIRCULAR'),
        pol_params=pol_params_circ,
        file_params=img_params,
        oversaturation=oversat,
        undersaturation=undersat,
        circ_test=True,
        DoCP_SP_avg=DoCP_SP_avg
    )

    # return the most recent raw image and CPI image side-by-side
    # bounding box in left image ('img') showing area analysed
    img_preview = hf.gs3c(hf.img_res(img.copy(), 30))
    x1, x2 = int(img_preview.shape[1]*gvars.x1()), int(img_preview.shape[1]*gvars.x2())
    y1, y2 = int(img_preview.shape[0]*gvars.y1()), int(img_preview.shape[0]*gvars.y2())
    img_preview = cv2.rectangle(img_preview, (x1, y1), (x2, y2), cv2fontcol, 2)
    CPI_AoI = cv2.resize(CPI_AoI, (img_preview.shape[1], img_preview.shape[0])) # resize the CPI image to match the raw image
    return np.hstack([img_preview, CPI_AoI])

def display_demosaiced_images(img):
    '''Demosaic the image into 4 images at each angle and display in a 2x2 window'''
    pol_angles = hf.pol_split(img)
    # Assume 4 angles: 0, 45, 90, 135
    # Normalize for display
    pol_angles_norm = [(angle / 255.0 * 255).astype(np.uint8) for angle in pol_angles]
    # Create 2x2 grid
    top_row = np.hstack([pol_angles_norm[0], pol_angles_norm[1]])
    bottom_row = np.hstack([pol_angles_norm[2], pol_angles_norm[3]])
    combined = np.vstack([top_row, bottom_row])
    # Resize to fit screen, say max 1200x800
    h, w = combined.shape[:2]
    scale = min(1200 / w, 800 / h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Demosaiced Images (0, 45, 90, 135)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def demosaic_combined(img):
    pol_angles = hf.pol_split(img)
    pol_angles_norm = [(angle / 255.0 * 255).astype(np.uint8) for angle in pol_angles]
    top_row = np.hstack([pol_angles_norm[0], pol_angles_norm[1]])
    bottom_row = np.hstack([pol_angles_norm[2], pol_angles_norm[3]])
    return np.vstack([top_row, bottom_row])


def _save_figure(fig, path):
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def save_dolp_map(DoLP, path, pol_type='linear'):
    fig, ax = plt.subplots()
    if pol_type == 'circular':
        im = ax.imshow(DoLP, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('DoCP')
    else:
        im = ax.imshow(DoLP, cmap='viridis')
        ax.set_title('DoLP')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def save_aolp_map(AoLP, path):
    fig, ax = plt.subplots()
    ax.imshow(AoLP, cmap='hsv')
    ax.set_title('AoLP')
    plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    _save_figure(fig, path)


def save_histogram(DoLP, path, pol_type='linear'):
    fig, ax = plt.subplots()
    ax.hist(DoLP.flatten(), bins=50, alpha=0.7)
    ax.set_title('DoLP Histogram' if pol_type == 'linear' else 'DoCP Histogram')
    ax.set_xlabel('DoLP' if pol_type == 'linear' else 'DoCP')
    ax.set_ylabel('Frequency')
    _save_figure(fig, path)


def save_polarization_maps(pol_params, output_dir, prefix='pol', pol_type='linear'):
    os.makedirs(output_dir, exist_ok=True)
    S, pol_vect = pol_params
    DoLP, AoLP = pol_vect
    save_dolp_map(DoLP, os.path.join(output_dir, f'{prefix}_dolp.png'), pol_type=pol_type)
    if pol_type == 'linear':
        save_aolp_map(AoLP, os.path.join(output_dir, f'{prefix}_aolp.png'))
    save_histogram(DoLP, os.path.join(output_dir, f'{prefix}_hist.png'), pol_type=pol_type)


def save_hsv_image(pol_vect, output_path):
    XoLP_to_HSV_export(pol_vect, output_path)


def save_circular_intensity(DoCP, output_path):
    DoCP_to_circ_pol_intensity_img_export(DoCP, np.nanmean(DoCP), output_path)


class PolarizationProcessor:
    def __init__(self):
        self.img_org = None
        self.img = None
        self.pol_params = None
        self.pol_type = 'linear'
        self.roi = None
        self.original_img = None

    def load_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return False
        self.img_org = image.copy()
        self.img = image.copy()
        self.original_img = Image.fromarray(self.img)
        self.pol_params = None
        self.roi = None
        return True

    def reset_image(self):
        if self.img_org is not None:
            self.img = self.img_org.copy()
            self.pol_params = None

    def apply_filter(self, filter_name='None', kernel=5, sigma=1.0):
        if self.img_org is None:
            return
        if filter_name == 'Gaussian Blur':
            if kernel % 2 == 0:
                kernel += 1
            pol_angles = hf.pol_split(self.img_org)
            filtered_angles = [cv2.GaussianBlur(angle, (kernel, kernel), sigma) for angle in pol_angles]
            self.img = self.pol_combine(filtered_angles)

        else:
            self.reset_image()

    def load_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return False
        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
        batch_files = sorted(
            [os.path.join(folder_path, f) for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in supported_exts]
        )
        batch_params = []
        valid_files = []
        for file_path in batch_files:
            name = os.path.splitext(os.path.basename(file_path))[0]
            try:
                params = hf.param_extract(name)
            except Exception:
                continue
            valid_files.append(file_path)
            batch_params.append(params)

        if not valid_files:
            return False

        self.batch_files = valid_files
        self.batch_params = batch_params
        self.batch_results = []
        self.batch_mode = 'raw'
        self.batch_pol_type = 'linear'
        self.batch_roi = None
        self.batch_folder = folder_path

        first_image = cv2.imread(self.batch_files[0], cv2.IMREAD_GRAYSCALE)
        if first_image is None:
            return False
        self.batch_first_img = first_image.copy()
        self.img_org = first_image.copy()
        self.img = first_image.copy()
        self.original_img = Image.fromarray(self.img)
        self.pol_params = None
        return True

    def batch_set_roi(self, roi):
        self.batch_roi = roi
        self.roi = roi

    def batch_calculate(self, mode='raw', pol_type='linear'):
        if not hasattr(self, 'batch_files') or not self.batch_files:
            return False
        if self.batch_roi is None or self.batch_roi[2] <= 0 or self.batch_roi[3] <= 0:
            return False

        self.batch_mode = mode
        self.batch_pol_type = pol_type
        self.batch_results = []

        x, y, w, h = self.batch_roi
        for file_path, params in zip(self.batch_files, self.batch_params):
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            c_bounds = [x / img.shape[1], (x + w) / img.shape[1], y / img.shape[0], (y + h) / img.shape[0]]
            pol_angles = polarsens_to_cropped_pol_angles(img, c_bounds)

            if pol_type == 'linear':
                if mode == 'raw':
                    pol_angles_avg = [np.mean(pa) for pa in pol_angles]
                    S = stokes_linear(pol_angles_avg)
                    pol_vect = XoLP(S)
                else:
                    S_pixels = stokes_linear(pol_angles)
                    S = np.array([np.mean(s_component) for s_component in S_pixels])
                    pol_vect = XoLP(S)

                
                DoLP_img, AoLP_img = XoLP(stokes_linear(pol_angles))
                dolp_std = float(np.std(DoLP_img)) if isinstance(DoLP_img, np.ndarray) else 0.0
                aolp_std = float(np.std(AoLP_img)) if isinstance(AoLP_img, np.ndarray) else 0.0
            else:
                if mode == 'raw':
                    pol_angles_avg = [np.mean(pa) for pa in pol_angles]
                    S = stokes_circular(pol_angles_avg)
                    pol_vect = XoCP(S)
                else:
                    S_pixels = stokes_circular(pol_angles)
                    S = np.array([np.mean(s_component) for s_component in S_pixels])
                    pol_vect = XoCP(S)
                
                poly_vect_pixel = XoCP(stokes_circular(pol_angles))
                DoCP_img = poly_vect_pixel[0]
                dolp_std = float(np.std(DoCP_img)) if isinstance(DoCP_img, np.ndarray) else 0.0
                aolp_std = 0.0

            roi_img = img[int(y):int(y+h), int(x):int(x+w)]
            oversat, undersat = hf.sat_pct(roi_img)

            self.batch_results.append({
                'file': file_path,
                'params': params,
                'stokes': S,
                'pol': pol_vect,
                'dolp_std': dolp_std,
                'aolp_std': aolp_std,
                'saturation': oversat
            })

        return self.batch_results

    def export_batch_csv(self, file_path):
        if not hasattr(self, 'batch_results') or not self.batch_results:
            return False
        csv_base = file_path[:-4] if file_path.lower().endswith('.csv') else file_path
        if self.batch_pol_type == 'linear':
            fields = ['file', 'az', 'ze', 'cze', 'exp', 'ISO', 'obs', 'DoLP', 'AoLP_deg', 'DoLP_std', 'AoLP_std', 'Saturation_pct']
        else:
            fields = ['file', 'az', 'ze', 'cze', 'exp', 'ISO', 'obs', 'DoCP', 'DoCP_std', 'Saturation_pct']
        hf.csv_init(csv_base, fields)

        for result in self.batch_results:
            params = result['params']
            pol = result['pol']
            if self.batch_pol_type == 'linear':
                DoP, AoP = pol
                row = [
                    result['file'], params['az'], params['ze'], params['cze'], params['exp'], params['ISO'], params['obs'],
                    float(DoP), float(np.rad2deg(AoP)), float(result['dolp_std']), float(result['aolp_std']), float(result['saturation'])
                ]
            else:
                DoCP = pol[0]
                row = [
                    result['file'], params['az'], params['ze'], params['cze'], params['exp'], params['ISO'], params['obs'],
                    float(DoCP), float(result['dolp_std']), float(result['saturation'])
                ]
            hf.csv_append(csv_base, row)
        return True

    def preview_filtered_demosaiced(self, filter_name='None', kernel=5, sigma=1.0):
        if self.img_org is None:
            return None
        if filter_name == 'Gaussian Blur':
            if kernel % 2 == 0:
                kernel += 1
            pol_angles = hf.pol_split(self.img_org)
            filtered_angles = [cv2.GaussianBlur(angle, (kernel, kernel), sigma) for angle in pol_angles]
        else:
            filtered_angles = hf.pol_split(self.img_org)
        pol_angles_norm = [(angle / 255.0 * 255).astype(np.uint8) for angle in filtered_angles]
        top_row = np.hstack([pol_angles_norm[0], pol_angles_norm[1]])
        bottom_row = np.hstack([pol_angles_norm[2], pol_angles_norm[3]])
        combined = np.vstack([top_row, bottom_row])
        return combined
    
    def calculate_saturation_histogram(self, roi=None):
        if self.img is None:
            return None
        
        if roi is not None and roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            h_img, w_img = self.img.shape[:2]
            c_bounds = [x / w_img, (x + w) / w_img, y / h_img, (y + h) / h_img]
            img_AoI = hf.crop_to_proportions(self.img, c_bounds)
            
        else:
            img_AoI = hf.crop_to_proportions(self.img, gvars.get_AoI_bounds())
        oversat, undersat = hf.sat_pct(img_AoI)
        fig, ax = plt.subplots()
        ax.hist(img_AoI.flatten(), bins=50, alpha=0.7)
        ax.set_title(f'Saturation Histogram (Oversat: {oversat:.2f}%, Undersat: {undersat:.2f}%)')
        ax.set_xlabel('Pixel Intensity')
        ax.set_ylabel('Frequency')
        return fig

    def calculate(self, pol_type='linear', roi=None):
        self.pol_type = pol_type
        self.roi = roi
        c_bounds = None
        if roi is not None and roi[2] > 0 and roi[3] > 0:
            x, y, w, h = roi
            h_img, w_img = self.img.shape[:2]
            c_bounds = [x / w_img, (x + w) / w_img, y / h_img, (y + h) / h_img]
        if self.pol_type == 'linear':
            self.pol_params = img_to_AoI_to_linear_pol_params_pipeline(self.img, c_bounds)
        else:
            self.pol_params = img_to_AoI_to_circular_pol_params_pipeline(self.img, c_bounds)
        return self.pol_params

    def save_image(self, path):
        if self.img is None:
            return False
        cv2.imwrite(path, self.img)
        return True

    def save_demosaic(self, path):
        if self.img is None:
            return False
        combined = demosaic_combined(self.img)
        cv2.imwrite(path, combined)
        return True

    def save_polarization_outputs(self, output_dir, prefix='pol', file_params=None, oversat=0.0, undersat=0.0, save_csv=False):
        if self.pol_params is None:
            return False
        os.makedirs(output_dir, exist_ok=True)
        save_polarization_maps(self.pol_params, output_dir, prefix=prefix, pol_type=self.pol_type)
        if self.pol_type == 'linear':
            save_hsv_image(self.pol_params[1], os.path.join(output_dir, f'{prefix}_HSV'))
        else:
            DoCP = self.pol_params[1][0]
            save_circular_intensity(DoCP, os.path.join(output_dir, f'{prefix}_DoCP'))
        if save_csv:
            csv_path = os.path.splitext(os.path.join(output_dir, f'{prefix}.csv'))[0]
            self.export_csv(csv_path, file_params=file_params, oversat=oversat, undersat=undersat)
        return True

    def export_csv(self, file_path, file_params=None, oversat=0.0, undersat=0.0):
        if self.pol_params is None:
            return False
        file_params = file_params or {'az': 0, 'ze': 0, 'cze': 0, 'exp': 100, 'ISO': 100, 'obs': False}
        circ_test = self.pol_type == 'circular'
        DoLP_SP_avg, AoLP_SP_avg, DoCP_SP_avg = None, None, None
        if self.img is not None:
            if not circ_test:
                DoLP_SP_avg, AoLP_SP_avg = XoLP_superpixel_avg_from_img(self.img)
            else:
                DoCP_SP_avg = XoCP_superpixel_avg_from_img(self.img)
        pol_summary_export(file_path, self.pol_params, file_params, oversat, undersat, DoLP_SP_avg=DoLP_SP_avg, AoLP_SP_avg=AoLP_SP_avg, circ_test=circ_test, DoCP_SP_avg=DoCP_SP_avg)
        return True

    def pol_combine(self, pol_angles):
        deg0, deg45, deg90, deg135 = pol_angles
        h, w = deg0.shape
        full_h, full_w = 2*h, 2*w
        img = np.zeros((full_h, full_w), dtype=np.uint8)
        img[0::2, 0::2] = deg90
        img[0::2, 1::2] = deg45
        img[1::2, 0::2] = deg135
        img[1::2, 1::2] = deg0
        return img
    

#%% LINEAR POLARISATION ANALYSIS FUNCTIONS
def img_to_AoI_to_linear_pol_params_pipeline(img, c_bounds=None):
    '''Calculate polarisation parameters from an image (isolating a region of interest)
    inputs:
        img: 1 channel polarsens image to analyse (dtype=uint8) (see: polarsens_to_cropped_pol_angles function)
        c_bounds: boundaries (as a fraction of the image size) for the area of interest (dtype=float)
                  bounds are [x1, x2, y1, y2] where x1, x2 are the left and right bounds, and y1, y2 are the top and bottom bounds
        i_thresh: intensity threshold for the area of interest (any pixel with intensity below this value will be ignored)
    outputs:
        stokes parameters [I, Q, U, V]: intensity, linear polarisation (0-90), linear polarisation(45-135), circular polarisation (dtype=float)
        pol_vector [DoP, AoP]: degree of polarisation, angle of polarisation (dtype=float)
    '''
    pol_img_AoI = polarsens_to_cropped_pol_angles(img, c_bounds)
    S_lin = stokes_linear(pol_img_AoI) # Stokes parameters
    polvect_lin = XoLP(S_lin) # Degree and Angle of Polarisation (the * unpacks the list into arguments)

    # I ~ [0,510]
    # Q, U, V ~ [-255,255]
    # DoP = sqrt(Q^2 + U^2 + V^2) / I
    # DoP ~ [0, sqrt(255^2 + 255^2 + 255^2)] / [0, 510] = [0, 1]
    # DoP ~ [[0, 65025] + [0, 65025] + 0] / [0, 510] = [0, 255]
    
    return S_lin, polvect_lin

def stokes_linear(pol_angles):
    '''Calculate Stokes parameters from polarisation angles
    inputs:
        pol_angles: a list of 4 images; in order, the polarisation angles 0, 45, 90, 135 (dtype=uint8)
        circular: whether to calculate circular polarisation (not implemented yet) (dtype=bool)
    outputs:
        stokes parameters [I, Q, U, V]: intensity, linear polarisation (0-90), linear polarisation(45-135), circular polarisation (dtype=float)
    '''
        # Stokes Parameters
    I_m = (pol_angles[D0].astype(int) + pol_angles[D45].astype(int) \
        + pol_angles[D90].astype(int) + pol_angles[D135].astype(int)) / 2 # manhattan (considering all four angles)
    I_m_2angles = pol_angles[D0].astype(int) + pol_angles[D90].astype(int) # from Field Guide to Polarization
    I_e = np.sqrt(pol_angles[D0].astype(int)**2 + pol_angles[D90].astype(int)**2) # euclidean considering only 0 and 90
    # I_m in [0, 2*255 = 510]; I_e in [0, sqrt(2*255^2) ~= 22]; Q, U, V in [-255, 255]
    I = I_m_2angles # Intensity
    Q = pol_angles[D0].astype(int) - pol_angles[D90].astype(int) # I_0 - I_90
    U = pol_angles[D45].astype(int) - pol_angles[D135].astype(int) # I_45 - I_135
    V = np.zeros_like(Q) # no circular polarisation currently but maybe we'll do it later

    I_e2 = np.sqrt(Q**2 + U**2 + V**2) # euclidean defined by Q, U, V; ALSO from field guide to polarization
    return np.array([I, Q, U, V])

def XoLP(S_lin):
    '''Calculate degree of polarisation and angle of polarisation from Stokes parameters
    inputs:
        S_lin: linear [I, Q, U, V==0] (dtype=float)
    outputs:
        [DoLP, AoLP]: linear polarisation vector [degree of linear polarisation, angle of linear polarisation] (dtype=float)
    '''
    I, Q, U, V = S_lin
    #TODO: Sam, please confirm removal of V term from square root as DoLP does not assess circular component.
    # Degree and Angle of Polarisation
    DoLP = stokes_norm( # DoP in [0, 1] (stokes_norm handles division by 0)
        np.sqrt( np.square(Q) + np.square(U) + np.square(V) ), 
                                I                           )       
    AoLP = np.arctan2(U, Q) / 2 # AoP in [-pi/2, pi/2]  # equation in most of literature (including the / 2);  

    if isinstance(AoLP, list) or isinstance(AoLP, np.ndarray): # default approach
        AoLP[AoLP < 0] += np.pi     # AoP in [0    ,   pi]
        if AoLP.max() > np.pi or AoLP.min() < 0: 
            Exception(f'AoP is not in [0, pi] (max is {AoLP.max()}, min is {AoLP.min()})')
    else: # scalar values, 'super pixel' approach
        AoLP += 0 if AoLP >= 0 else np.pi
        if AoLP > np.pi or AoLP < 0: 
            Exception(f'AoP is not in [0, pi] ({AoLP})')
    return [DoLP, AoLP]

    # DoP = np.sqrt(Q**2 + U**2 + V**2) / I # degree of polarisation
    # AoP = 0.5 * np.arctan2(U, Q) # angle of polarisation
    # return DoP, AoP

def XoLP_superpixel_avg_from_img(img):
    '''
    Calculate the degree and angle of polarisation by averaging the intensities of each polarisation angle (i.e. treat the whole image as a superpixel)
    inputs: img: 1-channel polarsens image (dtype=uint8)
    outputs: [DoP, AoP]: superpixel average degree of polarisation and angle of polarisation (dtype=float)
    '''
    pol_angles = polarsens_to_cropped_pol_angles(img)
    pol_angles_avg = [np.mean(pol_angle) for pol_angle in pol_angles]
    S_lin_SP = stokes_linear(pol_angles_avg) # (averaged) Stokes parameters
    return XoLP(S_lin_SP) # Degree and Angle of Polarisation


def XoLP_cam_normalise(XoLP_img):
    '''Normalise the degree of polarisation and angle of polarisation images from the camera (given a DoP/AoP PixelFormat image from the camera)
    inputs:
        XoP_img: an 8-bit 2-channel image of the degree of polarisation and angle of polarisation
    outputs:
        normalised degree of polarisation and angle of polarisation images
    '''
    DoLP, AoLP = XoLP_img
    DoLP = DoLP.astype(np.float64) / 255.0            # DoP in [0, 1]
    AoLP = AoLP.astype(np.float64) / 255.0 * np.pi    # AoP in [0, pi]
    return [DoLP, AoLP]

#%% CIRCULAR POLARISATION ANALYSIS FUNCTIONS
def img_to_AoI_to_circular_pol_params_pipeline(img, c_bounds=None): 
    pol_img_AoI = polarsens_to_cropped_pol_angles(img, c_bounds)
    
    # Calculate V (or S_3) given the 45 degree intensity behind a quarter waveplate
    S_circ = stokes_circular(pol_img_AoI) # Q (S[1]) and U (S[2]) are 0
    #TODO: Sam, not sure if this is the best place to put standard deviation calculation and return it

    # Calculate the DoP and AoP
    pol_vector_circ = XoCP(S_circ) # 
    return S_circ, pol_vector_circ

def stokes_circular(pol_angles):
    '''
    returns a (slightly incomplete) circular stokes vector given circular polarisation angles
    '''
    # TODO: can decouple circular stokes from linear stokes if we just return V and I_circ (but would need restructuring elsewhere)
    I_0_w, I_45_w, I_90_w, I_135_w = pol_angles #intensity behind the waveplate and linear filter pixel

    # I = (I_0_w + I_45_w + I_90_w + I_135_w)/2 # intensity from behind the 1/4 waveplate, calculated from muler_math to cancel all but oridional intensity - NOT REQUIRED DUE TO NEW V CALCULATION
    I = I_0_w + I_90_w # from Field Guide to Polarization
    # I = np.zeros_like(I_0_w)
    Q = np.zeros_like(I_0_w)
    U = np.zeros_like(I_0_w)
    V = I_135_w - I_45_w # Updated circular polarisation calculation
    # V = I - (2 * I_45_w) # old version of circular polarisation calculation
    
    
    return np.array([I, Q, U, V])

def XoCP(S_circ):
    '''
    Calculate the degree of circular polarisation and angle of circular polarisation (NOT IMPLEMENTED) from Stokes parameters
    inputs: S_circ, a stokes vector with: [I_circ, Q==0, U==0, V] (dtype=float)
    outputs: [DoCP, AoCP]: circular polarisation vector [degree of polarisation, angle of polarisation==0] (dtype=float)
    (disclaimer: not really a proper polarisation vector - no angle, DoCP is just normalised V)
    '''
    I, V = S_circ[0], S_circ[3]
    V_Norm = stokes_norm(V, I)
    AoCP = np.zeros_like(V_Norm) # NO CIRCULAR POLARISATION ANGLE #TODO: remove, value has no meaning
    return [V_Norm, AoCP] # different data types so not np.array

def XoCP_superpixel_avg_from_img(img):
    '''
    Calculate the degree and angle of polarisation by averaging the intensities of each polarisation angle (i.e. treat the whole image as a superpixel)
    inputs: img: 1-channel polarsens image (dtype=uint8)
    outputs: [DoP, AoP]: superpixel average degree of polarisation and angle of polarisation (dtype=float)
    '''
    pol_angles = polarsens_to_cropped_pol_angles(img)
    pol_angles_avg = [np.mean(pol_angle) for pol_angle in pol_angles]
    S_circ_SP = stokes_circular(pol_angles_avg) # (averaged) Stokes parameters
    return XoLP(S_circ_SP) # Degree and Angle of Polarisation

#%% DATA EXPORT FUNCTIONS
## export polarisation data to csv file
# export desired info to the next row of a csv file
def pol_summary_export(csv_path, pol_params, file_params, oversaturation, undersaturation, DoLP_SP_avg=None, AoLP_SP_avg=None, circ_test=False, DoCP_SP_avg=None):
    '''Export polarisation data to a csv file
    inputs:
        csvpath: path to save csv file (without .csv extension) (dtype=str)
        polparams: list of [stokes vector [I, Q, U, V] (dtype=float), polarisation vector [DoP, AoP] (dtype=float)], i.e. return of pol_params
        angles: azimuth, zenith, and camera zenith angles tuple (dtype=float)
        exposure: exposure time of the image (dtype=int)
        saturation: percentage of pixels that are saturated (dtype=float)
        undetected: percentage of pixels that are undetected (dtype=float)
        DoLP_SP_avg: 'superpixel' average; the degree of polarisation calculated by averaging the intensities of each polarisation angle (magnitude)
        AoLP_SP_avg: 'superpixel' average; the angle of polarisation calculated by averaging the intensities of each polarisation angle (rad)
        circ_test: whether the test was for circular polarisation (dtype=bool)
    outputs:
        save data to csv file
    '''
    # angles is a tuple of azimuth, zenith, and camera zenith
    angles = [file_params['az'], file_params['ze'], file_params['cze']]
    exposure = file_params['exp']
    gain = file_params['ISO']
    obstructed = file_params['obs']
    az, zen, cam_zen = angles
    phase_angle = pc.phase_angle(cam_zen, az, zen)
    plane_angle = pc.plane_azimuth_angle(cam_zen, az, zen)
    
    # stokes and polarisation vectors
    S, pol_vect = pol_params
    S_avg = [np.mean(Sn) for Sn in S]
    DoP, AoP = pol_vect
    
    # calculate averages, prepare data
    if not circ_test: 
        fields = gvars.linear_csv_fields
        
        # DoLP_v_avg, AoLP_v_avg = pc.vector_average(DoP, AoP) # currently in radians
        
        ## use vector averages unless superpixel averages are provided in the function call
        # DoLP_avg = DoLP_SP_avg if DoLP_SP_avg else DoLP_v_avg
        # AoLP_avg = AoLP_SP_avg if AoLP_SP_avg else AoLP_v_avg
        DoLP_avg = DoLP_SP_avg
        AoLP_avg = AoLP_SP_avg

        AoLP_avg_deg = np.rad2deg(AoLP_avg)
        DoP_SD = np.std(DoP)
        AoP_SD = np.rad2deg(np.std(AoP))
        
        data = [az, zen, cam_zen, phase_angle, plane_angle, 
                S_avg[0], S_avg[1], S_avg[2], S_avg[3],
                DoLP_avg, AoLP_avg_deg, DoP_SD, AoP_SD,
                exposure, gain, oversaturation, undersaturation, obstructed]
    else: 
        fields = gvars.circular_csv_fields
        V_avg = S_avg[3]
        # use np average unless superpixel averages are provided in the function call
        DoCP_avg = DoCP_SP_avg if DoLP_SP_avg else np.mean(DoP)
        DoCP_SD = np.std(DoP)
        
        data = [az, zen, cam_zen, phase_angle, plane_angle, 
                V_avg, DoCP_avg, DoCP_SD,
                exposure, gain, oversaturation, undersaturation, obstructed]
    
    # write data to csv file
    # if file doesn't exist, create it with headers
    if not os.path.isfile(f"{csv_path}.csv"): hf.csv_init(csv_path, fields)
    hf.csv_append(csv_path, data)
        
# export polarisation data to pickle file
def pol_export(polpath, poldata):
    with open(f"{polpath}.pkl", 'wb') as f:
        pickle.dump([poldata], f)
        
def XoLP_to_HSV_export(pol_vector, hsv_path=None):
    '''Export polarisation data as an HSV image
    inputs:
        hsvpath: path to save image (without .png extension) (dtype=str)
        polvect: polarisation vector [DoP: 2D np array in [0,1], AoP: 2D np array in [0, pi]] (dtype=float)
    outputs:
        HSV image (in HSV) (3-channel np array) (dtype=uint8)
        save HSV image (converted to BGR) as a .png file
    '''
    HSV_img = DoP2HSV(pol_vector) # AoP and DoP encoded directly 'H' and 'V' channels (S is constant 255)
    if hsv_path: cv2.imwrite(f"{hsv_path}.png", hf.hsv2bgr(HSV_img)) # save image as BGR if path is provided
    return HSV_img                                    # return image as HSV

def save_lin_img_as_HSV(filepath, img): # saves a polarisation image as an HSV image
    '''encode polarisation info from a polarisation image as HSV values, then convert it back to BGR and save it
    inputs:
        filepath: the filepath to save the image to
        img: a mono8 polarisation image
    '''
    if not isinstance(filepath, str): raise Exception('filepath must be a string')
    # TODO: validate file extension
    if len(img.shape) != 2: raise Exception('img must have 2 dimensions')
    polvect = img_to_AoI_to_linear_pol_params_pipeline(img)[1]
    return XoLP_to_HSV_export(polvect, filepath)

def DoCP_to_circ_pol_intensity_img_export(DoCP, DoCP_SP, ci_path=None):
    """
    Plots and saves a 2D numpy array (DoCP) with each pixel in the range -1 (LH, red) to 1 (RH, blue).
    Includes a colorbar and sets the title to "Degree of Circular Polarisation per Pixel".
   
    Parameters:
    filepath (str): the filepath to save the image to
    DoCP (numpy.ndarray): degree of circular polarisation per pixel as a 2D numpy array with pixel values in the range -1 to 1.
    """
    # plot the DoCP image with a colorbar
    plt.figure(figsize=(10, 8))
    # anything under -1 or over 1 should be represented as a unique color
    cmap = plt.cm.RdBu
    # set the color of the over and under values to black
    cmap.set_bad('black')
    cmap.set_over('purple')
    cmap.set_under('green')
    plt.imshow(DoCP, cmap=cmap, vmin=-2, vmax=2)
    cbar = plt.colorbar(extend='both')
    cbar.set_label('Degree of Circular Polarisation')

    # report stats on colour bar
    xpos = 3.5
    stats = [np.min(DoCP), np.mean(DoCP), np.median(DoCP), np.max(DoCP)]
    stat_strings = [f'Min: {stats[0]:.2f}', f'Mean: {stats[1]:.2f}', f'Median: {stats[2]:.2f}', f'Max: {stats[3]:.2f}']
    # stat string colours are red, green, orange, black
    for stat, stat_str in zip(stats, stat_strings):
        ypos = stat
        if ypos > 2: ypos = 2
        if ypos < -2: ypos = -2
        cbar.ax.text(xpos, stat, stat_str)
        cbar.ax.axhline(stat)
    cbar.ax.text(xpos-0.5, 2.5, f'SP Mean: {DoCP_SP:.2f}')
    
    plt.title('Degree of Circular Polarisation per Pixel')
    # label the x axis 'x Pixel Position (px)' and the y axis 'y Pixel Position (px)'
    plt.xlabel('X Pixel Position (px)'), plt.ylabel('Y Pixel Position (px)')
    # plt.show()

    # convert the plot to an image
    fig = plt.gcf()
    
    fig.canvas.draw()
    circ_pol_intensity_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    circ_pol_intensity_img = circ_pol_intensity_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    # save the image
    if ci_path: 
        if ci_path[-4] == '.': ci_path = ci_path[:-4] # remove three-letter file extension if it exists
        cv2.imwrite(f"{ci_path}.png", circ_pol_intensity_img) # save image if path is provided
    return circ_pol_intensity_img

#%% LOCAL HELPER FUNCTIONS

imgcvt = lambda pol: (pol*255).astype(np.uint8)
'''converts a 2D array of floats in [0,1] to a 2D array of uint8s in [0,255]
    inputs:
        pol: 2D array of floats in [0,1]
    outputs:
        2D array of uint8s in [0,255]
'''

def stokes_norm(pol, I):
    '''Normalise Stokes parameters and remove nan values
    inputs:
        pol: polarisation parameter (such as Q, U, V, or DoP)
        I: intensity
    outputs:
        normalised polarisation parameter
    '''
    with warnings.catch_warnings(): # I know I'm dividing by 0, don't worry about it
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # intensity 0 = infinite result, however also no DoP (so make it 0 whether +ve or -ve)
        diff_norm = np.nan_to_num(posinf=0, neginf=0, x=( 
            pol / I # 'Q / I'
        ))
        return diff_norm

def DoP2HSV(polvect): # encodes DoP and AoP into HSV image as 'H' and 'V' channels
    '''Encode degree of polarisation and angle of polarisation into an HSV image
    inputs:
        polvect: polarisation vector [DoP, AoP]
            DoP is magnitude [0, 1] (pixel intensity)
            AoP is angle [-pi/2, pi/2] (pixel colour)
    outputs:
        HSV image
    '''
    DoP, AoP = polvect
    AoP = np.rad2deg(AoP).astype(np.uint8)  # AoP in [0, 180]
    DoP = imgcvt(DoP)                       # DoP in [0, 255]

    HSV = np.zeros_like(hf.gs3c(DoP))
    HSV[:,:,0] = AoP # hue
    HSV[:,:,1] = 255 # saturation
    HSV[:,:,2] = DoP # value

    return HSV # return image in HSV (NB: if you want to imshow, need to do e.g. imshow(HSV, cv2.COLOR_HSV2BGR) or imshow(hsv2bgr(HSV))

def saturation_masking(pol_imgs, limits):
    '''Mask out saturated and unsaturated pixels in a four-channel polarisation image
    input: 
        pol_imgs - four-channel polarisation image to mask (dtype=uint8) <== DO NOT LEAVE THIS AS A UINT8
        limits - two element tuple of ints in the uint8 range, i.e. (0, 255) for strictly accurate under-/over-saturation
    output: masked_pol_imgs - masked four-channel polarisation image (dtype=masked_array)
    '''
    lower, upper = limits
    ## if any pixel in a cluster of 4 is saturated, the whole cluster is masked (affects DoP)
    # unsaturated_mask = areas where the intensity is not saturated (i.e. not 0 or 255)
    saturated_mask = np.zeros_like(pol_imgs[0])
    for pol_num, pol_img in enumerate(pol_imgs): # construct a mask which is True for all non-saturated values in all four pol images
        
        # unsaturated_mask = np.logical_or(pol_img > 0, pol_img < 255, *[unsaturated_mask] if pol_num else []) # conditional argument 3
        # saturated_mask = np.logical_or(pol_img == 0, pol_img == 255, *[saturated_mask] if pol_num else [])
        saturated_mask = np.logical_or.reduce((pol_img <= lower, pol_img >= upper, saturated_mask))
        # cv2.imshow('preview', hf.gs3c(unsaturated_mask.astype(uint8)))
        # cv2.waitKey(0)
    
    # saturated_mask = ~unsaturated_mask # inverting so false values are valid and vice versa (convention of masked arrays)
    masked_pol_imgs = [np.ma.masked_array(pol_img, saturated_mask, fill_value=0) for pol_img in pol_imgs]

    ''' masked array notes:
    <marrayname> will return only the valid entries of the array
    <marrayname>.data will return the unmasked data (i.e. regular, unmasked array)
    <marrayname>.mask will return the mask
    <marrayname>.filled() will return all data with invalid/masked entries replaced with fill_value
    '''

    return masked_pol_imgs

# def polarsens_to_cropped_pol_angles(img: np.ndarray, c_bounds: list[float] = AoI_Bounds): 
def polarsens_to_cropped_pol_angles(img, c_bounds=None, mask=True, mask_limits=(0, 255)): 
    '''converts a mono8 polarsens image (i.e every square of 4 pixels is 4 different polarisation angles) to a 4-channel image for
        each angle, cropped to some area of interest
        inputs:
            img: a mono8 polarsens image
            c_bounds: a list in [0, 1] (dtype=float) of x1 (left), x2 (right), y1 (top), y2 (bottom) values (proportion of image)
            mask: whether to mask out saturated and unsaturated pixels (dtype=bool)
        outputs: 
            cropped_angles: a list of 4 cropped images (each representing a different polarisation angle), which are masked if mask=True
    '''
    # c_bounds=[0.41, 0.58, 0.35, 0.55] # for wider view
    # c_bounds=[0.452, 0.5, 0.375, 0.4] # for small rectangle inside sample tray
    c_bounds = c_bounds if c_bounds else gvars.get_AoI_bounds() # dynamic default argument for c_bounds
    pol_angles = hf.pol_split(img)
    cropped_angles = np.array([hf.crop_to_proportions(angle, c_bounds) for angle in pol_angles])
    cropped_angles = saturation_masking(cropped_angles, mask_limits) if mask else cropped_angles

    return [angle.astype(np.float64) for angle in cropped_angles] # OTHERWISE IT'S AN UNSIGNED 8 BIT INTEGER - can you imagine!
    # pre_img = np.array(pol_angles[0]).astype(np.uint8)*255
    # cv2.imshow('rectangle preview', pre_img)
    # cv2.waitKey(0)
    # TODO: get pol_angles for circular polarisation

    # TODO: do the cropping at the camera level, rather than here (see Happy Snaps for why)

if __name__ == "__main__":
    # Uncomment to run GUI instead of main
    main()

# CODE GRAVEYARD
'''previous version of stokes_norm with a gain of 0.5 (it made sense at the time)'''
# def stokes_norm(pol, I):
#     # normalise stokes param (i.e. Q/I), and remove nan values
#     with warnings.catch_warnings(): # I know I'm dividing by 0, don't worry about it
#         warnings.simplefilter("ignore", category=RuntimeWarning)
#         # divide by 2 to get max magnitude in range 0:1
#         diff_norm = np.nan_to_num(posinf=1, neginf=-1, x=(0.5* 
#             pol / I # 'Q / I'
#         ))
#         return diff_norm

# def stokes_circular_old(stokes_vector, I_45_w):
    #     #TODO; need to load in previous version of corresponding linear image and recalculate (linear) stokes
#     I = stokes_vector[0]
#     V = I - (2 * I_45_w)
#     #TODO: normalise V to [-1,1], somewhere
#     #TODO: V -> image, somewhere
#     stokes_vector[3] = V
#     return stokes_vector