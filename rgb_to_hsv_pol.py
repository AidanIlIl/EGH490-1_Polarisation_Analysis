import cv2
import numpy as np
import warnings
import os
from tkinter import filedialog, Tk, messagebox

root = Tk()
root.withdraw() # Hide the root window

# Global variable for DoLP enhancement
dolp_enhancement = 10

def rgb_to_hsv_dolp_aolp(img):
    """
    Convert an RGB image (representing 0, 45, and 90 degree polarised intensities) to an HSV image illustrating DoP and AoP.
    input:
        img: RGB image where R=0, G=45, B=90 degree polarisation intensities
    output:
        HSV image illustrating DoP and AoP
    """
    if len(img.shape) != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image.")
    
    # Extract the RGB channels, ensuring correct polarization angles
    I_0   = img[:, :, 0].astype(np.float64)  # Red channel = 0 degrees
    I_45  = img[:, :, 1].astype(np.float64)  # Green channel = 45 degrees
    I_90  = img[:, :, 2].astype(np.float64)  # Blue channel = 90 degrees
    
    # Calculate Stokes parameters (assuming V=0)
    I = I_0 + I_90
    Q = I_0 - I_90
    U = 2 * I_45 - I  # Approximation, assuming I_45 is the average of I_0 and I_90 plus some U component
    
    # Calculate DoLP and AoLP
    DoLP = stokes_norm(np.sqrt(Q**2 + U**2), I)
    AoLP = np.arctan2(U, Q) / 2
    AoLP[AoLP < 0] += np.pi

    # Convert to HSV
    AoLP = np.rad2deg(AoLP).astype(np.uint8)
    DoLP = imgcvt(DoLP * dolp_enhancement) # Apply enhancement here

    HSV = np.zeros_like(img)
    HSV[:,:,0] = AoLP # hue
    HSV[:,:,1] = 255 # saturation
    HSV[:,:,2] = DoLP # value
    
    return HSV

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

imgcvt = lambda pol: (pol*255).astype(np.uint8)
'''converts a 2D array of floats in [0,1] to a 2D array of uint8s in [0,255]
    inputs:
        pol: 2D array of floats in [0,1]
    outputs:
        2D array of uint8s in [0,255]
'''

if __name__ == '__main__':
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        messagebox.showerror("Error", "No file selected.")
        exit()

    # Load the image
    image = cv2.imread(file_path)
    if image is None:
        messagebox.showerror("Error", "Could not read the image.")
        exit()
    # Convert the image to RGB format (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # this is needed to convert from BGR to RGB

    # Convert the RGB image to HSV with DoLP and AoLP
    hsv_img = rgb_to_hsv_dolp_aolp(image)

    # Save the HSV image to the same directory
    base_name, ext = os.path.splitext(file_path)
    output_filepath = f"{base_name}_POL-HSV.png"
    cv2.imwrite(output_filepath, cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR))
