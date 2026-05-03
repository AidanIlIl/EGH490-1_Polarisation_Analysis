import cv2
import numpy as np
import os
import sys
import re
import csv
import pandas as pd
import inspect
from tkinter import filedialog
from typing import List
try: import gvars
except: import src.gvars as gvars
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# import gvars

'''
Function Directory + Summaries

Image Processing:
- hsv2bgr(img) -> 3D numpy array of a BGR image: converts an HSV image to a BGR image (wrapper for cv2.cvtColor)
- match_img_size(img_to_change, img_to_match) -> 3D numpy array: resizes one image to match the dimensions of another
- pol_split(img) -> list of 2D numpy arrays: splits a polarized image into four images, each with a different polarization angle
- pol_preview(img) -> 2D numpy array: creates a preview of all polarization angles in a single image
- img_res(img, scale_percent) -> 2D numpy array: resizes an image by a percentage
- img_resize_aspect_ratio(img, dim, fill_colour=(0,0,0)) -> 2D numpy array: resizes an image to specific dimensions while maintaining aspect ratio
- sat_pct(img, thresh=0) -> tuple: calculates the percentage of saturated and undersaturated pixels in an image
- is_full_img_size(img) -> bool: checks if an image has the default full size
- crop_to_proportions(img, croppos=None) -> 2D numpy array: crops an image to a fraction of its size
- gs3c(img) -> 3D numpy array: converts a 2D grayscale image to a 3-channel grayscale image
- draw_aoi(img, colour=(255, 0, 0), thickness=5, filename='', proportions=True) -> 2D numpy array: draws the AoI on an image
- draw_dotted_line(img, pt1, pt2, color, thickness, gap) -> None: draws a dotted line on an image
- draw_hsv_colour_bar(img, x1, x2, y1, y2, annotate=False, text_colour=(255, 255, 255)) -> 2D numpy array: draws an HSV colour bar on an image

File and Directory Operations:
- image_number_search(folder, tstr, imformat='png') -> 2D numpy array: loads an image from a folder based on a test string
- load_images_from_folder(folder, imformat='png', verbose=False) -> list of 2D numpy arrays: loads images from a folder
- listdir_complete(path, key=None) -> list: lists all files in a directory with their full paths
- add_upper_directory_to_path() -> None: adds the parent directory to the system path
- file_path_clean_val(filepath) -> tuple: cleans and validates a file path string
- multi_directory_select(request_string, selected_directory_list, search_directory) -> list: recursively selects multiple directories
- f_write(filepath, string) -> None: appends a string to a file, creating the file if it doesn't exist
- find_duplicate_im_num_imgs(im_path) -> list: finds duplicate image numbers in a directory

Parameter and CSV Handling:
- param_construct(im_num, vals, params=gvars.param_fields) -> str: constructs a parameter string from a list of values
- param_extract(filepath, fields, dtypes) -> dict: extracts parameters from a filename
- param_extract_single(filepath, field, dtype) -> value: extracts a single parameter from a filename
- param_change(param_path, fields_to_change, new_values, param_fields=gvars.param_fields) -> str: modifies parameters in a parameter string
- csv_init(csvpath, fields) -> None: creates a new CSV file with headers
- csv_append(csvpath, data) -> None: appends a row of data to a CSV file
- csv_patch(csv_path, force_patch=False) -> bool: patches a CSV file with additional fields and updates image filenames

Camera and Obstruction Handling:
- camera_is_obstructed(azimuth, zenith, camera_zenith, camera_azimuth=gvars.CAMERA_AZIMUTH) -> bool: checks if the camera is obstructed by the light arm
- load_AoI_settings(csvpath=gvars.settings_file) -> bool: loads the most recent AoI bounds from a settings file
- AoI2Px(img_width, img_height) -> list: converts AoI proportions to pixel positions
- AoI2OffsetPx(width=gvars.img_width_default, height=gvars.img_height_default) -> list: converts AoI proportions to offset pixel positions and dimensions

Utilities:
- stats(arr) -> None: prints detailed statistics of a numpy array
- linestats(arr) -> None: prints basic statistics of a numpy array on the same line
- find_min_max_idxs(arr) -> tuple: finds the indices of the minimum and maximum values in a numpy array
- tracefunc(frame, event, arg, indent=[0]) -> callable: debugging tool to trace function calls and exits
- round_up_to_num(to_round, num) -> int: rounds a number up to the nearest multiple of another number
- round_down_to_num(to_round, num) -> int: rounds a number down to the nearest multiple of another number
- script_print(print_string, end="\n") -> None: prints the script name and a message
- function_print(print_string, end="\n") -> None: prints the script and function name along with a message
- pop_multiple(list_to_pop, pop_idxs) -> list: removes multiple elements from a list by indices
- tqdmp(*args, **kwargs) -> generator: wrapper for tqdm that redirects print to tqdm.write

'''

hsv2bgr = lambda img: cv2.cvtColor(img, cv2.COLOR_HSV2BGR) # HSV to BGR
'''converts an HSV image to a BGR image
    input: img: 3D numpy array of an HSV image
    output: 3D numpy array of a BGR image
'''

linestats = lambda arr: print(f"shape: {np.shape(arr)}, min: {np.min(arr)}, max: {np.max(arr)}, mean: {np.mean(arr)}, dtype: {arr.dtype}")
'''prints basic statistics of a numpy array on the same line
    input: arr: numpy array
'''

match_img_size = lambda img_to_change, img_to_match: cv2.resize(img_to_change, (img_to_match.shape[1], img_to_match.shape[0]))
'''changes one image to have the same dimensions as another image
    inputs:
        img_to_change: the image to be resized
        img_to_match: the image that sets the size
    usage: img_to_change = match_img_size(img_to_change, img_to_match)
'''

find_min_max_idxs = lambda arr: (np.unravel_index(np.argmin(arr), arr.shape), np.unravel_index(np.argmax(arr), arr.shape))

## longer functions
polarsens_array = np.array([[0  , 45], 
                            [135, 90]]) # array of the four polarization angles in the order they're stored in the image

def pol_split(img): 
    '''splits a polarized image into four images, each with a different polarization angle
    input: img: 2D numpy array of a polarized image
    output: list of 4 2D numpy arrays (polarized images at 0, 45, 90, 135 degrees)
    '''

    # splits a polarized image into four images, each with a different polarization angle
    # python array indexing in 2d is array[y, x]=array[row, column], starting from the top left
    
    # assuming:   0, 45     (calibration image assumption)
    #             135, 90:
    # deg0 = img[0::2, 0::2] # top left
    # deg45 = img[0::2, 1::2] # top right
    # deg90 = img[1::2, 1::2] # bottom right
    # deg135 = img[1::2, 0::2] # bottom left

    # assuming: 0, 45   (first version)
    #          90, 135:
    # deg0 = img[0::2, 0::2] # top left
    # deg45 = img[0::2, 1::2] # top right
    # deg90 = img[1::2, 1::2] # bottom right
    # deg135 = img[1::2, 0::2] # bottom left
    
    # assuming: 90, 45  (lucid vision guide)
    #          135, 0
    
    # TODO: split positions automatically per polarsens_array
    deg0 = img[1::2, 1::2] # bottom right
    deg45 = img[0::2, 1::2] # top right
    deg90 = img[0::2, 0::2] # top left
    deg135 = img[1::2, 0::2] # bottom left
    
    # ROTATED CCW 90 DEGREES: 45, 0
                            # 90, 135

    return np.array([deg0, deg45, deg90, deg135])

def pol_preview(img):
    # display preview of (all of the) polarization angles
    # 				90, 45
    # 				135, 0
    pol_angles = pol_split(img) # [0d, 45d, 90d, 135d]
    sample_pol_angle = pol_angles[0]
    
    pol_text = ['0d', '45d', '90d', '135d']
    # get [x, y] positions to put an element of pol_text on each corner of sample_pol_angle (using cv2.putText) based on its shape
    tl = (0, 70)
    tr = (sample_pol_angle.shape[1]-200, 70)
    bl = (0, sample_pol_angle.shape[0]-50)
    br = (sample_pol_angle.shape[1]-200, sample_pol_angle.shape[0]-50)
    pol_text_pos = [br, tr, tl, bl]
    
    for angle, text_pos, text in zip(pol_angles, pol_text_pos, pol_text): # display which angle this is
        angle = cv2.putText(angle, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    top = np.hstack((pol_angles[2], pol_angles[1]))     # 90, 45
    bottom = np.hstack((pol_angles[3], pol_angles[0]))  # 135, 0
    img = np.vstack((top, bottom))

    # put a white line through the middle of the image in both axes to separate the quadrants
    img = cv2.line(img, (0, img.shape[0]//2), (img.shape[1], img.shape[0]//2), (0, 0, 0), 100)
    img = cv2.line(img, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0, 0, 0), 100)

    return img

def img_res(img, scale_percent):
    '''resizes an image by a percentage
    input: img: 2D numpy array of an image
           scale_percent: integer percentage to scale the image by
    output: 2D numpy array of the resized image
    '''
    if not 0 < scale_percent <= 100: raise Exception("must have 0 < scale_percent <= 100")
    if scale_percent == 100: return img
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def img_resize_aspect_ratio(img, dim, fill_colour=(0,0,0)):
    '''
    resize an image to a specific dimension while maintaining its aspect ratio
    the image will be centered on a canvas of the specified dimensions and the shortfall will be filled with a specified colour
    inputs: 
        img: 3D numpy array of the image (if a 2D array is passed, it will be gs3c'd)
        dim: tuple of the desired dimensions (width, height)
        fill_colour: 3-tuple of the colour to fill the canvas with
    outputs:
        2D numpy array of the resized image
    '''
    if len(img.shape) == 2 or img.shape[2]==1: img = gs3c(img)
    # get the dimensions of the image
    img_height, img_width = img.shape[:2]
    new_width, new_height = dim
    # calculate the aspect ratio of the image
    img_aspect_ratio = img_width / img_height
    new_aspect_ratio = new_width / new_height
    
    # if the image is wider than it is tall
    if img_aspect_ratio > new_aspect_ratio:
        # calculate the new height based on the new width and the aspect ratio
        new_height = int(new_width / img_aspect_ratio)
    # if the image is taller than it is wide
    else:
        # calculate the new width based on the new height and the aspect ratio
        new_width = int(new_height * img_aspect_ratio)

    # resize the image to the new dimensions
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # create a canvas of the desired dimensions
    canvas = np.full((dim[1], dim[0], img.shape[2]), fill_colour, dtype=np.uint8)
    # calculate the position to place the resized image on the canvas
    x_offset = (dim[0] - new_width) // 2
    y_offset = (dim[1] - new_height) // 2
    # place the resized image on the canvas
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_img
    return canvas

def sat_pct(img, thresh=0):
    '''saturation percentage: gets the percentage of a (uint8) image with satured pixel values (i.e. % of pixels that are 255)
    input: image
    output: 
        saturation: the percentage of saturated pixels (that are 255)
        undetected: the percentage of empty pixels (that are <= 'thresh')
    '''
    oversat = np.sum(img == 255) / np.size(img) * 100
    undersat = np.sum(img <= thresh) / np.size(img) * 100
    return oversat, undersat

is_full_img_size = lambda img: img.shape == (gvars.img_height_default, gvars.img_width_default) or img.shape == (gvars.img_height_default//2, gvars.img_width_default//2)
def crop_to_proportions(img, croppos=None):
    '''Crop an image to a fraction of its size
    inputs:
        img: image to crop
        croppos: [x1, x2, y1, y2] where x1, x2 are the left and right bounds, and y1, y2 are the top and bottom bounds
    outputs:
        cropped image
    '''
    croppos = croppos if croppos is not None else gvars.get_AoI_bounds() # dynamic default argument
    if not is_full_img_size(img): return img # already cropped
    
    # crop image to a fraction of its size
    x1, x2, y1, y2 = croppos
    cropped = img[int(y1*img.shape[0]):int(y2*img.shape[0]), int(x1*img.shape[1]):int(x2*img.shape[1])]
    # pre_img = np.array(cropped[0]).astype(np.uint8)*255
    # cv2.imshow('rectangle preview', pre_img)
    # cv2.waitKey(0)
    return cropped

def stats(arr): 
    '''prints basic statistics of a numpy array
        input: arr: numpy array
    '''
    print("shape: {}".format(np.shape(arr)))
    print("min: {}".format(np.min(arr)))
    print("max: {}".format(np.max(arr)))
    print("mean: {}".format(np.mean(arr)))
    print("median: {}".format(np.median(arr)))
    print("std: {}".format(np.std(arr)))
    print("type: {}".format(arr.dtype))

def f_write(filepath, string):
    '''appends a string to a file
        input: filepath: path to the file
               string: string to append to the file
    '''
    # create file if it doesn't exist
    if not os.path.exists(filepath):
        open(filepath, 'w').close()
    # append string to file (without overwriting its contents)
    with open(filepath, 'a') as f:
        f.write(string)

# HELPER FUNCTIONS
def gs3c(img):
    '''grayscale to 3-channel: converts a 2D grayscale image to a 3-channel grayscale image
    input: img: 2D numpy array (or 3D with the last axis being 1) of a grayscale image
    output: 3D numpy array of a 3-channel grayscale image
    '''
    # validation: ensure the image either only has 2 axes, or it has 3 axes but the last one is 1
    assert len(np.shape(img)) == 2 or (len(np.shape(img)) == 3 and np.shape(img)[2] == 1), "image array must be 2D, or 3D with the last axis being 1"

    # if the image array has 3 axes, but the last one is 1, remove the last axis
    if len(np.shape(img)) == 3 and np.shape(img)[2] == 1: img = img[:,:,0]
    
    # if the image array only has 2 axes, add a third axis to make it a 3-channel image
    return np.stack((img,)*3, axis=-1)

def image_number_search(folder, tstr, imformat='png'):
    '''loads an image from a folder
    input: folder: path to the folder containing the image
              tstr: test string (first two characters/a number) to match the image to
              imformat: format of the image to load
    output: 2D numpy array of the image
    '''
    for filename in os.listdir(folder):
        if tstr in filename[:2] and imformat in filename: # match first two characters of filename to test number
            img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
            # if verbose: print("loading {}".format(filename))
            return img
    raise Exception("no image found")

def load_images_from_folder(folder, imformat='png', verbose=False):
    '''loads images from a folder
    input: folder: path to the folder containing the images
           imformat: format of the images to load
           verbose: boolean to print status messages
    output: list of 2D numpy arrays of the images
    '''
    # imgs, parameters = [], []
    imgs = []
    if verbose:
        print('loading test images...')
        print("looking in {}".format(folder))
    print(sorted(os.listdir(folder)))
    for filename in sorted(os.listdir(folder)):
        if imformat in filename:
            img = cv2.imread(os.path.join(folder,filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs.append(img)
                # parameters.append(param_extract(filename))
    if imgs == []:
        raise Exception("no images found")
    return imgs#, parameters

def param_construct(im_num:int, vals, params=gvars.param_fields) -> str:
    '''
    Construct a parameter string from a list of values
    inputs: 
        im_num: the image number
        vals: a list of the values of the parameters (or a dictionary, if that dictionary has the same keys as params)
        params: the names of the parameters
    '''
    ## validation
    # if vals is a dictionary, check that its key match params and convert it to a list if so
    if isinstance(vals, dict): 
        # check that all keys match all params
        if not all([param in vals.keys() for param in params]): raise Exception(f'keys in vals do not match params: {vals.keys()} vs {params}')
        vals = [vals[param] for param in params] # convert the dictionary to a list
    elif isinstance(vals, list): # if vals is not a dictionary, check that the number of values matches the number of parameters
        if len(vals) != len(gvars.param_fields): raise Exception(f"{len(vals)} values provided (expected {len(params)})")
        # if it's not a dictionary and params is the default, check that the vals are the right type
        # if params == vars.param_fields: # validate that the data is the right type (with reference to vars.dtypes) for each val
        #     for val, param, dtype in zip(vals, params, vars.param_dtypes): 
        #         if not isinstance(val, dtype): raise Exception(f"{param}: expected {dtype} but got {type(val)}")
    else: raise Exception(f"vals must be a list or dictionary, not {type(vals)}")
        
    paramstr = f"{im_num}"
    for val, param in zip(vals, params): paramstr = f"{paramstr}_{param}:{val}"
    return paramstr

def param_extract(filepath: str, fields: List[str] = gvars.param_fields, dtypes=gvars.param_dtypes) -> dict:
    '''
    Extract the image number, azimuth, zenith, exposure, ISO, and whether it's raw or HSV from a param string
        example image names: 
            '1_cze:0_az:0_ze:0_exp:0_ISO:0_RAW.png' (Linux)
            '1_cze_0_az_0_ze_0_exp_0_ISO_0_HSV.png' (Windows)
            ***but the paramstr doesn't have a file extension - remove if present***
        param string format: f"{im_num}_cze:{int(camera_zenith)}_az:{azimuth}_ze:{zenith}_exp:{exposure}_ISO:{gain}"
    inputs: 
        filepath: the path to the image file (EXCLUDING the extension)
        fields: the fields to extract from the image name
        dtypes: the data types of the fields to extract from the image name
    outputs: params, a dictionary containing the extracted parameters
    '''
    # TODO: validate that the parameters are present and in the expected format
    params = {}
    
    # remove any file extensions, if present: 
    # # if '.' is within 4 characters of the end of the filename [i.e. 'string.jpeg'], but not within 2 [i.e. 'ISO:4.55'], remove it and everything after it
    # if f_name.rfind('.') > len(f_name) - 5: f_name = f_name[:f_name.rfind('.')]
    
    # remove RAW or HSV at the end, if present
    if filepath.endswith('RAW') or filepath.endswith('HSV'): filepath = filepath[:-3]
    
    # remove any file pathing, if present
    if '/' in filepath: filepath = filepath[filepath.rfind('/')+1:]
   
    
    filepath = filepath.replace(':', '_') # windows doesn't like colons in filenames
    filepath = filepath.replace('\uf03a', '_') # unicode character for ':'
    
    # get the image number from the filename
    params['im_num'] = filepath[:filepath.find('_')]
    # the below code works, but not for getting the im_num or type
    for field, dtype in zip(fields, dtypes):
        '''
        if any field is not surrounded by '_' characters, it is a substring of another
        field (see: 'ze', 'cze'). if this happens, find the one surrounded by '_' characters
        '''
        # find all instances of the field in the filename
        substr_idxs = [m.start() for m in re.finditer(field, filepath)]
        if len(substr_idxs) == 0: # field not found
            raise Exception(f"Could not find {field} in filename: {filepath}")
        if len(substr_idxs) > 1: # field is a substring of another field
            start = substr_idxs[0] # default start: first instance of the field
            for substr_idx in substr_idxs:
                # if a field starting at substr_idx is surrounded by underscores, make `start` equal that
                if filepath[substr_idx-1] == '_' and filepath[substr_idx+len(field)] == '_':
                    start = substr_idx
        else: start = substr_idxs[0] # field is not a substring of another field
        
        ## replace the next underscore after start with a colon
        u_idx = filepath.find('_', start) #TODO: 'referenced before assignment'
        # replace u_idx in f_name with a colon (to make it easier to extract the value)
        filepath = filepath[:u_idx] + ':' + filepath[u_idx+1:]
        # find the end of the field in the filename
        end = filepath.find('_', start)
        if end == -1:
            end = len(filepath)
        # extract the value of the field from the filename
        value = filepath[start+len(field)+1:end]
        # convert the value to the appropriate type (rather than a string)
        
        # if dtype == float: # if float, don't 
        #     value = np.round(dtype(value),2)
        # else: value = dtype(value)
        value = dtype(value)
        
        # add the value to the dictionary
        params[field] = value
    return params #NB: all of the values are strings

param_extract_single = lambda filepath, field, dtype: param_extract(filepath, [field], [dtype])[field] 
'''extract a single parameter from a filename given the field and dtype'''

def param_change(param_path, fields_to_change, new_values, param_fields=gvars.param_fields):
    '''change the params in a paramstr, then reconstruct it and return it
    paramstr = f"{im_num}_cze:{int(camera_zenith)}_az:{azimuth}_ze:{zenith}_exp:{exposure}_ISO:{gain}"
    img_name = f"{idir}/{paramstr}_RAW"

    inputs: paramstr: the string of parameters
            fields: the fields to extract from the image name
    outputs: new_param_path: the new path with the changed parameters (no 'RAW', 'HSV', or file extension - must be added back separately)
    '''
    # if paramstr is `anything/at/all/paramstr`, remove everything before the last '/' (and store it in a variable)
    dir_str = ''
    if '/' in param_path: 
        param_str = param_path[param_path.rfind('/')+1:]
        dir_str = param_path[:param_path.rfind('/')] # +1 includes the '/'
    else: param_str = param_path

    #TODO:  extension handling (shouldn't come up?)

    # extract the parameters from the paramstr
    params = param_extract(param_str, param_fields)

    # change the fields_to_change
    if not isinstance (fields_to_change, list): fields_to_change = [fields_to_change]
    if not isinstance (new_values, list): new_values = [new_values]
    for field, new_value in zip(fields_to_change, new_values):
        if field in ['exp', 'iso']: new_value = round(float(new_value), 1)
        params[field] = new_value
    
    # reconstruct the paramstr
    # TODO: just give that function im_num and the actual dictionary (might need to modify the param_construct function)
    new_param_str = param_construct(params['im_num'], [params['cze'], params['az'], params['ze'], params['exp'], params['ISO'], params['obs']])
    # new_param_str = f"{params['im_num']}_cze:{params['cze']}_az:{params['az']}_ze:{params['ze']}_exp:{params['exp']}_ISO:{params['ISO']}"
    new_param_path = os.path.join(dir_str, new_param_str)
    return new_param_path

def csv_init(csvpath, fields):
    ''' creating a new csv file with headers 
        inputs:
            csvpath is the path to the existing csv file (with or without a .csv extension)
            fields are the comma-separated headings to start the new file with
        outputs: 
            appends a new line of data to the csv file
    '''
    # ensure csvpath has a .csv extension
    if csvpath[-4:] == '.csv': csvpath = csvpath[:-4] # trim .csv off if it was passed; add .csv next line
    with open(f"{csvpath}.csv", 'w') as f: # 'w' mode overwrites the file if it exists
        writer = csv.writer(f)
        writer.writerow(fields)

def csv_append(csvpath, data):
    ''' append a row of data to the csv file 
        inputs:
            csvpath is the path to the existing csv file (with or without a .csv extension)
            data are the comma-separated values to append to the existing csv file
        outputs: 
            appends a new line of data to the csv file
    '''
    # ensure csvpath has a .csv extension
    if csvpath[-4:] == '.csv': csvpath = csvpath[:-4] # if .csv was passed, trim it off; regardless, add .csv next line
    with open(f"{csvpath}.csv", 'a') as f: # 'a' mode appends to the file if it exists
        writer = csv.writer(f)
        writer.writerow(data)

def csv_patch(csv_path, force_patch=False): # also in utilities/fix_obstruction.py
    ''' 
    inputs:
        csv_path: the path to the csv file to fix obstructions
        force_patch: boolean to force the patch even if the columns are already present (Default False) (tbd)
    csv changes: 
        "Camera_Obstruction" field at the end
        "Gain_db" field immediately after "Exposure_Time_us"
        "Plane_Angle_deg" field immediately after "Phase_Angle_deg"
        "I_rotated, Q_rotated, U_rotated, V_rotated, DoLP_rotated, AoLP_rotated" fields at end (linear only)
    usage: on a csv file within a test directory, or on a csv file within a directory containing other test summary csv files
    '''
    # variables to handle the case where the csv file doesn't exist
    csv_exists = os.path.exists(csv_path)
    test_dir = os.path.dirname(csv_path)
    
    lin_test = gvars.test_is_linear(test_dir)
    circ_test = gvars.test_is_circular(test_dir)
    if lin_test and circ_test: raise ValueError("test must be either linear or circular, not both - run utilities/enforce_file_structure.py to update test directory/ies")
    if not lin_test and not circ_test: raise ValueError("no linear or circular data found")
    # if csv_exists and os.path.basename(csv_path) != gvars.get_test_summary(test_dir): raise ValueError(f"csv file should be named {gvars.get_test_summary(test_dir)} (actually named {os.path.basename(csv_path)})")
    
    # linear and circular csv fields live in vars.py, in linear_csv_fields and circular_csv_fields
    df = None
    if csv_exists: 
        df = pd.read_csv(csv_path)
    
        new_columns = ['Plane_Angle_deg', 'Camera_Obstruction', 'Gain_db']
        # if all the new columns are already in the dataframe (and we're not forcing a patch), return False
        if all([col in df.columns for col in new_columns]) and not force_patch: return False

        # added a "Gain_db" field immediately after "Exposure_Time_us"
        if 'Gain_db' not in df.columns: # no force_patch condition because we *only* want to add it if it's not there (no calcs necessary)
            exposure_col_idx = df.columns.get_loc('Exposure_Time_us')
            df.insert(exposure_col_idx+1, 'Gain_db', 0.0) # gain is 0.0 if it's not present

        # added a "Camera_Obstruction" field at the end (and an 'obs:<val>' param in the filename)
        if 'Camera_Obstruction' not in df.columns or force_patch: # default obstruction is 0
            if force_patch and 'Camera_Obstruction' in df.columns: df.drop(columns=['Camera_Obstruction'], inplace=True)  # remove the column if it exists (to avoid duplicate columns)
            df.insert(len(df.columns), 'Camera_Obstruction', 0, allow_duplicates=False) # add the "Camera_Obstruction" field to the dataframe as int64
            ## fix camera obstruction - for each row of df, if az_exclude_range_min <= azimuth <= az_exclude_range_max and zen_exclude_range_min <= zenith <= zen_exclude_range_max, set obs = 1
            df['Camera_Obstruction'] = df.apply(lambda row: 1 if camera_is_obstructed(row['Azimuth_deg'], row['Zenith_deg'], row['Camera_Zenith_deg']) else row['Camera_Obstruction'], axis=1).astype('int64')

            ## we also want to rename the image files to have obs:1 rather than obs:0 (moved to below)

        if 'Plane_Angle_deg' not in df.columns or force_patch:
            try: import polarisation_calcs as pc
            except: import src.polarisation_calcs as pc
            phase_col_idx = df.columns.get_loc('Phase_Angle_deg')
            # if the 'Plane_Angle_deg' column is present, remove it (in the force_patch case)
            if force_patch and 'Plane_Angle_deg' in df.columns: df.drop(columns=['Plane_Angle_deg'], inplace=True)
            # add the 'Plane_Angle_deg' column to the dataframe
            df.insert(phase_col_idx+1, 'Plane_Angle_deg', df.apply(lambda row: pc.plane_azimuth_angle(row['Camera_Zenith_deg'], row['Azimuth_deg'], row['Zenith_deg']), axis=1))

        # Save the updated dataframe
        df.to_csv(csv_path, index=False, na_rep='nan')

        # added a "I_rotated, Q_rotated, U_rotated, V_rotated, DoLP_rotated, AoLP_rotated" fields at end (linear only)
        if 'I_rotated' not in df.columns: # or force_patch:
            try: from scatter_plane_transform import csv_add_rotation_correction
            except: from src.scatter_plane_transform import csv_add_rotation_correction
            csv_add_rotation_correction(csv_path)
    
    ## we also want to rename the image files to have obs:1 rather than obs:0; image files have a number at the beginning of the filename which corresponds to the row number in the csv-1 (i.e. it starts at 1 rather than 2)
    # check if it's a linear or circular test based on the csv filename:
    if (df is not None and 'Camera_Obstruction' in df.columns) or not csv_exists or force_patch: # 'df' in the conditional so it doesn't get caught on the <string> in df.columns if df is None
        if lin_test: imgpaths = gvars.get_lin_path_strs(test_dir)
        elif circ_test: imgpaths = gvars.get_circ_path_strs(test_dir)
        else: raise ValueError("No linear or circular data found (this error should not be triggerable as this condition is checked earlier...)")

        # if imgpaths[1] exists and contains an empty directory: delete it from the filesystem
        cpihsv_contents = sorted(os.listdir(imgpaths[1])) if os.path.exists(imgpaths[1]) else None
        if cpihsv_contents:
            # check if there is a directory in here
            hypothetical_empty_directory = os.path.join(imgpaths[1], cpihsv_contents[0])
            if os.path.exists(hypothetical_empty_directory) and os.path.isdir(hypothetical_empty_directory) and not os.listdir(hypothetical_empty_directory): os.rmdir(hypothetical_empty_directory)
        else: imgpaths.pop(1) # remove imgpaths[1] if it doesn't exist in the file system (it's not needed)

        # for each image in the image folder, if the image number corresponds to a row in the csv where obs = 1, rename the part of the image filename that has 'obs:<old_obs>' to 'obs:1' (or obs_<old_obs> to obs_1)
        # df is 0-indexed, image numbers are 1-indexed, csv is 2-indexed
        #   to get the df-image correspondence: subtract 1 from the image number
        #   to check the csv row:               add 1 to the image number

        # double check that the image paths exist before modifying images (i.e. can modify the csv without modifying the images if its standalone)
        if not all([os.path.exists(imgpath) for imgpath in imgpaths]): function_print("Modifying image filenames for camera obstruction: image paths not found, skipping image renaming")
        else: # it's a csv with images nearby
            get_img_num = lambda imgname: int(imgname.split('_')[0]) # get the image number from the filename as an integer (for sorting purposes)
            for imgpath in sorted(imgpaths):
                print(f"Fixing obstructions in {imgpath}")
                for imgname in sorted(os.listdir(imgpath), key=get_img_num):
                    # going from windows to linux, some characters (':', '-')(?) get converted to '', which is parsed as '\uf03a' (unicode) - this is a problem for the regex search, so replace it with '_'
                    imgname = imgname.replace('\uf03a', '_')
                    
                    new_imgname = imgname
                    img_num = get_img_num(imgname)

                    # if the image is from an older test and doesn't have 'obs' in the filename (after 'ISO_<float>_' or 'ISO:<float>'), add it to the new_imgname string
                    if '_obs' not in imgname:
                        if 'ISO_' in imgname: 
                            iso_value = re.search(r'ISO_(\d+\.\d+)', imgname).group(1)
                            new_imgname = new_imgname.replace(f'ISO_{iso_value}', f'ISO_{iso_value}_obs_0')
                        elif 'ISO:' in imgname: 
                            iso_value = re.search(r'ISO:(\d+\.\d+)', imgname).group(1)
                            new_imgname = new_imgname.replace(f'ISO:{iso_value}', f'ISO:{iso_value}_obs:0')
                        else: raise ValueError(f"Image {imgpath} does not contain 'ISO_' or 'ISO:'")

                    # update the image name to have the correct obstruction value (obs:0 or obs:1)
                    ze_param, az_param, cze_param = gvars.param_details[gvars.ZE], gvars.param_details[gvars.AZ], gvars.param_details[gvars.CZE]
                    zenith = param_extract_single(imgname, ze_param[0], ze_param[1])
                    azimuth = param_extract_single(imgname, az_param[0], az_param[1])
                    c_zenith = param_extract_single(imgname, cze_param[0], cze_param[1])
                    if camera_is_obstructed(azimuth, zenith, c_zenith): 
                        new_imgname = new_imgname.replace('obs:0', 'obs:1') # may have a different syntax
                        new_imgname = new_imgname.replace('obs_0', 'obs_1')
                    else: 
                        new_imgname = new_imgname.replace('obs:1', 'obs:0')
                        new_imgname = new_imgname.replace('obs_1', 'obs_0')
                        
                    # if the image name has changed, rename the image
                    if imgname != new_imgname:
                        os.rename(os.path.join(imgpath, imgname), os.path.join(imgpath, new_imgname))
                        print(f"Renamed\t{imgname} to \n\t{new_imgname}")
                print('')

    return True # the csv (or the test's images) was/were patched

def camera_is_obstructed(azimuth, zenith, camera_zenith, camera_azimuth=gvars.CAMERA_AZIMUTH):
    '''
    Check if the camera is obstructed by the light arm
    inputs: 
        azimuth: the azimuth of the light
        zenith: the zenith of the light
        camera_zenith: the zenith of the camera
        CAMERA_AZIMUTH: the azimuth of the camera arm (defaults to the constant in gvars.py)
    outputs:
        True if the camera is obstructed, False otherwise
    '''
    az_exclude_range_min  =  camera_azimuth - gvars.az_block_range
    az_exclude_range_max  =  camera_azimuth # if it's to the other side of the camera, the arm holding the light will block the camera
    zen_exclude_range_min =  camera_zenith - gvars.zen_block_range #9 degrees above the camera
    zen_exclude_range_max =  camera_zenith + gvars.zen_block_range #9 degrees below the camera
    return az_exclude_range_min <= azimuth <= az_exclude_range_max and \
           zen_exclude_range_min <= zenith <= zen_exclude_range_max

def load_AoI_settings(csvpath=gvars.settings_file):
    '''
    Load the most recent AoI bounds from a settings file
    inputs: csvpath: path to the settings file (with or without .csv)
    outputs: True (if it found a settings file and set AoI bounds) / False (otherwise)
    '''
    if csvpath[-4:] == '.csv': csvpath = csvpath[:-4] # trim .csv off if it was passed; add .csv next line
    if os.path.exists(f'{csvpath}.csv'):
        settings_df = pd.read_csv(f'{csvpath}.csv')
        gvars.AoI_bounds = settings_df.iloc[-1][gvars.settings_fields[2:]].values
        return True
    else:
        return False
        
AoI2Px = lambda img_width, img_height: [int(gvars.x1()*img_width), int(gvars.x2()*img_width), int(gvars.y1()*img_height), int(gvars.y2()*img_height)]
def AoI2OffsetPx(width=gvars.img_width_default, height=gvars.img_height_default):
    '''
    convert AoI proportions to offset pixel positions (for x1, y1) and width/height (for x2, y2)
    inputs: img: image to get the pixel positions from (default is a blank image of the max dimensions)
    outputs: OffX, OffY, width, height: the pixel positions and dimensions of the AoI
    '''
    OffX, x2, OffY, y2 = AoI2Px(width, height)
    width, height = x2-OffX, y2-OffY
    # match the order of camera.img_bounds_nodes
    return [width, height, OffX, OffY] # set width + height (this adjusts OffX + OffY max values)

def find_duplicate_im_num_imgs(im_path):
    '''finds if an image number is already in a directory
    input: im_path: path to an image with an image number, in a directory with other images
    output: a list of filenames containing the image number
    '''
    im_dir = os.path.dirname(im_path)
    im_num = param_extract(im_path)['im_num']
    duplicate_files = [f for f in os.listdir(im_dir) if param_extract(f)['im_num'] == im_num] # find duplicates
    duplicate_files = [os.path.join(im_dir, f) for f in duplicate_files] # add the directory to the filenames
    return duplicate_files

# https://stackoverflow.com/questions/8315389/how-do-i-print-functions-as-they-are-called
def tracefunc(frame, event, arg, indent=[0]):
      '''Debugging tool: prints to the console (in tree format) when a function is entered or exited
      Usage: call "sys.setprofile(hf.tracefunc)" at the point after which you want this debugging output'''
      if event == "call":
          indent[0] += 2
          print("-" * indent[0] + "> call function", frame.f_code.co_name)
      elif event == "return":
          print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
          indent[0] -= 2
      return tracefunc

'''rounds a number 'to_round' up to 'num' and returns it as an integer'''
round_up_to_num = lambda to_round, num: int(to_round + num-(to_round % num)) 
'''rounds a number 'to_round' down to 'num' and returns it as an integer'''
round_down_to_num = lambda to_round, num: int(to_round + num-(to_round % num)) 

'''script_print prints the name of the script that called it in front of the string to print (i.e. "<script_name>.py: <print_string>")'''
script_print = lambda print_string, end="\n": print(f'{inspect.stack()[1][1].split("/")[-1]}: {print_string}', end=end) # prints the name of the script that called it
'''function_print prints the name of the script and function that called it in front of the string to print (i.e. "<script_name>.py: <function_name> - <print_string>")'''
function_print = lambda print_string, end="\n": print(f'{inspect.stack()[1][1].split("/")[-1]}: {inspect.stack()[1][3]} - {print_string}', end=end) # prints the name of the function that called it
print()

def pop_multiple(list_to_pop, pop_idxs):
    '''pops multiple elements from a list "pop_list" given indices "idxs"'''
    if not isinstance(list_to_pop, list)                  : raise Exception('pop_list must be a list object')
    if not isinstance(pop_idxs, list)                     : raise Exception('idxs must be a list of integers')
    if not all([isinstance(idx, int) for idx in pop_idxs]): raise Exception('idxs must be a list of integers')
    if len(set(pop_idxs)) != len(pop_idxs)                : raise Exception('idxs must not have any duplicate values')

    # grab the values to return from the list prior to popping them (so they're returned in the correct order - see later step)
    popped_items = [list_to_pop[idx] for idx in pop_idxs]
    # pop in reverse sort order of the pop_idxs so that as list_to_pop vals are removed, they don't interfere with later removals
    [list_to_pop.pop(idx) for idx in sorted(pop_idxs, reverse=True)]
    return popped_items

def file_path_clean_val(filepath):
    '''clean up and validate a file path string
    inputs: filepath: path to the file
    outputs: 
        filepath: cleaned up filepath string
        does_exist: boolean indicating whether the file path exists
    '''
    if not isinstance(filepath, str): raise TypeError(f'filepath must be a string')
    # clean up filepath string
    # remove leading/trailing whitespace
    filepath = filepath.strip() 
    # if filepath has spaces in it, enclose it in quotes
    if ' ' in filepath: filepath = f'"{filepath}"'
    does_exist = os.path.exists(filepath)
    return filepath, does_exist

def multi_directory_select(request_string: str, selected_directory_list: list, search_directory):
    directory_path_string = filedialog.askdirectory(initialdir=search_directory, title=request_string)
    if len(directory_path_string) > 0:
        selected_directory_list.append(directory_path_string)
        multi_directory_select('Select the next Directory or Cancel to end', 
                                selected_directory_list,
                                os.path.dirname(directory_path_string))
    return selected_directory_list

def draw_aoi(img, colour=(255, 0, 0), thickness=5, filename='', proportions=True):
    '''draws the AoI on an image
    input: img: 2D numpy array of an image
    output: 2D numpy array of the image with the AoI drawn on it
    '''
    # get AoI bounds (e.g. vars.x1()) in px (e.g. x1)
    x1, x2 = [int(gvars.x1()*img.shape[1]), int(gvars.x2()*img.shape[1])]
    y1, y2 = [int(gvars.y1()*img.shape[0]), int(gvars.y2()*img.shape[0])]
    # draw the AoI on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness)
    if proportions:
        ## show AoI values next to each edge of the rectangle: 
        horz_offset, vert_offset = 70, 30# offset values in px to move text outside the rectangle
        X1pos = (x1-horz_offset , 	y1 + (y2-y1)//2)	# vars.X1 between [x1,y1] and [x2, y1] left of the rectangle edge,
        X2pos = (x2				, 	y1 + (y2-y1)//2) 	# vars.X2 between [x1,y2] and [x2, y2] right of the rectangle edge,
        Y1pos = (x1 + (x2-x1)//2, 	y1-10) 				# vars.Y1 between [x1,y1] and [x1, y2] above the rectangle edge,
        Y2pos = (x1 + (x2-x1)//2, 	y2+vert_offset) 	# vars.Y2 between [x2,y1] and [x2, y2] below the rectangle edge
        
        cv2.putText(img, f"{gvars.x1():.2f}", X1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"{gvars.x2():.2f}", X2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"{gvars.y1():.2f}", Y1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
        cv2.putText(img, f"{gvars.y2():.2f}", Y2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
    if filename != '':
        cv2.putText(img, filename, (10, img.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2, cv2.LINE_AA)
    return img

def draw_dotted_line(img, pt1, pt2, color, thickness, gap):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))
    for p in range(len(pts) - 1):
        if p % 2 == 0:
            cv2.line(img, pts[p], pts[p + 1], color, thickness)

def draw_hsv_colour_bar(img, x1, x2, y1, y2, annotate=False, text_colour=(255, 255, 255)):
    '''
    draw a colour bar on an image 'img' to show the HSV colour space
    vertically from 0-255 value, horizontally from 0-180 hue, always 255 saturation
    inputs: 
        img: 2D numpy array of the image to draw the colour bar on
        x1, x2, y1, y2: the pixel positions of the top left and bottom right corners of the colour bar
        annotate: boolean to add value ranges to the corners of the colour bar
    outputs:
        2D numpy array of the image with the colour bar drawn on it
    '''
    # create a blank image to draw the colour bar on
    hsv_colour_bar = np.zeros((256, 181, 3), dtype=np.uint8)
    # draw the colour bar
    for x in range(181):
        for y in range(256):
            hsv_colour_bar[y, x] = [x, 255, 255 - y]
    # convert the HSV colour bar to BGR
    hsv_colour_bar = cv2.cvtColor(hsv_colour_bar, cv2.COLOR_HSV2BGR)
    # resize the colour bar to fit the corners
    hsv_colour_bar = cv2.resize(hsv_colour_bar, (x2-x1, y2-y1), interpolation=cv2.INTER_AREA)
    # draw the colour bar on the image
    img[y1:y2, x1:x2] = hsv_colour_bar
    if annotate:
        scale = 0.5
        thickness = 1
        # put a '0' bottom-left of the bottom left corner of the colour bar
        cv2.putText(img, '0', (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, thickness, cv2.LINE_AA)
        # put a '255' bottom-right of the bottom right corner of the colour bar
        cv2.putText(img, '180', (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, thickness, cv2.LINE_AA)
        # put a '180' top-left of the top left corner of the colour bar
        cv2.putText(img, '1', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, text_colour, thickness, cv2.LINE_AA)
    return img

def listdir_complete(path, key=None):
    if key is None: return [os.path.join(path, f) for f in os.listdir(path)]
    return [os.path.join(path, f) for f in sorted(os.listdir(path), key=key)]

def add_upper_directory_to_path():
    # adds the directory above the current one to the system path
    # if sys and os have not already been imported, import them:
    if 'sys' not in globals(): import sys
    if 'os' not in globals(): import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# progress bar setup - redirects print to tqdm.write so as not to break the progress bar for the duration of the loop
def tqdmp(*args, **kwargs): # wrapper for tqdm that redirects print (the 'p' in 'tqdmp') to tqdm.write
    if 'contextlib' not in globals():   import contextlib
    if 'tqdm' not in globals():         from tqdm.auto import tqdm
    @contextlib.contextmanager # context manager to redirect print to tqdm.write for the duration of the test
    def redirect_to_tqdm():
        # Store builtin print
        old_print = print
        def new_print(*args, **kwargs):
            # If tqdm.tqdm.write raises error, use builtin print
            try:
                tqdm.write(*args, **kwargs)
            except:
                old_print(*args, ** kwargs)

        try:
            # Globally replace print with new_print
            inspect.builtins.print = new_print
            yield
        finally:
            inspect.builtins.print = old_print
    with redirect_to_tqdm():
        for x in tqdm(*args, **kwargs):
            yield x