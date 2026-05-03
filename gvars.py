# a script to store global variables shared between scripts
import numpy as np
import os

CAMERA_AZIMUTH = 180 # azimuth of the camera in degrees

# fields and dtypes to extract from image filenames
#               camera zenith   , azimuth,    zenith,     exposure,    ISO,       obstructed (0 or 1)
param_fields = ["cze"           , "az"      , "ze"      , "exp"     , "ISO"     , "obs"       ] # substring keys to extract from the filename  (minus im_num)
param_dtypes = [float           , float     , float     , np.float64, np.float64, int         ] # the dtypes of the parameters in the filename (minus np.int)
param_details = [(param_fields[i], param_dtypes[i]) for i in range(len(param_fields))] # list of tuples of the param fields and dtypes
CZE, AZ, ZE, EXP, ISO, OBS = 0, 1, 2, 3, 4, 5 # index constants for the above param variables, e.g. gvars.param_details[gvars.AZ] returns ('az', float)

# csv fields
linear_csv_fields = ["Azimuth_deg", "Zenith_deg", "Camera_Zenith_deg", "Phase_Angle_deg", "Plane_Angle_deg", 
                "I_Avg","Q_Avg","U_Avg","V_Avg",
                "DoP_Avg_magnitude", "AoP_Avg_deg", "DoP_SD", "AoP_SD",
                "Exposure_Time_us", "Gain_db", "Pixel_Oversaturation", "Pixel_Undersaturation", "Camera_Obstruction"]
circular_csv_fields = ["Azimuth_deg", "Zenith_deg", "Camera_Zenith_deg", "Phase_Angle_deg", "Plane_Angle_deg",
                       "V_Avg", "DoCP_Avg", "DoCP_SD",
                        "Exposure_Time_us", "Gain_db","Pixel_Oversaturation", "Pixel_Undersaturation", "Camera_Obstruction"]

# settings file
settings_file = 'settings_history' # file to store the settings history (cv_init will chuck a .csv on the end)
settings_fields = ['timestamp', 'camera_zenith', 'AoI_x1', 'AoI_x2', 'AoI_y1', 'AoI_y2']

# calibration file
## create csv in testdir with the calibration data (azimuth, zenith, camera_zenith, exposure, ISO, saturation, undetected)
calibration_fields = settings_fields + ['azimuth', 'zenith', 'exposure', 'ISO', 'saturation', 'undetected']

# area of interest proportional bounds
# AoI_Bounds: left, right, up, down proportion of image to crop (left < right, up < down)
#               x1,    x2, y1, y2
X1, X2, Y1, Y2 = 0,   1,     2,    3 # indices for AoI_Bounds
AoI_bounds =    [0.4, 0.525, 0.46, 0.535] # default AoI bounds for cam_zen 45 (but mostly it will be read from the settings file)
# lambda functions to get *current* AoI_bound values
x1, x2 = lambda: AoI_bounds[X1], lambda: AoI_bounds[X2]
y1, y2 = lambda: AoI_bounds[Y1], lambda: AoI_bounds[Y2]
# lambda function to get AoI_bounds in the form [x1, x2, y1, y2]
get_AoI_bounds = lambda: [x1(), x2(), y1(), y2()]

# camera max width / height
MAXWIDTH=2448
MAXHEIGHT=2048
img_width_default=MAXWIDTH
img_height_default=MAXHEIGHT

# angle ranges for camera obstruction
zen_block_range = 7 # degrees above and below the camera zenith
az_block_range  = 9  # degrees right (from the camera's perspective) of the camera azimuth (i.e. camera_azimuth - az_block_range)

# AoI_Bounds = [0.425, 0.55, 0.46, 0.535] # for cam_zen 45

'''
Old File Structure (v1):
	<test without pol state in directory name>
		raw [dir]
			linear [dir] -> <images> [.pngs]
			circular [dir] -> <images> [.pngs]
            <settings csv> [.csv] (possibly)
		hsv
			[incorrect (v1i)] linear [dir] -> <hsv images> [.pngs]
			[correct]   <linear hsv images> [.pngs]
		<summary plot file> [.png]
		<summary csv file> [.csv]

Current file structure (v2):
	<test with pol state in directory name>
		raw [dir]
			<settings csv> [.csv]
			linear/circular [dir]
				<images> [.pngs]
		hsv/cpi [dir]
			<images> [.pngs]
		<summary plot> [.png]
		<summary csv> [.csv]

TODO: turns out in v2 it's actually 'hsv/linear'...
TODO2: v3 will have raw_linear/raw_circular and hsv/cpi directories with no subdirectories

a pol state is one of '0_Polarised', '90_Polarised', 'Unpolarised', or 'Circ_Polarised'
'''
## file structure variables (relative to the test directory, e.g. os.path.join(test_dir, lin_dir))
lin_dir_str =   os.path.join(   'raw', 'linear'     )
hsv_dir_str =   os.path.join(   'hsv', 'linear'     )
circ_dir_str =  os.path.join(   'raw', 'circular'   )
cpi_dir_str =   os.path.join(   'cpi', 'circular'   )
test_summary_start_str = lambda test_dir: f'{test_dir}_POL_SUMMARY_' # returns the start of the test summary file name based on the test directory
# ^ is in the format 'testname_POL_SUMMARY_LINEAR.csv' or 'testname_POL_SUMMARY_CIRCULAR.csv', in full

## file structure functions
get_lin_path_strs = lambda test_dir: [os.path.join(test_dir, lin_dir_str), os.path.join(test_dir, hsv_dir_str)] # returns the theoretical path to the raw linear images and hsv images directories, given a test directory
get_circ_path_strs = lambda test_dir: [os.path.join(test_dir, circ_dir_str), os.path.join(test_dir, cpi_dir_str)] # returns the theoretical path to the raw circular images and cpi files directories, given a test directory
get_linear_summary_str = lambda test_dir: os.path.join(test_dir, f'{test_dir}_POL_SUMMARY_LINEAR') # returns the path to the linear summary csv
get_circular_summary_str = lambda test_dir: os.path.join(test_dir, f'{test_dir}_POL_SUMMARY_CIRCULAR') # returns the path to the circular summary csv
test_is_linear = lambda test_path: os.path.exists(get_lin_path_strs(test_path)[0]) # returns whether test directory has linear data based on the presence of a 'linear' folder in the 'raw' folder
test_is_circular = lambda test_path: os.path.exists(get_circ_path_strs(test_path)[0]) # returns if a test directory has circular data based on the presence of a 'circular' folder in the 'raw' folder

def get_test_type(test_path): 
    '''given the path to a test dir, returns the type of test in ['LINEAR', 'CIRCULAR', None] based on the presence of the 'linear' and 'circular' folders in the 'raw' folder'''
    # it's a valid test if it has either linear or circular data, but not both or neither
    if      test_is_linear(test_path) and test_is_circular(test_path):  return None
    elif    test_is_linear(test_path):                                  return 'LINEAR'
    elif    test_is_circular(test_path):                                return 'CIRCULAR'
    else:                                                               return None 
    
def get_test_summary_str(test_path):
    '''given the path to a test dir, returns the theoretical filename of the summary csv based on the test type'''
    test_type = get_test_type(test_path)
    if not test_type: return None # invalid test
    test_summary_str = f'{test_summary_start_str(test_path)}{test_type}.csv' # i.e. 'testname_POL_SUMMARY_LINEAR.csv' or 'testname_POL_SUMMARY_CIRCULAR.csv'
    return os.path.basename(test_summary_str) # only want to return the filename, not the full path

get_test_summary_csv = lambda test_path: get_test_summary_str(test_path) if os.path.exists(os.path.join(test_path, get_test_summary_str(test_path))) else None
'''given the path to a test dir, returns the filename of the summary csv based on the test type if it exists, else None'''

# TODO: private variables which should be accessed through getter and setter functions
exp = 40.0
def exposure(new_exposure=None):
    global exp
    if new_exposure is not None:
        if not isinstance(new_exposure, float):
            if isinstance(new_exposure, int): new_exposure = float(new_exposure)
            else: raise ValueError("exposure must be a float or int")
        exp = new_exposure
    return exp

# PLACES WHERE FILE I/O CURRENTLY HAPPENS WRT AOI SETTINGS
'''
- camera.calibrate will load settings_history if it exists, and after a completed calibration, will save the settings to settings_history
- 1_polarisation_raw_image_collect.polarisation_test will save the settings used in a linear/circular test 
    to an appropriately-named settings csv in the image parent folder (e.g. testname/raw/settings_LINEAR.csv)
NOT IMPLEMENTED: 
    2_pol_analysis.py will load the settings from the settings csv in the image parent folder
'''

# TODO THINGS WE WANT TO DO WITH THE AOI SETTINGS
'''
UNIVERSAL BEST AOI SETTINGS FOR EACH CAMERA ZENITH ANGLE:
- directory of the best settings for each camera zenith angle that can be, once populated, automatically retrieved and set as the AoI
'''