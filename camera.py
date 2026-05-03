# -----------------------------------------------------------------------------
# Copyright (c) 2022, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

# if not connecting, do:
# sudo ifconfig enp0s8 169.254.0.1 netmask 255.255.0.0
# (or set wired connection manual settinsg to that ip address and that netmask)

from arena_api.system import system
from arena_api.buffer import *
from arena_api.enums import PixelFormat
from arena_api.__future__.save import Writer

import ctypes
import numpy as np
import cv2
import time
import os
import atexit
import sys
# try: # running as script
# 	import helper_functions as hf
# 	import light
# 	import gvars
# except: # running as module
# 	import src.helper_functions as hf
# 	import src.light as light
# 	import src.gvars as gvars

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import helper_functions as hf
import light
import gvars

# import matplotlib.pyplot as plt

'''
Live Stream: Introduction
	This example introduces the basics of running a live stream 
	from a single device. This includes creating a device, selecting
	up stream dimensions, getting buffer cpointer data, creating an
	array of the data and reshaping it to fit image dimensions using
	NumPy and displaying using OpenCV-Python.

Save: Introduction
	This example introduces the basic save capabilities of the save library. It
	shows the construction of an image parameters object and an image writer, and
	saves a single image.
'''
## Settings
# MAXWIDTH=2448
# MAXHEIGHT=2048
triton = None
TAB1 = "  "
img_bounds_nodes = 		['Width'			   	, 'Height'				 , 'OffsetX'		, 'OffsetY'		  ]
exp_bounds_nodes = 		['AutoExposureAOIWidth'	, 'AutoExposureAOIHeight', 'AutoExposureAOIOffsetX', 'AutoExposureAOIOffsetY']
img_bounds_defaults = 	[gvars.img_width_default, gvars.img_height_default,		  		  0, 		 		 0] # should be good for both of the above
debug = False # True for slightly more verbose console output
if debug: changed_nodes = []
debug_node_strings = ['ExposureTime', 'GainAuto', 'PixelFormat', # handy node strings for testing
				'Width', 'Height', 'OffsetX', 'OffsetY', 
				'AutoExposureAOIWidth', 'AutoExposureAOIOffsetX', 
				'AutoExposureAOIHeight', 'AutoExposureAOIOffsetY']

'''
common enum entries list:
ExposureAuto: 'Off', 'Once', 'Continuous'
GainAuto: 'Off', 'Once', 'Continuous'
PixelFormat: 'Mono8', 'Mono10', 'Mono10p', 'Mono10Packed', 'Mono12', 'Mono12p', 'Mono12Packed', 'Mono16', 
	'PolarizeMono8', 'PolarizeMono12', 'PolarizeMono12p', 'PolarizeMono12Packed', 'PolarizeMono16', 
	'PolarizedAngles_0d_45d_90d_135d_Mono8', 'PolarizedStokes_S0_S1_S2_S3_Mono8', 'PolarizedDolpAolp_Mono8', 
	'PolarizedDolpAolp_Mono12p', 'PolarizedDolp_Mono8', 'PolarizedDolp_Mono12p', 'PolarizedAolp_Mono8', 
	'PolarizedAolp_Mono12p', 'PolarizedDolpAngle_Mono8', 'PolarizedDolpAngle_Mono12p', 'PolarizedDolpAngle_Mono16'
'''

exposure_max = 43000.0 #44833.288 # theroetically 47183.896 but whenever you restart stream above this value, it *is* this value
exposure_min = 33.48 # this may not be true, but officially this is the value

def main():
	triton = init(exposure=-1, gain=0, offx=0, offy=0)
	if img_calibration(cam_zenith=45):
		start_stream()
		try:
			while True: 
				# img, exp, code = get_unsaturated_img_from_stream()
				# if code == 5: 
				# 	print('exposure is caught in an infinite loop!')
				# 	break # caught in a loop
				# img, exp = get_img_from_stream('/media/apollo/E01A-DACE/test2')
				img = get_img_from_stream('/media/apollo/E01A-DACE/Tests/Painted_Empty_CircPol/raw/circular/1_cze:45.0_az:0.0_ze:0.0_exp:28003.7_ISO:0.0_obs:0_RAW')[0]
				cv2.imshow('preview', img)
				cv2.waitKey(1)
		except KeyboardInterrupt:
			cv2.destroyAllWindows()
			terminate(triton)
	else: # calibration aborted
		cv2.destroyAllWindows()
		terminate(triton)

## CAMERA FUNCTIONS
def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			hf.script_print(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				hf.script_print(f'{sec_count + 1 } seconds passed ' +
					'.' * sec_count, end='\r')
			tries += 1
		else:
			hf.script_print(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the script again.')

def init(gain=0.0, exposure=40000.0, width=gvars.img_width_default, height=gvars.img_height_default, offx=0, offy=0, pixel_format='Mono8'):
	global triton
	devices = create_devices_with_tries()
	triton = devices[0]
	reset_settings() # make sure each run is a blank slate
	setup(gain, exposure, width, height, pixel_format)

def setup(gain, exposure, width, height, pixel_format='Mono8'):#PixelFormat.Mono8):
	"""
	Setup stream dimensions and stream nodemap
		inputs:
			gain: ISO value in db (0-24 analog, 24-48 digital)
			exposure: exposure time in us (~0 to ~40000)
			width: image width in px (typically MAXWIDTH)
			height: image width in px (typically MAXHEIGHT)
			pixel_format: str valid pixel format for camera (e.g. RGB8, Mono8, PolarizedAngles_0d_45d_90d_135d_Mono8)
	"""
	##               TECHNICAL DOCUMENTATION: https://support.thinklucid.com/triton-tri050s/
	
	# width and height < 2448, 2048 is just cropped towards the top left
	# need nodes['OffsetX' and 'OffsetY'] to change the crop location, probably
	triton_exists()

	# dynamic exposure / gain if values are < 0
	exposure_auto = gain_auto = 'Off'
	if exposure < 0: exposure_auto = 'Continuous'
	if gain < 0: gain_auto = 'Continuous'
	# NB: if ExposureAuto or GainAuto are 'Continuous', they will make their corresponding params read_only (so they go first)
	nodestrs = ['Width', 'Height', 'PixelFormat', 'ExposureAuto', 'GainAuto'] + (['ExposureTime'] if exposure>=0 else []) + (['Gain'] if gain>=0 else [])
	nodevals = [ width ,  height ,  pixel_format,  exposure_auto,  gain_auto] + ([exposure] if exposure>=0 else []) + ([gain] if gain>=0 else [])

	## set camera settings
	change_settings(nodestrs, nodevals)
	tl_stream_nodemap = triton.tl_stream_nodemap
	tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

def change_settings(node_strings, node_vals):
	'''changes camera settings (i.e. node values) robustly given a list of nodes and values
		also handles the case where a stream is ongoing (detects this, stops the stream, changes settings, restarts stream)
		inputs: 
			node_strings: a list of nodes
			node_vals: the corresponding values to set each node in node_strings to (same length)
	'''
	triton_exists()
	if debug: [changed_nodes.append(node_str) for node_str in node_strings] # note nodes change from default if vals should be printed
	# make sure it's a list even if it's only one item (also don't want np arrays)
	if isinstance(node_strings, np.ndarray) or isinstance(node_vals, np.ndarray): Exception("Inputs must be lists or scalars, not ndarray")
	if not isinstance(node_strings, list): node_strings = [node_strings] 
	if not isinstance(node_vals, list): node_vals = [node_vals] # <^ you can input scalars, but they'll be converted to lists
	if len(node_strings) != len(set(node_strings)): raise Exception("Each string in node_strings must be unique")
	if len(node_strings) != len(node_vals): raise Exception(f"node strings and values length mismatch")
	
	# check that all node strings are valid
	nodemap = triton.nodemap
	node_list = nodemap._Nodemap__get_feature_names()
	for nstr, nval in zip(node_strings, node_vals):
		if nstr not in node_list: raise Exception(f"{nstr} is not a valid node name")

	# validation prior to node value assignment
	nodes = nodemap.get_node(node_strings)
 
	# handling offset and width/height nodes (changing width/height will change offset max/min, and vice versa)
	# if any two nodes contain 'Width' and 'OffsetX' or 'Height' and 'OffsetY' in their names, they must be changed in the correct order
	# first, get all the matched pairs
	x_node_substr_pairs = ['Width', 'OffsetX'] # both need to be present to consider this case
	y_node_substr_pairs = ['Height', 'OffsetY'] # both need to be present to consider this case
	for dim_node_substr_pairs in [x_node_substr_pairs, y_node_substr_pairs]:
		matched_pairs = set() # Use a set to store found pairs to avoid duplicates
		# get all the strings that contain the pair
		dim_node_strings = [node_str for node_str in node_strings if any(dim_str in node_str for dim_str in dim_node_substr_pairs)] 
		# if there are two strings that contain the pair, and they share the same prefix, add them to the found_pairs set
		matched_pairs.update((dim_node_string, offset_node_string) for dim_node_string in dim_node_strings for offset_node_string in dim_node_strings 
							if dim_node_string != offset_node_string and dim_node_string.split(dim_node_substr_pairs[0])[0] == offset_node_string.split(dim_node_substr_pairs[1])[0])
		# handle the matched pairs
		for dim_str, offset_str in matched_pairs:
			# pop the matched pairs out of node_strings and node_vals (to re-add later)
			dim_idx, offset_idx = node_strings.index(dim_str), node_strings.index(offset_str)
			dim_val, offset_val = hf.pop_multiple(node_vals, [dim_idx, offset_idx])
			hf.pop_multiple(node_strings, [dim_idx, offset_idx]) # remove the strings from the list (but we already have the values)

			## if it's a dim/offset node that requires a feature to be enabled, and that setting has been passed to this function: do it early
			for node_prefix_substr in ['AutoExposureAOI', 'AwbAOI', 'Chunk']:
				if any(node_prefix_substr in node for node in [dim_str, offset_str]):
					enable_node_str = f'{node_prefix_substr}Enable'
					if get_node(enable_node_str).value is True: pass # if it's already enabled, don't do anything
					elif enable_node_str in node_strings and node_vals[node_strings.index(enable_node_str)] is True:
						## AutoExposureAOIEnable itself requires ExposureAuto to be Continuous, to be able to be enabled
						if node_prefix_substr == 'AutoExposureAOI':
							if get_node('ExposureAuto').value == 'Continuous': pass # can set AutoExposureAOIEnable
							elif 'ExposureAuto' in node_strings and 'Continuous' in node_vals[node_strings.index('ExposureAuto')]:
								if node_exists_and_is_writable(nodes['ExposureAuto']):
									exposure_auto_str_idx = node_strings.index('ExposureAuto')
									node_strings.pop(exposure_auto_str_idx)
									node_vals.pop(exposure_auto_str_idx)
									nodes['ExposureAuto'].value = 'Continuous'
									hf.function_print(f"{'ExposureAuto'} set to {nodes['ExposureAuto'].value}{['', f' (input: {'Continuous'})'][debug]}") # TODO: debug print function in hf
							else:
								raise Exception('ExposureAuto must be Continuous to enable AutoExposureAOIEnable to let you access offset/dimension nodes')
						
						## TODO: there may be other similar conditions to AwbAOI and Chunk, but we'll have to figure that out as and when it comes up
						if node_exists_and_is_writable(nodes[enable_node_str]):
							enable_node_str_idx = node_strings.index(enable_node_str)
							node_strings.pop(enable_node_str_idx)
							node_vals.pop(enable_node_str_idx)
							nodes[enable_node_str].value = True
							hf.function_print(f"{enable_node_str} set to {nodes[enable_node_str].value}{['', f' (input: {'True'})'][debug]}") # TODO: debug print function in hf
					else:
						raise Exception(f'Need to enable {enable_node_str} before setting {offset_str} and {dim_str}')
			
			# if re-attaching the strings + vals: add them to the end 
			# (so, e.g., if AutoExposureAOIEnable = True has been passed to this function as well - that will happen first)
			if offset_val > nodes[offset_str].max: # new offset too big: set the dimension first
				node_strings = node_strings + [dim_str, offset_str]
				node_vals = node_vals + [dim_val, offset_val]
			elif dim_val > nodes[dim_str].max: # new dimension too big: set the offset first
				node_strings = node_strings + [offset_str, dim_str]
				node_vals = node_vals + [offset_val, dim_val]
			else: # neither of the above cases (order doesn't matter)
				node_strings = node_strings + [offset_str, dim_str]
				node_vals = node_vals + [offset_val, dim_val]

	# validate node values based on node properties, then set them 
	# need to val->set in a loop rather than val(all)->set(all) because different nodes affect each other
	# consequently, may need to undo a call to change_settings (by setting nodes to their initial values):
	init_node_strs = []
	init_node_vals = []
	# if triton.start_stream() has been called, need to stop it, *then* change node settings, then start it again
	ongoing_stream = triton.tl_stream_nodemap['StreamIsGrabbing'].value
	if ongoing_stream: triton.stop_stream()
	# triton.nodemap.get_node(['AcquisitionStop'])['AcquisitionStop'].execute() # required for OffsetX/Y(?)

	# for nidx, (nstr, nval) in enumerate(zip(node_strings, node_vals)):
	for nstr, nval in zip(node_strings, node_vals):
		init_node_strs.append(nstr)
		init_node_vals.append(nodes[nstr].value)
		
		try:
			nodetype = nodes[nstr].properties['NodeType']['value']

			# make sure nodes are the correct value
			if nodetype == 'Integer': nval = int(nval)#node_vals[nidx] = int(nval)
			elif nodetype == 'Float': nval = float(nval)#node_vals[nidx] = float(nval)
			elif nodetype == 'Enumeration' and not isinstance(nval, str): Exception(f'{nstr} expects a string')

			## width/height/offset node type values must be a multiple of four
			is_multiple_of_four_node = False
			if any([multiple_of_four_node in nstr for multiple_of_four_node in ['Width', 'Height', 'OffsetX', 'OffsetY']]):
				if nval % 4: # if that's any value but 0, i.e. not divisible by 4
					is_multiple_of_four_node = True
					newval = hf.round_up_to_num(nval, num=4)
					hf.function_print(f'{nstr} - rounded {nval} up to {newval} (must be divisible by 4)')
					nval = newval#node_vals[nidx] = newval

			# ## numerical nodes
			if hasattr(nodes[nstr], 'max') and nodes[nstr].max != "N/A" and nval > nodes[nstr].max:
				# node_vals[nidx] = nodes[nstr].max if not is_multiple_of_four_node else hf.round_down_to_num(nodes[nstr].max, num=4)
				nval = nodes[nstr].max if not is_multiple_of_four_node else hf.round_down_to_num(nodes[nstr].max, num=4)
				# hf.function_print(f'{nstr} - {nval} greater than maximum value {nodes[nstr].max}; changing to {node_vals[nidx]}.')
				hf.function_print(f'{nstr} - {nval} greater than maximum value {nodes[nstr].max}; changing to {nval}.')
			if hasattr(nodes[nstr], 'min') and nodes[nstr].min != "N/A" and nval < nodes[nstr].min: 
				# node_vals[nidx] = nodes[nstr].min if not is_multiple_of_four_node else hf.round_up_to_num(nodes[nstr].min, num=4)
				nval = nodes[nstr].min if not is_multiple_of_four_node else hf.round_up_to_num(nodes[nstr].min, num=4)
				# hf.function_print(f'{nstr} - {nval} less than minimum value {nodes[nstr].min}; changing to {node_vals[nidx]}.')
				hf.function_print(f'{nstr} - {nval} less than minimum value {nodes[nstr].min}; changing to {nval}.')

			# exposure dodgy maximum handling
			# FIXME: figure out why the maximum exposure time is 47183.896 but the camera stops at 44833.288 and handle it properly
			if nstr == 'ExposureTime':
				if nval > exposure_max:
					# node_vals[nidx] = float(exposure_max)
					nval = float(exposure_max)
					hf.function_print(f'{nval} greater than {nstr} node maximum {exposure_max}; changing to maximum.')
				if nval < exposure_min: 
					# node_vals[nidx] = float(exposure_min)
					nval = float(exposure_min)
					hf.function_print(f'{nval} greater than {nstr} node maximum {exposure_min}; changing to maximum.')
			
			## string / enum entries nodes
			if hasattr(nodes[nstr], 'enumentry_names') and nval not in nodes[nstr].enumentry_names: 
				raise Exception(f'invalid argument {nval} for {nstr} node')
			
			# there may be other cases than max/min/enumentry; tbd
		
			if node_exists_and_is_writable(nodes[nstr]): nodes[nstr].value = nval # node value assignment

		except: 
			hf.function_print(f'{sys.exc_info()[0]}: {sys.exc_info()[1].args[0]}') # print exception info
			hf.function_print(f'an exception occurred for {nstr}; resetting settings...')
			change_settings(init_node_strs.reverse(), init_node_vals.reverse()) # change them back in reverse order
		else: hf.function_print(f"{nstr} set to {nodes[nstr].value}{['', f' (input: {nval})'][debug]}") # TODO: debug print function in hf

	# triton.nodemap.get_node(['AcquisitionStart'])['AcquisitionStart'].execute()
	if ongoing_stream: triton.start_stream()

def node_exists_and_is_writable(node):
	'''checks if a node exists and is writable
		inputs:
			node_str: the node name to check
		outputs:
			True if the node exists and is writable, throw an Exception otherwise
	'''
	if node is None: raise Exception(f"{node.name} node not found")
	if node.is_writable is False: raise Exception(f"{node.name} node not writeable")
	return True

def img_calibration(cam_zenith):
	'''
	This function should take a picture from the camera and display it with the current area of interest from `analysis.py` overlaid as a rectangle
	If 'settings_history.csv' exists, load the last AoI values from that file
	Prompt the user to input the camera zenith angle
	Print to the console how the user can interact with the preview
	The user should then be able to keyboard input the following (and see the changes in real time):
		wasd: 		move the top left corner of the rectangle 
		arrow keys: move the bottom right corner of the rectangle
						(for both of the above, the new AoI values for each edge should be displayed next to that edge)
		# p: 			take a new picture		# trying just making it take a new picture when a setting is changed
		esc: 		quit the test
		enter: 		set `analysis.AoI_Bounds' to the new values, save the new AoI values and camera zenith 'settings_history.csv' along with a timestamp',
					and finish executing the function as per normal
	'''
	triton_exists()
	# TODO: on-the-fly camera settings adjustment

	# if the settings file exists, load the most recent AoI_bounds
	hf.function_print(['No settings file found, using default values', 
					   'Loading most recent AoI bounds from settings file']
					   [hf.load_AoI_settings(gvars.settings_file)])
	# change_settings(img_bounds_nodes, img_bounds_defaults) # reset the image dimensions to the full image
	change_settings(img_bounds_nodes 	+ ['AutoExposureAOIEnable'], 
				 	img_bounds_defaults + [False]) # reset the image dimensions to the full image

	cv2.imshow('Preview', np.zeros((2,2))) # dummy image so imshow doesn't show a blank window initially
	cv2.waitKey(1)

	# move current_auto_states through 1-2, looping back; get init states
	exp_auto_state = 	[0] if get_node('ExposureAuto') == 'Off' else [1]# index of auto_states list of strings
	gain_auto_state = 	[0] if get_node('GainAuto') 	== 'Off' else [1]
	# one element lists to approximate passing by reference (i.e. modify the input without needing to assign it at the function call)
	def get_next_auto_state(state_var): 
		# auto_states = ['Off', 'Once', 'Continuous']
		auto_states = ['Off', 'Continuous'] # do not want 'Once'
		state_var[0] = state_var[0] + 1 if state_var[0] < len(auto_states)-1 else 0 
		return auto_states[state_var[0]]
	
	with triton.start_stream():
		print("Stream started")

		img, exp, gain = get_img_from_stream()
		img2 = img.copy()
		AoI_move = 0.005 # proportion of the image to move the AoI by
		light_move = 0.05
		exp_move = 1000
		gain_move = 2

		## KEYBINDINGS FOR SETTINGS TELEOP
		tl_up_k, tl_left_k, tl_down_k, tl_right_k = 'w', 'a', 's', 'd' 	# top-left AoI rectangle
		br_up_k, br_left_k, br_down_k, br_right_k = 'i', 'j', 'k', 'l' 	# bottom-right AoI rectangle
		li_up_k, li_down_k = 						'+', '-' 			# light up/down
		e_up_k, e_down_k = 82, 84 				  # up/down arrow keys	# exposure up/down
		g_up_k, g_down_k = 83, 81 			   # right/left arrow keys	# gain up/down (right/left arrow keys)
		ea_toggle_k, ga_toggle_k = 					'v', 'b' 			# exposure, gain auto toggle
		pic_k, auto_exp_k = 						'p', 'n'					# take new picture
		reset_k, reset_most_k =						'r', 't'			# reset to initial settings / to initial settings but for AoI
		quit_k, cont_k = 27, 13 					# esc, enter		# quit or continue the test

		print("\n\nIMAGE CALIBRATION CONTROLS")
		print("===========================")
		print("Adjust the Area of Interest (AoI) to the desired bounds")
		print("Adjust exposure/light/gain (in that order) until (preferably) no under- or over-saturation")
		print(f"{tl_up_k}/{tl_left_k}/{tl_down_k}/{tl_right_k}		: move the top left corner of the rectangle")
		print(f"{br_up_k}/{br_left_k}/{br_down_k}/{br_right_k}		: move the bottom right corner of the rectangle")
		print(f"{li_up_k}/{li_down_k}		: increase or decrease the light level [0, 1]")
		print(f"up/down 	: increase or decrease the exposure time (us) (auto exposure must be 'Off')")
		print(f"right/left 	: increase or decrease the ISO (db) (auto gain must be 'Off')")
		print(f"{pic_k}/{auto_exp_k}		: take a new picture / take unsaturated picture")
		''' camera-level autoexposure disabled
		print(f"{ea_toggle_k}/{ga_toggle_k}		: toggle auto exposure ('v') and auto gain ('b') states")
		'''
		print(f"{reset_k}/{reset_most_k}		: reset all / all except AoI to initial settings")
		print(f"esc		: quit the test")
		print(f"enter		: save the settings and continue the test")
		print(f"===========================\n")

		nodestr = ['ExposureAuto', 'GainAuto', 'ExposureTime', 'Gain']
		nodemap = triton.nodemap
		nodes = nodemap.get_node(nodestr)
		exposure_bounds = [exposure_min, exposure_max] #[nodes['ExposureTime'].min, nodes['ExposureTime'].max]
		gain_bounds = [nodes['Gain'].min, nodes['Gain'].max]
		auto_exposure_active = lambda: True if nodes['ExposureAuto'].value != 'Off' else False
		auto_gain_active = lambda: True if nodes['GainAuto'].value != 'Off' else False

		# initial setting values
		new_AoI_Bounds = gvars.get_AoI_bounds()
		new_brightness = np.round(light.get_pwm(), 2)
		new_exposure = nodes['ExposureTime'].value
		new_exp_auto = nodes['ExposureAuto'].value
		new_gain = nodes['Gain'].value
		new_gain_auto = nodes['GainAuto'].value
		
		# initial settings to return to
		init_AoI = new_AoI_Bounds.copy() # is a list, so need to copy (don't want pass by reference)
		init_brightness = new_brightness
		init_exp_auto, init_exposure = new_exp_auto, new_exposure
		init_gain_auto, init_gain = new_gain_auto, new_gain

		display_size_pc = 30 # % percent image size
		# px_avg_maxes = [0, 0, 0, 0] # for linear polarisation calibration - moved
		while True:
			key = None
			## CAMERA FEED + PARAM CHANGE
			reset = False
			img_preview = hf.gs3c(img.copy()) # 3-channel, 30% scale
			# get AoI bounds (e.g. vars.x1()) in px (e.g. x1)
			x1, x2 = [int(gvars.x1()*img_preview.shape[1]), int(gvars.x2()*img_preview.shape[1])]
			y1, y2 = [int(gvars.y1()*img_preview.shape[0]), int(gvars.y2()*img_preview.shape[0])]

			# draw the AoI rectangle
			cv2.rectangle(img_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

			## show AoI values next to each edge of the rectangle: 
			horz_offset, vert_offset = 70, 30# offset values in px to move text outside the rectangle
			X1pos = (x1-horz_offset , 	y1 + (y2-y1)//2)	# vars.X1 between [x1,y1] and [x2, y1] left of the rectangle edge,
			X2pos = (x2				, 	y1 + (y2-y1)//2) 	# vars.X2 between [x1,y2] and [x2, y2] right of the rectangle edge,
			Y1pos = (x1 + (x2-x1)//2, 	y1-10) 				# vars.Y1 between [x1,y1] and [x1, y2] above the rectangle edge,
			Y2pos = (x1 + (x2-x1)//2, 	y2+vert_offset) 	# vars.Y2 between [x2,y1] and [x2, y2] below the rectangle edge
			
			cv2.putText(img_preview, f"{gvars.x1():.2f}", X1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img_preview, f"{gvars.x2():.2f}", X2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img_preview, f"{gvars.y1():.2f}", Y1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(img_preview, f"{gvars.y2():.2f}", Y2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

			# for linear polarisation calibration - moved
			# sat_img, px_avgs = make_sat_img(hf.crop_to_proportions(img.copy(), vars.AoI_Bounds), img_preview, px_avg_maxes)
			# for i in range(len(px_avg_maxes)): 
			# 	if px_avgs[i] > px_avg_maxes[i]: px_avg_maxes[i] = px_avgs[i]
			sat_img = make_sat_img(hf.crop_to_proportions(img.copy(), gvars.get_AoI_bounds()), img_preview)
			
			combined_img = hf.img_res(np.hstack((img_preview, sat_img)), display_size_pc)
			# show the image, draw the AoI, and get user input
			cv2.imshow('Preview', combined_img)

			# r: reset to initial config
			def calibration_reset(AoI=False):
				print('Resetting to initial values...')
				nonlocal new_AoI_Bounds, new_brightness, new_exp_auto, new_gain_auto, new_exposure, new_gain, reset
				if AoI: new_AoI_Bounds = init_AoI
				new_brightness = init_brightness
				new_exp_auto, new_exposure = init_exp_auto, init_exposure
				new_gain_auto, new_gain = init_gain_auto, init_gain
				reset = True

			# get user input
			key = cv2.waitKey(1)
			
			# if the 'p' key is pressed, update the base image
			if key == ord(pic_k):
				print("Taking new image...")
				img, exp, gain = get_img_from_stream()
				print("New image taken")
			elif key == ord(auto_exp_k): 
				img, exit_code, exp, gain = get_unsaturated_img_from_stream()
				new_exposure = exp

			# wasd: move the top left corner of the rectangle (whether upper or lower case)
			if   key == ord(tl_up_k): new_AoI_Bounds[gvars.Y1] -= AoI_move # w
			elif key == ord(tl_left_k): new_AoI_Bounds[gvars.X1] -= AoI_move # a
			elif key == ord(tl_down_k): new_AoI_Bounds[gvars.Y1] += AoI_move # s
			elif key == ord(tl_right_k): new_AoI_Bounds[gvars.X1] += AoI_move # d

			# ijkl: move the bottom right corner of the rectangle (whether upper or lower case)
			elif key == ord(br_up_k): new_AoI_Bounds[gvars.Y2] -= AoI_move # i
			elif key == ord(br_left_k): new_AoI_Bounds[gvars.X2] -= AoI_move # j
			elif key == ord(br_down_k): new_AoI_Bounds[gvars.Y2] += AoI_move # k
			elif key == ord(br_right_k): new_AoI_Bounds[gvars.X2] += AoI_move # l

			# +/-: increase/decrease the light level
			elif key == ord(li_up_k): new_brightness += light_move
			elif key == ord(li_down_k): new_brightness -= light_move

			# up/down: increase/decrease the exposure
			elif key == e_up_k: new_exposure += exp_move # up
			elif key == e_down_k: new_exposure -= exp_move # down

			# right/left: increase/decrease the gain
			elif key == g_up_k: new_gain += gain_move # right
			elif key == g_down_k: new_gain -= gain_move # left

			# # v/b: toggle auto exposure/gain state
			# elif key == ord(ea_toggle_k): new_exp_auto = get_next_auto_state(exp_auto_state)
			# elif key == ord(ga_toggle_k): new_gain_auto = get_next_auto_state(gain_auto_state)
			
			elif key == ord(reset_k): calibration_reset(AoI=True)
			elif key == ord(reset_most_k): calibration_reset(AoI=False)

			# if the 'esc' key is pressed
			elif key == quit_k: # ord('')
				proceed = False
				print("Image calibration aborted, exiting...")
				break

			# if the 'enter' key is pressed
			elif key == cont_k:
				proceed = True
				print("Image calibration complete, proceeding...")
				break

			# arrow keys - left: 81 | up: 82 | right: 83 | down: 84

			else: 
				continue # do nothing, wait for the next key press

			'''
			key debugging code (i.e. what is each key to waitKey):
			cv2.imshow('test', np.zeros((100,100)))
			o = 'a'
			while True:
				k = cv2.waitKey(1)
				if k != o: print(k)
				o = k
				# press keys in the imshow window to print their codes to console, ctrl-c to exit
			'''

			## INPUT VALIDATION:
			img_bound_warnings = ["Warning: X1 is out of bounds", "Warning: X2 is out of bounds", "Warning: Y1 is out of bounds", "Warning: Y2 is out of bounds"]
			small_AoI_warning = "Warning: AoI is too small"
			light_bound_warnings = ["Warning: light value cannot go below 0 (0.15 recommended)", "Warning: light value cannot go above 1 (0.15 recommended)"]

			exposure_bound_warnings = [f"Warning: exposure time cannot go below {exposure_bounds[0]}", f"Warning: exposure time cannot go above {exposure_bounds[1]}"]
			gain_bound_warnings = [f"Warning: ISO cannot go below {gain_bounds[0]}", f"Warning: ISO cannot go above {gain_bounds[1]}"]

			bounds_valid = True
			# not checking for inversion because we're already checking that the difference between the two values is not too small
			# if the new AoI is too small, print a warning
			if new_AoI_Bounds[gvars.X2] - new_AoI_Bounds[gvars.X1] < 0.01 or new_AoI_Bounds[gvars.Y2] - new_AoI_Bounds[gvars.Y1] < 0.01:
				print(small_AoI_warning)
				bounds_valid = False
			# if all checks pass, update the AoI values
			# if new_AoI_bounds is not within the bounds of preview_img, print a warning
			for i in range(len(new_AoI_Bounds)):
				if new_AoI_Bounds[i] < 0 or new_AoI_Bounds[i] > 1:
					print(img_bound_warnings[i])
					bounds_valid = False
			if bounds_valid: gvars.AoI_bounds = new_AoI_Bounds # update AoI_bounds
			else: new_AoI_Bounds = gvars.get_AoI_bounds()	  # reset new_AoI_bounds
				# img, exp = get_img_from_ongoing_stream() # update the image

			# Light level validation
			light_bound_warnings = ["Warning: light value cannot go below 0 (0.15 recommended)", "Warning: light value cannot go above 1 (0.15 recommended)"]
			if new_brightness < 0: 
				print(light_bound_warnings[0])
				new_brightness = 0
			elif new_brightness > 1: 
				print(light_bound_warnings[1])
				new_brightness = 1
			elif new_brightness != light.get_pwm(): # only change anything or print feedback if there is a change
				new_brightness = np.round(new_brightness, 2) # cut off distant floating point value
				light.set_pwm(new_brightness)
				img, exp, gain = get_img_from_stream() # update the image
				# print(f'LED brightness ([0, 1]) set to {new_brightness}') # light.set_pwm already prints something like this

			# Exposure validation
			if new_exp_auto != nodes['ExposureAuto'].value: 
				change_settings('ExposureAuto', new_exp_auto)
				new_exposure = nodes['ExposureTime'].value # reset it
			if new_exposure != nodes['ExposureTime'].value: # only change anything or print feedback if there is a change
				if auto_exposure_active(): 
					print('Auto Exposure is not "Off", cannot manually change exposure')
					new_exposure = nodes['ExposureTime'].value # reset it
				elif int(new_exposure) < int(exposure_bounds[0]) : print(exposure_bound_warnings[0])
				elif int(new_exposure) > int(exposure_bounds[1]) : print(exposure_bound_warnings[1])
				else: 
					change_settings('ExposureTime', new_exposure)
					new_exposure = nodes['ExposureTime'].value # make sure they're *exactly* the same
					img, exp, gain = get_img_from_stream() # update the image

			# Gain validation
			if new_gain_auto != nodes['GainAuto'].value: 
				change_settings('GainAuto', new_gain_auto)
				new_gain = nodes['Gain'].value # reset it
			if new_gain != nodes['Gain'].value: # only change anything or print feedback if there is a change
				if auto_gain_active(): 
					print('Auto Gain is not "Off", cannot manually change gain')
					new_gain = nodes['Gain'].value # reset it
				elif int(new_gain) < int(gain_bounds[0]) : print(gain_bound_warnings[0])
				elif int(new_gain) > int(gain_bounds[1]) : print(gain_bound_warnings[1])
				else:
					change_settings('Gain', new_gain)
					new_gain = nodes['Gain'].value # make sure they're *exactly* the same
					img, exp, gain = get_img_from_stream() # update the image

			if reset:
				img, exp, gain = get_img_from_stream() # update the image
				print('Done Resetting.')

		gvars.AoI_bounds = [round(i, 2) for i in gvars.get_AoI_bounds()] # round to 2 decimal places before saving
		
		# change_settings(img_bounds_nodes, hf.AoI2OffsetPx()) # reset the camera offset to 0, 0

		# AutoExposureAOIEnable can only be set to true if ExposureAuto is Continuous
		# AutoExposureAOI Width/Height/OffsetX/OffsetY can only be set if AutoExposureAOIEnable is True
		if nodes['ExposureAuto'].value == 'Continuous':
			[w, h, ox, oy] = hf.AoI2OffsetPx() # get pixel values for width, height, offsetx, offsety from proportionate AoI
			# TODO: preprocess w, h, ox, oy to be a multiple of 4 (2?)
			# change_settings(img_bounds_nodes  + ['AutoExposureAOIEnable'] + exp_bounds_nodes, 
			# 				  [w, h, ox, oy] 	+ [True] 					+ [w, h, 0, 0]
			
			change_settings(['AutoExposureAOIEnable'] + exp_bounds_nodes,  # image not cropped, autoexposure decided by AOI
							[True] 					  + [w, h, ox, oy]) 
			# change_settings(['AutoExposureAOIEnable'] + img_bounds_nodes,  # image cropped to AOI, autoexposure decided by AOI
			# 				[True] 			  		  + [w, h, ox, oy]) 
			# change_settings(img_bounds_nodes,								# image cropped to AOI, autoexposure decided by ???
			# 						[w, h, ox, oy]) 
		# else: change_settings(img_bounds_nodes, [w, h, ox, oy]) 

	triton.stop_stream()
	cv2.destroyAllWindows()
	print("Stream stopped")
	
	# if the settings are good: save the settings [datetimestamp, camera_zenith, AoI_x1, AoI_x2, AoI_y1, AoI_y2] to 'settings_history.csv'
	if proceed: 
		date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
		settings_data = [date_time, cam_zenith, gvars.x1(), gvars.x2(), gvars.y1(), gvars.y2()]
		# create the settings_history.csv file if it doesn't exist, then append the settings_data to it
		if not os.path.exists(f'{gvars.settings_file}.csv'): hf.csv_init(gvars.settings_file, gvars.settings_fields)
		hf.csv_append(gvars.settings_file, settings_data)
			
	# if the user wants to quit, exit the entire program
	return proceed

def get_img_from_stream(img_name='', img_format='png', pixel_format='Mono8'): #PixelFormat.Mono8): # preferred method of getting images
	'''
	Grab a single image an ongoing camera stream
		inputs: 
			a system.create_device object
			img_name: optionally a file path without the extension if the image is to be saved
			img_format: file extension for the image to be saved
			pixel_format: optionally a pixel format other than PixelFormat.Mono8 if trying something exotic
		outputs: 
			a single image from the camera stream (currently only Mono8 is supported, so only the first channel is returned)
			optionally saves the image
	
	Usage in other scripts/functions:
		def other_fun():
			with device.start_stream():
				while True:
					img, exp = get_img_from_ongoing_stream(device)
			device.stop_stream()
	'''
	triton_exists()
	if not triton.tl_stream_nodemap['StreamIsGrabbing'].value: 
		print("Stream not running; starting stream...")
		triton.start_stream() # start the stream if it's not already running

	# TODO: if img_format has a '.' in it, REMOVE IT

	# get camera settings
	nodemap = triton.nodemap
	nodes = nodemap.get_node(['PixelFormat', 'ExposureTime', 'Gain'])
	num_channels = 1 if nodes['PixelFormat'].value == 'Mono8' else Exception("Pixel format not supported (only Mono8 supported)")

	# get the image buffer, optionally save it, requeue the buffer aftewards
	buffer = triton.get_buffer() 

	exposure = nodes['ExposureTime'].value
	gain = nodes['Gain'].value
	# modify img_name with the *actual* exposure time
	if img_name != '': 
		img_name = hf.param_change(img_name, 'exp', exposure) + '_RAW' # param_change strips the 'RAW'/'HSV' out
		# find and delete any duplicates (based on im_num)
		duplicates = hf.find_duplicate_im_num_imgs(img_name)
		if duplicates: [os.remove(f) for f in duplicates] # true if duplicates is not an empty list (duplicate file names are full paths)
		# save image if given image path; must save before buffer requeued (handle saving to external drives differently; no ':' allowed)
		img_path = f'{img_name.replace(':','_')}.{img_format}' if img_name.startswith('/media/') else f'{img_name}.{img_format}'
		save(buffer, img_path, pixel_format) # save image if given image path; must save before buffer requeued
	
	item = BufferFactory.copy(buffer)
	triton.requeue_buffer(buffer) # requeue buffer to avoid running out of buffers
	img = triton_buffer_item_to_cv2_img(item)
	# cleanup
	BufferFactory.destroy(item) # destroy the copied item to prevent memory leaks

	# return the image array and the actual exposure value
	if debug: hf.function_print(''.join([f'{nstr}: {get_node(nstr).value}, ' for nstr in set(changed_nodes)]))
	return img[:,:,0], exposure, gain # only want first two dimensions of image, despite the above

def get_unsaturated_img_from_stream(img_name='', saturation_threshold=1.0, delta_saturation_threshold=2.0):
	'''
	Iteratively adjust camera settings to get an image that is neither oversaturated nor undersaturated with a bracketing approach
	if it ends up being both: switch to gradient descent to find the optimal exposure time to minimise both
	inputs: 
		saturation_threshold: the percentage of saturation at which to consider the image oversaturated or undersaturated
		delta_saturation_threshold: the percentage difference between over- and under-saturation (when both are present) 
			at which to consider the image saturation globally minimised
	outputs:
		img: an image that is neither oversaturated nor undersaturated
		current_exposure: the exposure time at which the image was taken
		exit_code: an integer which indicates a particular problem has arisen (if it isn't 0); see `print_unsat_image_code`, below
	'''

	# get exposure, gain, autoexposure, and autogain settings
	triton_exists()
	nodes = triton.nodemap.get_node(['ExposureTime', 'Gain', 'ExposureAuto', 'GainAuto'])
	if nodes['ExposureAuto'].value != 'Off': Exception("ExposureAuto must be Off to run this function")
	if nodes['GainAuto'].value != 'Off': Exception("GainAuto must be Off to run this function")

	# saturation_threshold = np.float64(saturation_threshold) # make it the same type as what it's being compared to
	current_exposure = 0.0
	prev_exposure = 1.0
	prev_delta_sat = 100.0 # must be lower than this the first time, necessarily
	exit_code = 0
	unsaturated = stuck = gradient_descent = False
	lower_bound_exp, upper_bound_exp = exposure_min, exposure_max # initial bounds for exposure time
	# TODO: gradient descent currently gets global minimum of over- and under-saturation functions; make it get the global minimum of the sum of the two
	while not (unsaturated or stuck): # always run at least once
		## get the next image from the ongoing stream and crop it to the AoI
		if gradient_descent: prev_delta_sat = delta_sat
		prev_exposure = current_exposure
		img, current_exposure, gain = get_img_from_stream(img_name)
		cropped = hf.crop_to_proportions(img.copy(), gvars.get_AoI_bounds())

		## case 1: repeated exposure value (stuck on softmax/softmin - thanks Lucid)
		# if both saturation present, don't do this (you'll figure it out)
		if (int(prev_exposure) == int(current_exposure)) and not unsaturated: #and not gradient_descent: 
			exit_code = 4
			stuck = True # will exit the loop
		
		oversat, undersat = hf.sat_pct(cropped) # get the saturation percentages of the current image
		delta_sat = undersat - oversat # difference between the two saturation percentages
		total_sat = undersat + oversat # total saturation percentage
		# info_str = f'{undersat:.2f}%/{oversat:.2f}% Undersat/Oversat (Δ {delta_sat:.2f}) for {current_exposure:.2f} us Exposure'
		info_str = f'{undersat:.2f}%/{oversat:.2f}% (Δ {delta_sat:.2f}) {current_exposure:.2f} us | (Undersat/Oversat) Exposure'

		## gradient descent logic variables
		# gradient_descent_gain   = 1000
		gradient_descent_gain   = 0.2 * (upper_bound_exp - lower_bound_exp)
		# gradient_descent_gain   = exposure_max - exposure_min # around 40000
		gradient_descent_delta  = gradient_descent_gain * (delta_sat)/100 # % difference of delta_sat becomes proportion of gain

		## case 2: over- and under-saturated, need to do gradient descent to find the global minimum
		if gradient_descent: # check whether the difference between over- and under-saturation has been sufficiently minimised
			if abs(delta_sat) < delta_saturation_threshold: unsaturated = True
		# check whether gradient descent is necessary, which it is if the over- and under-saturation functions ever intercept
		if undersat > saturation_threshold and oversat > saturation_threshold:
			gradient_descent = True; exit_code = 1
			if abs(delta_sat) > abs(prev_delta_sat): # if delta_sat has increased: jumped too far last step, set new upper/lower bound
				# condition: -ve if oversat > undersat, +ve if undersat > oversat
				if delta_sat < 0: 	upper_bound_exp = current_exposure 
				else: 				lower_bound_exp = current_exposure
			delta_exposure = gradient_descent_delta
			print(f"{info_str} - both; {'increasing' if delta_sat > 0 else 'reducing'} exposure...")
		
		# if it's saturated, use the 'bracketing' method to choose a new exposure value
		elif undersat > saturation_threshold:
			# if undersaturated but exposure time is already at minimum, print a message and break
			if int(current_exposure) >= int(exposure_max): exit_code = 2; break
			# move halfway between the current exposure and the upper bound
			if gradient_descent: delta_exposure = gradient_descent_delta
			else: 				 delta_exposure = (upper_bound_exp - current_exposure) / 2 # bracketing logic
			# update the lower bound for the next iteration
			lower_bound_exp = current_exposure # update the lower bound for the next iteration
			print(f"{info_str} - increasing exposure...")
		
		elif oversat > saturation_threshold:
			if int(current_exposure) <= int(exposure_min): exit_code = 3; break
			# move halfway between the current exposure and the lower bound
			if gradient_descent: delta_exposure = gradient_descent_delta
			else: 				 delta_exposure = -(current_exposure - lower_bound_exp) / 2 # bracketing logic
			# update the list of oversaturation values, and the upper bound for the next iteration
			upper_bound_exp = current_exposure # update the upper bound for the next iteration
			print(f"{info_str} - reducing exposure...")
		
		else: unsaturated = True # good to go!
		
		if not unsaturated: 
			new_exposure = current_exposure + delta_exposure
			if new_exposure > exposure_max or new_exposure < exposure_min: new_exposure = [exposure_min, exposure_max][new_exposure > exposure_max]
			change_settings('ExposureTime', new_exposure)
		
	return img, exit_code, current_exposure, gain

def print_unsat_img_exit_code(exit_code):
	exit_code_msgs = ["Exit Code 0: Image is neither over- nor under-saturated",
					  "Exit Code 1: Image is over- and under-saturated near the intersection of both",
					  "Exit Code 2: Image is under-saturated and exposure time is already at maximum",
					  "Exit Code 3: Image is over-saturated and exposure time is already at minimum",
					  "Exit Code 4: Tried to set new exposure value but caught on softmax/min"]
	if isinstance(exit_code, int) and exit_code in range(len(exit_code_msgs)): print(exit_code_msgs[exit_code])
	else: 	   print(f"Exit Code {exit_code}: Unknown")
	# TODO: move this functionality into the base function?

def make_sat_img(cropped, img_to_match, historical_max=None):
	'''
	Make a saturation image from a cropped image
	inputs:
		cropped: an image cropped to the AoI
		img_to_match: the image to match the size of the saturation image to
		historical_max: a maximum recorded pixel value in the form (0deg_max, 45deg_max, 90deg_max, 135deg_max), or None
	outputs:
		sat_img: an image with the same dimensions as img_to_match, with saturated pixels in red and undersaturated pixels in blue'''
	if historical_max is not None and not (isinstance(historical_max, list) and len(historical_max) == 4): raise Exception('historical_max must be a list of four elements, or None')
	oversat, undersat = hf.sat_pct(cropped) # get the % of pixels that are saturated
	sat_img = hf.gs3c(cropped.copy()) # get rid of third dimension of npndarray for sat_img
	sat_img[cropped == 255] = (0,0,255) # saturated img pixels: corresponding sat_img pixels to red
	sat_img[cropped == 0] = (255, 0, 0) # undetected img pixels: corresponding sat_img pixels to blue
    
    # enlarge the saturation image to match half the display image (i.e. pol split), or just half the max size if the image is camera-cropped, and put some stats on it
	sat_img = hf.match_img_size(sat_img, [ np.zeros((gvars.MAXHEIGHT, gvars.MAXWIDTH))  ,  img_to_match ] [ hf.is_full_img_size(img_to_match) ] )
	init_y, dist_y = 80, 80 # initial y position of text + how far to move it down each time
	sat_info = [
		f'Min/Max Px Value: {np.min(cropped)}/{np.max(cropped)}',
		f'% Oversaturated (Red): {oversat:.2f}',
		f'% Undersaturated (Blue): {undersat:.2f}',
	]
	if historical_max is not None: 
		pol_img = hf.pol_split(img_to_match.copy())
		pol_avg = [np.mean(img) for img in pol_img]
		sat_info = sat_info + [
			f'0/45/90/135 u     : {pol_avg[0]:.2f}/{pol_avg[1]:.2f}/{pol_avg[2]:.2f}/{pol_avg[3]:.2f}',
			f'0/45/90/135 u_max: {historical_max[0]:.2f}/{historical_max[1]:.2f}/{historical_max[2]:.2f}/{historical_max[3]:.2f}'
		]
	for i, msg in enumerate(sat_info):
		cv2.putText(sat_img, msg, (7, init_y+(dist_y*i)), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
	# cv2.putText(sat_img, f'Min/Max Px Value: {np.min(cropped)}/{np.max(cropped)}', (7, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # disp max pixel value
	# cv2.putText(sat_img, f'% Oversaturated (Red): {oversat:.2f}', (7, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # disp % of saturated pixels
	# cv2.putText(sat_img, f'% Undersaturated (Blue): {undersat:.2f}', (7, 280), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # disp % of unsaturated pixels)
	if historical_max is not None: return sat_img, pol_avg
	return sat_img
    
def triton_buffer_item_to_cv2_img(item):
	'''
	Converts a buffer to a cv2 image
	'''
	# get camera settings
	nodemap = triton.nodemap
	nodes = nodemap.get_node(['PixelFormat', 'ExposureTime'])
	num_channels = 1 if nodes['PixelFormat'].value == 'Mono8' else 3
	
	## prepare local copy of img to return
	buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
	# Buffer data as cpointers can be accessed using buffer.pbytes
	array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
	# create a reshaped NumPy array to display using OpenCV
	npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel)) # probably 1 channel
	img = npndarray.copy() # so the bufferfactory destroy doesn't remove it from memory
	# cleanup
	
	return img

def save(buffer, impath, pixel_format):
	'''
	demonstrates saving an image
	(1) converts image to a displayable pixel format
	(2) prepares image parameters
	(3) prepares image writer
	(4) saves image
	(5) destroys converted image
	'''
	'''
	Convert image
		Convert the image to a displayable pixel format. It is worth keeping in
		mind the best pixel and file formats for your application. This example
		converts the image so that it is displayable by the operating system.
	'''
	save_debug = False

	converted = BufferFactory.convert(buffer, pixel_format)
	if save_debug: print(f"{TAB1}Converted image to {pixel_format.name}")

	'''
	Prepare image writer
		The image writer requires 2 parameters to save an image: the buffer and
		specified file name or pattern. Default name for the image is
		'image_<count>.jpg' where count is a pre-defined tag that gets updated
		every time a buffer image.
	'''
	if save_debug: print(f'{TAB1}Prepare Image Writer')
	writer = Writer()
	writer.pattern = impath
	# writer.pattern = 'images/image_<count>.jpg'

	# Save converted buffer
	writer.save(converted)
	if save_debug: print(f'{TAB1}Image saved')

	# Destroy converted buffer to avoid memory leaks
	BufferFactory.destroy(converted)

def terminate(device=None): 
	system.destroy_device(device) # if None, destroys all devices
	cv2.destroyAllWindows()
	hf.script_print('Exiting... Camera object terminated')
atexit.register(terminate)

def reset_settings():
	triton_exists()
	# Reset to default user set
	triton.nodemap['UserSetSelector'].value = 'Default'
	triton.nodemap['UserSetLoad'].execute()
	hf.function_print('Device settings have been reset to \'Default\' user set')

## CAMERA FUNCTIONS (for use externally)

start_stream = lambda : triton.start_stream() if triton_exists() else None

stop_stream = lambda: triton.stop_stream() if triton_exists() else None

# return a node value ('nodes' is the output of device.nodemap.get_nodes(['some nodes', 'here'])
# get_nval = lambda nodes, nstr: nodes[nstr].value

## diagnostic / utility functions (check node properties at a glance in console / vscode view window)
def triton_exists(): 
	# FIXME: possible unintended behaviour if this is ever used in a try/except block (will just return True without blocking execution with the exception)
	if triton is None: raise Exception('Camera not initialised')
	return True
get_node = lambda nstr: triton.nodemap.get_node([nstr])[nstr] if triton_exists() else None
def diag_attr_print(attribute='value', nodestr=['OffsetX', 'OffsetY', 'Width', 'Height', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain', 'PixelFormat']):
	for nstr in (nodestr if isinstance(nodestr, list) else [nodestr]): print(f'{nstr} {attribute}: {getattr(get_node(nstr), attribute)}')
'''search the list of all nodes for any instances of the input string "nstr"'''
def nodesearch(nstr):
	triton_exists()
	allnodes = triton.nodemap._Nodemap__get_feature_names()
	for node in allnodes: 
		if nstr in node: print(node)

## DEPRECATED FUNCTIONS

def view(device):
	nodemap = device.nodemap
	nodes = nodemap.get_node(['PixelFormat', 'ExposureTime'])
	num_channels = 1 if nodes['PixelFormat'].value == 'Mono8' else 3
	exposure = nodes['ExposureTime'].value

	curr_frame_time = 0
	prev_frame_time = 0
	print("press Enter to begin capture, Esc to abort capture")
	with device.start_stream():
		"""
		Infinitely fetch and display buffer data until esc is pressed (if not running in headless mode)
		"""
		while True:
			# Used to display FPS on stream
			curr_frame_time = time.time()

			buffer = device.get_buffer()
			"""
			Copy buffer and requeue to avoid running out of buffers
			"""
			item = BufferFactory.copy(buffer)
			device.requeue_buffer(buffer)

			img = triton_buffer_item_to_cv2_img(item)

			# display saturation
			sat_img = hf.gs3c(img[:,:,0].copy()) # get rid of third dimension of npndarray for sat_img
			sat_img[img[:,:,0] == 255] = (0,0,255) # for npndarray pixels that are saturated, set the sat_img pixel to (0,0,255) (red)
			oversat, undersat = hf.sat_pct(sat_img) # get the % of pixels that are saturated
			cv2.putText(sat_img, str(np.max(img)), (7, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # disp max pixel value
			cv2.putText(sat_img, str(oversat), (7, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # disp % of saturated pixels

			split_view = hf.pol_preview(img[:,:,0]) # split the polarization angles and display them in the corners of the image
			# display fps
			fps = str(1/(curr_frame_time - prev_frame_time))
			cv2.putText(split_view, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA) # fps

			#imshow on the pi - put `export QT_QPA_PLATFORM=xcb` in ~/.bashrc to avoid wayland qt plugin nonsense
			scale_pct = 30
			# display fps and saturation with those titles as two columns on a plt figure
			# plt.subplot(1,2,1)
			# plt.imshow(img_res(split_view, scale_pct))
			# plt.title('Camera View')
			# plt.subplot(1,2,2)
			# plt.imshow(img_res(sat_img, scale_pct))
			# plt.title('Saturation')
			# plt.show()
			
			# TODO: fix qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/home/apollo/.local/lib/python3.12/site-packages/cv2/qt/plugins"
			preview_img = np.hstack((hf.gs3c(split_view), sat_img))
			cv2.imshow('Preview', hf.img_res(preview_img, scale_pct))
			# cv2.imshow('Camera View', img_res(npndarray, scale_pct)) # display resized image
			# cv2.imshow('Saturation', img_res(sat_img, scale_pct))
			
			"""
			Destroy the copied item to prevent memory leaks
			"""
			BufferFactory.destroy(item)

			prev_frame_time = curr_frame_time

			"""
			Break if enter key is pressed, break and throw an exception (i.e. stop the full code from running) if the esc key is pressed
			"""
			# key = cv2.waitKey(int(exposure/1000))
			key = cv2.waitKey(100)
			# if the enter key is pressed
			proceed = True
			if key == 13:
				print("Continuing to Image Capture")
				break

			# if the esc key is pressed
			if key == 27:
				print("Image Capture Aborted")
				proceed = False
				break

		device.stop_stream()
		cv2.destroyAllWindows()
		# plt.close('All')
		# cv2.destroyWindow('Saturation')
		
		# if key == 27:
		# 	terminate(device)
		# 	exit

	return proceed

def snap(device, img_name, img_format, pixel_format=PixelFormat.Mono8): # save (and return) a picture
	"""
	Takes a picture (absent any other stream); saves the resulting image to the filesystem, and also returns it as an array
		inputs:
			device: a system.create_device object (see 'create_devices_with_tries' function above)
			img_path: path to where the image will be saved (without file extension), e.g. './Tests/images/img_001' without a '.png'
						if a blank path is entered, it will not save
			img_format: a string of the image's desired file extension (e.g. .png)
			pixel_format: valid pixel format for camera (e.g. RGB8, Mono8, PolarizedAngles_0d_45d_90d_135d_Mono8)
	"""
	# ensure that pixel_format is a PixelFormat enum
	if not isinstance(pixel_format, PixelFormat):
		raise Exception("pixel_format must be a PixelFormat enum")
	
	# 1 channel if Mono8, 3 channels if RGB8, 
	# 4 channels if PolarizedAngles_0d_45d_90d_135d_Mono8, 3 channels if PolarizedStokes_S0_S1_S2_Mono8, 2 channels if PolarizedDolpAolp_Mono8
	if pixel_format == PixelFormat.Mono8:
		num_channels = 1
	elif pixel_format == PixelFormat.RGB8:
		num_channels = 3
	elif pixel_format == PixelFormat.PolarizedAngles_0d_45d_90d_135d_Mono8:
		num_channels = 4
	elif pixel_format == PixelFormat.PolarizedStokes_S0_S1_S2_Mono8:
		num_channels = 3
	elif pixel_format == PixelFormat.PolarizedDolpAolp_Mono8:
		num_channels = 2
	else:
		raise Exception("pixel_format has an unknown number of channels")

	img_path = img_name + "." + img_format # e.g. path/to/image.png

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	device.start_stream()

	buffer = device.get_buffer()

	# prepare local copy of img to return
	nodemap = device.nodemap
	nodes = nodemap.get_node(['PixelFormat', 'ExposureTime'])
	
	item = BufferFactory.copy(buffer)
	buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
	array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
	img = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width))#, buffer_bytes_per_pixel)) # probably 1 channel
	# img = np.moveaxis(img, 2, 0) # move the channel axis to the front

	# save image (if path is not empty)
	if img_name != '': save(buffer, img_path, pixel_format) 

	device.requeue_buffer(buffer)

	# Clean up
	device.stop_stream()

	# check that the image was saved with a while loop but time out after 5 seconds with an error
	start = time.time()
	if img_name != '':
		while not os.path.exists(img_path):
			time.sleep(0.1)
			if time.time() - start > 5:
				raise Exception("Image was not saved")
			
	if img is None:
		raise Exception("Image not returnable")
	
	# return the image array and the actual exposure value
	exposure = nodes['ExposureTime'].value

	return img, exposure # 

def grab(device, pixel_format=PixelFormat.Mono8): # returns a picture (absent any other image stream) but doesn't save it
	# ensure that pixel_format is a PixelFormat enum
	if not isinstance(pixel_format, PixelFormat):
		raise Exception("pixel_format must be a PixelFormat enum")
	
	# 1 channel if Mono8, 3 channels if RGB8, 
	# 4 channels if PolarizedAngles_0d_45d_90d_135d_Mono8, 3 channels if PolarizedStokes_S0_S1_S2_Mono8, 2 channels if PolarizedDolpAolp_Mono8
	if pixel_format == PixelFormat.Mono8:
		num_channels = 1
	elif pixel_format == PixelFormat.RGB8:
		num_channels = 3
	elif pixel_format == PixelFormat.PolarizedAngles_0d_45d_90d_135d_Mono8:
		num_channels = 4
	elif pixel_format == PixelFormat.PolarizedStokes_S0_S1_S2_S3_Mono8:
		num_channels = 4
	elif pixel_format == PixelFormat.PolarizedDolpAolp_Mono8:
		num_channels = 2
	else:
		raise Exception("pixel_format has an unknown number of channels")

	'''
	Setup stream values
	'''
	tl_stream_nodemap = device.tl_stream_nodemap
	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
	tl_stream_nodemap['StreamPacketResendEnable'].value = True

	device.start_stream()

	buffer = device.get_buffer()

	# prepare local copy of img to return
	nodemap = device.nodemap
	nodes = nodemap.get_node(['PixelFormat'])
	
	item = BufferFactory.copy(buffer)
	img = triton_buffer_item_to_cv2_img(item)
	device.requeue_buffer(buffer)

	# Clean up
	device.stop_stream()

	return img # if that doesn't work - TODO: retrieve image from file system

# TODO: have just the one camera object that lives inside this file and is passed to all functions that need it (but not externally)
# TODO: is_cropped = lambda _: True if triton.nodemap.get_node(['OffsetX'])['OffsetX'].value != 0 or triton.nodemap.get_node(['OffsetY'])['OffsetY'].value != 0 else False

# Run main function only if camera.py (this code) is called directly. If it is called from within a code as a library main does not run.
if __name__ == '__main__':
	main()

## CODE GRAVEYARD
''' old version of calibrate which creates and destroys the stream every time a new picture is taken '''
# def calibrate(device, cam_zenith):
# 	'''
# 	This function should take a picture from the camera and display it with the current area of interest from `analysis.py` overlaid as a rectangle
# 	If 'settings_history.csv' exists, load the last AoI values from that file
# 	Prompt the user to input the camera zenith angle
# 	Print to the console how the user can interact with the preview
# 	The user should then be able to keyboard input the following (and see the changes in real time):
# 		wasd: 		move the top left corner of the rectangle 
# 		arrow keys: move the bottom right corner of the rectangle
# 						(for both of the above, the new AoI values for each edge should be displayed next to that edge)
# 		p: 			take a new picture
# 		esc: 		quit the test
# 		enter: 		set `analysis.AoI_Bounds' to the new values, save the new AoI values and camera zenith 'settings_history.csv' along with a timestamp',
# 					and finish executing the function as per normal
# 	'''
# 	# TODO: on-the-fly camera settings adjustment

# 	# if the settings file exists, load the most recent AoI_bounds
# 	load_AoI_settings(vars.settings_file)

# 	img = grab(device)
# 	img2 = img.copy()

# 	AoI_move = 0.005 # proportion of the image to move the AoI by
# 	print("Adjust the Area of Interest (AoI) to the desired bounds")
# 	print("wasd: move the top left corner of the rectangle")
# 	print("ijkl: move the bottom right corner of the rectangle")
# 	print("p: take a new picture")
# 	print("esc: quit the test")
# 	print("enter: set the AoI values and camera zenith, save the settings, and continue the test")
# 	while True:
# 		img2 = img.copy()
# 		img_preview = gs3c(img_res(img2, 50)) # 3-channel, 30% scale
# 		x1, x2 = [int(vars.AoI_Bounds[vars.X1]*img_preview.shape[1]), int(vars.AoI_Bounds[vars.X2]*img_preview.shape[1])]
# 		y1, y2 = [int(vars.AoI_Bounds[vars.Y1]*img_preview.shape[0]), int(vars.AoI_Bounds[vars.Y2]*img_preview.shape[0])]

# 		# draw the AoI rectangle
# 		cv2.rectangle(img_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 		## show AoI values next to each edge of the rectangle: 
# 		X1pos = (x1, y1 + (y2-y1)//2) # vars.X1 between [x1,y1] and [x2, y1] left of the rectangle edge,
# 		X2pos = (x2, y1 + (y2-y1)//2) # vars.X2 between [x1,y2] and [x2, y2] right of the rectangle edge,
# 		Y1pos = (x1 + (x2-x1)//2, y1) # vars.Y1 between [x1,y1] and [x1, y2] above the rectangle edge,
# 		Y2pos = (x1 + (x2-x1)//2, y2) # vars.Y2 between [x2,y1] and [x2, y2] below the rectangle edge
		
# 		cv2.putText(img_preview, f"{vars.AoI_Bounds[vars.X1]:.2f}", X1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# 		cv2.putText(img_preview, f"{vars.AoI_Bounds[vars.X2]:.2f}", X2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# 		cv2.putText(img_preview, f"{vars.AoI_Bounds[vars.Y1]:.2f}", Y1pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
# 		cv2.putText(img_preview, f"{vars.AoI_Bounds[vars.Y2]:.2f}", Y2pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		
# 		# show the image, draw the AoI, and get user input
# 		cv2.imshow('Preview', img_preview)

# 		new_AoI_Bounds = vars.AoI_Bounds.copy()
# 		# get user input
# 		key = cv2.waitKey(0)
		
# 		# if the 'p' key is pressed, update the base image
# 		if key == 112:
# 			# TODO: getting a new image glitches the display (seems to have stuff from img_preview in it even though it's new)
# 			print("Taking new image...")
# 			img = grab(device)
# 			print("New image taken")

# 		# wasd: move the top left corner of the rectangle (whether upper or lower case)
# 		elif key == 119: new_AoI_Bounds[vars.Y1] -= AoI_move # w
# 		elif key == 97:  new_AoI_Bounds[vars.X1] -= AoI_move # a
# 		elif key == 115: new_AoI_Bounds[vars.Y1] += AoI_move # s
# 		elif key == 100: new_AoI_Bounds[vars.X1] += AoI_move # d

# 		# ijkl: move the bottom right corner of the rectangle (whether upper or lower case)
# 		elif key == 105: new_AoI_Bounds[vars.Y2] -= AoI_move # i
# 		elif key == 106: new_AoI_Bounds[vars.X2] -= AoI_move # j
# 		elif key == 107: new_AoI_Bounds[vars.Y2] += AoI_move # k
# 		elif key == 108: new_AoI_Bounds[vars.X2] += AoI_move # l

# 		# if the 'esc' key is pressed
# 		elif key == 27:
# 			proceed = False
# 			print("Calibration aborted, exiting...")
# 			break

# 		# if the 'enter' key is pressed
# 		elif key == 13:
# 			proceed = True
# 			print("Calibration complete, proceeding...")
# 			break

# 		else: 
# 			pass # do nothing, wait for the next key press

# 		# AoI validation:
# 		img_bound_warnings = ["Warning: X1 is out of bounds", "Warning: X2 is out of bounds", "Warning: Y1 is out of bounds", "Warning: Y2 is out of bounds"]
# 		small_AoI_warning = "Warning: AoI is too small"
# 		valid = True
# 		# not checking for inversion because we're already checking that the difference between the two values is not too small
# 		# if the new AoI is too small, print a warning
# 		if new_AoI_Bounds[vars.X2] - new_AoI_Bounds[vars.X1] < 0.01 or new_AoI_Bounds[vars.Y2] - new_AoI_Bounds[vars.Y1] < 0.01:
# 			print(small_AoI_warning)
# 			valid = False
# 		# if all checks pass, update the AoI values
# 		# if new_AoI_bounds is not within the bounds of preview_img, print a warning
# 		for i in range(len(new_AoI_Bounds)):
# 			if new_AoI_Bounds[i] < 0 or new_AoI_Bounds[i] > 1:
# 				print(img_bound_warnings[i])
# 				valid = False
# 		if valid:
# 			vars.AoI_Bounds = new_AoI_Bounds
	
# 	vars.AoI_Bounds = [round(i, 2) for i in vars.AoI_Bounds] # round to 2 decimal places before saving
	
# 	# if the settings are good: save the settings [datetimestamp, camera_zenith, AoI_x1, AoI_x2, AoI_y1, AoI_y2] to 'settings_history.csv'
# 	if proceed: 
# 		date_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
# 		settings_data = [date_time, cam_zenith, vars.AoI_Bounds[vars.X1], vars.AoI_Bounds[vars.X2], vars.AoI_Bounds[vars.Y1], vars.AoI_Bounds[vars.Y2]]
# 		# create the settings_history.csv file if it doesn't exist, then append the settings_data to it
# 		if not os.path.exists(f'{vars.settings_file}.csv'): csv_init(vars.settings_file, vars.settings_fields)
# 		csv_append(vars.settings_file, settings_data)
			
# 	# if the user wants to quit, exit the entire program
# 	return proceed

## OLD SETUP
# def setup(device, gain, exposure, width, height, offx, offy, pixel_format=PixelFormat.Mono8):
# 	"""
# 	Setup stream dimensions and stream nodemap
# 		inputs:
# 			device: a system.create_device object (see 'create_devices_with_tries' function above)
# 			gain: ISO value in db (0-24 analog, 24-48 digital)
# 			exposure: exposure time in us (~0 to ~40000)
# 			width: image width in px (typically MAXWIDTH)
# 			height: image width in px (typically MAXHEIGHT)
# 			offx: x offset value for camera-level cropping TODO: implement this
# 			offy: y offset value for camera-level cropping TODO: implement this
# 			pixel_format: valid pixel format for camera (e.g. RGB8, Mono8, PolarizedAngles_0d_45d_90d_135d_Mono8)
# 	"""
# 	"""
# 	num_channels changes based on the PixelFormat
# 		Mono 8 has 1 channel, RGB8 has 3 channels
# 	"""
# 	if not isinstance(pixel_format, PixelFormat):
# 		raise Exception("pixel_format must be a PixelFormat enum")
	
# 	##               TECHNICAL DOCUMENTATION: https://support.thinklucid.com/triton-tri050s/
# 	nodemap = device.nodemap
# 	nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat'])

# 	# width and height < 2448, 2048 is just cropped towards the top left
# 	# need nodes['OffsetX' and 'OffsetY'] to change the crop location, probably
# 	nodes['Width'].value = width#1280
# 	nodes['Height'].value = height#720
# 	nodes['PixelFormat'].value = pixel_format#'Mono8'#'RGB8'

# 	## OPTIONAL CAMERA SETTINGS
# 	if(1):
# 		nodes2 = nodemap.get_node(['ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain'])#'ConversionGain']) # conversion gain seems to take 'Low' or 'High'
# 		# exposure settings (see py_exposure.py)
# 		if isinstance(exposure, int): exposure = float(exposure) # if exposure is an int, make it a float
# 		if(exposure > -1):                                                             ### <-- EXPOSURE ###
# 			exposure_time = exposure # 4000.0 from example - change as desired
# 			exposure_auto_initial = nodes2['ExposureAuto'].value
# 			exposure_time_initial = nodes2['ExposureTime'].value
# 			nodes2['ExposureAuto'].value = 'Off'
# 			# make sure the exposure time node exists and is writable
# 			if nodes2['ExposureTime'] is None:
# 				raise Exception("Exposure Time node not found")
# 			if nodes2['ExposureTime'].is_writable is False:
# 				raise Exception("Exposure Time node not writeable")
			
# 			# make sure the desired exposure time is within the maximum and minimum bounds
# 			if exposure_time > nodes2['ExposureTime'].max:
# 				nodes2['ExposureTime'].value = nodes['ExposureTime'].max
# 			elif exposure_time < nodes2['ExposureTime'].min:
# 				nodes2['ExposureTime'].value = nodes['ExposureTime'].min
# 			else:
# 				nodes2['ExposureTime'].value = exposure_time
# 		else:
# 			nodes2['ExposureAuto'].value = 'Continuous' # reset to initial value

# 		# gain settings (high/low?); this might only work on Helios cameras
# 		if isinstance(gain, int): gain = float(gain) # if gain is an int, make it a float
# 		if(gain > -1):                                                              ### <-- GAIN ###
# 			nodes2['GainAuto'].value = 'Off'   
			
# 			# make sure the gain node exists and is writable
# 			if nodes2['Gain'] is None:
# 				raise Exception("Gain node not found")
# 			if nodes2['Gain'].is_writable is False:
# 				raise Exception("Gain node not writeable")
			
# 			# make sure the desired exposure time is within the maximum and minimum bounds
# 			if gain > nodes2['Gain'].max:
# 				nodes2['Gain'].value = nodes['Gain'].max
# 			elif gain < nodes2['Gain'].min:
# 				nodes2['Gain'].value = nodes['Gain'].min
# 			else:
# 				nodes2['Gain'].value = gain
# 		else:
# 			nodes2['GainAuto'] = 'On' # reset to initial value]

# 	num_channels = 1 if nodes['PixelFormat'].value == 'Mono8' else 3

# 	# Stream nodemap
# 	tl_stream_nodemap = device.tl_stream_nodemap

# 	tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
# 	tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
# 	tl_stream_nodemap['StreamPacketResendEnable'].value = True

# 	# return num_channels


## save that handles external drives
# def save(buffer, impath, pixel_format):
# 	'''
# 	demonstrates saving an image
# 	(1) converts image to a displayable pixel format
# 	(2) prepares image parameters
# 	(3) prepares image writer
# 	(4) saves image
# 	(5) destroys converted image
# 	'''
# 	'''
# 	Convert image
# 		Convert the image to a displayable pixel format. It is worth keeping in
# 		mind the best pixel and file formats for your application. This example
# 		converts the image so that it is displayable by the operating system.
# 	'''
# 	save_debug = True
# 	external_drive = [False, True][impath.startswith('/media/')] # e.g. if the image path is on a USB drive
# 	if external_drive:
# 		# requeue the buffer and save it with cv2
# 		item = BufferFactory.copy(buffer)
# 		triton.requeue_buffer(buffer)
# 		cv2_img = triton_buffer_item_to_cv2_img(item)
# 		BufferFactory.destroy(item)
# 		# save the image with cv2
# 		cv2.imwrite(impath, cv2_img)
# 	else: 
# 		converted = BufferFactory.convert(buffer, pixel_format)
# 		if save_debug: print(f"{TAB1}Converted image to {pixel_format}")

# 		'''
# 		Prepare image writer
# 			The image writer requires 2 parameters to save an image: the buffer and
# 			specified file name or pattern. Default name for the image is
# 			'image_<count>.jpg' where count is a pre-defined tag that gets updated
# 			every time a buffer image.
# 		'''
# 		if save_debug: print(f'{TAB1}Prepare Image Writer')
# 		writer = Writer()
# 		writer.pattern = impath
# 		# writer.pattern = 'images/image_<count>.jpg'

# 		# Save converted buffer
# 		writer.save(converted)
# 		if save_debug: print(f'{TAB1}Image saved')

# 		# Destroy converted buffer to avoid memory leaks
# 		BufferFactory.destroy(converted)


## get_image_from_stream that handles external drives with cv2
# def get_img_from_stream(img_name='', img_format='png', pixel_format='Mono8'): #PixelFormat.Mono8): # preferred method of getting images
# 	'''
# 	Grab a single image an ongoing camera stream
# 		inputs: 
# 			a system.create_device object
# 			img_name: optionally a file path without the extension if the image is to be saved
# 			img_format: file extension for the image to be saved
# 			pixel_format: optionally a pixel format other than PixelFormat.Mono8 if trying something exotic
# 		outputs: 
# 			a single image from the camera stream (currently only Mono8 is supported, so only the first channel is returned)
# 			optionally saves the image
	
# 	Usage in other scripts/functions:
# 		def other_fun():
# 			with device.start_stream():
# 				while True:
# 					img, exp = get_img_from_ongoing_stream(device)
# 			device.stop_stream()
# 	'''
# 	triton_exists()
# 	if not triton.tl_stream_nodemap['StreamIsGrabbing'].value: 
# 		print("Stream not running; starting stream...")
# 		triton.start_stream() # start the stream if it's not already running

# 	# TODO: if img_format has a '.' in it, REMOVE IT

# 	# get camera settings
# 	nodemap = triton.nodemap
# 	nodes = nodemap.get_node(['PixelFormat', 'ExposureTime'])
# 	num_channels = 1 if nodes['PixelFormat'].value == 'Mono8' else Exception("Pixel format not supported (only Mono8 supported)")

# 	# get the image buffer, optionally save it, requeue the buffer aftewards
# 	buffer = triton.get_buffer() 
# 	item = BufferFactory.copy(buffer)
# 	img = None

# 	exposure = nodes['ExposureTime'].value
# 	# modify img_name with the *actual* exposure time
# 	if img_name != '': 
# 		img_name = hf.param_change(img_name, 'exp', exposure) + '_RAW' # param_change strips the 'RAW'/'HSV' out
# 		# find and delete any duplicates (based on im_num)
# 		duplicates = hf.find_duplicate_im_num_imgs(img_name)
# 		if duplicates: [os.remove(f) for f in duplicates] # true if duplicates is not an empty list (duplicate file names are full paths)
# 		# save image if given image path; must save before buffer requeued (handle saving to external drives differently)
# 		if img_name.startswith('/media/'): 
# 			img = triton_buffer_item_to_cv2_img(item)
# 			img_path = f'{img_name.replace(':','_')}.{img_format}'
# 			if not cv2.imwrite(img_path, img[:,:,0]): raise Exception(f"Image not saved for {img_name}.{img_format}")
# 		else: save(buffer, f'{img_name}.{img_format}', pixel_format) # save image if given image path; must save before buffer requeued
	
# 	triton.requeue_buffer(buffer) # requeue buffer to avoid running out of buffers
# 	if not isinstance(img, np.ndarray): img = triton_buffer_item_to_cv2_img(item) # don't re-get the image if it's already been saved
# 	# cleanup
# 	BufferFactory.destroy(item) # destroy the copied item to prevent memory leaks

# 	# return the image array and the actual exposure value
# 	if debug: hf.function_print(''.join([f'{nstr}: {get_node(nstr).value}, ' for nstr in set(changed_nodes)]))
# 	return img[:,:,0], exposure # only want first two dimensions of image, despite the above