import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from uuid import uuid4

from helpers.sci_style import *

# THEME_PRIMARY = "#4589ff"

def new_canvas_key():
	st.session_state['canvas_key'] = f'canvas_{str(uuid4())}'

@st.cache_data(show_spinner=False)
def resize(img: Image, max_height: int = 500, max_width: int = 500):
	'''
	Resize the image to be a max of 500x500 by default
	Args:
		img (Image): Image to be resized
		max_height (int, optional): Max height of the image. Defaults to 500.
		max_width (int, optional): Max width of the image. Defaults to 500.
	Returns:
		img (Image): Resized image
		ratio (float): Ratio of the resized image to the original image
	'''
	ratio = 1
	if img.height > max_height:
		ratio = max_height / img.height
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	if img.width > max_width:
		ratio = max_width / img.width
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))

	return img, ratio

def crop_and_calibrate(
		  img_original,
		  initial_roi_pos=(0.1, 0.1, 0.8, 0.7),
		  initial_scalebar_pos=(0.1, 0.9, 0.9, 0.9),
		  initial_scalebar_length=100,
		  initial_scalebar_units='nm'):
	'''
	Crop the image to the region of interest and calibrate the scalebar
	Args:
		img_original (Image): Original image
		initial_roi_pos (tuple, optional): Initial position of the ROI. Defaults to (0.1, 0.1, 0.8, 0.7).
		initial_scalebar_pos (tuple, optional): Initial position of the scalebar. Defaults to (0.1, 0.9, 0.9, 0.9).
		initial_scalebar_length (int, optional): Initial length of the scalebar. Defaults to 100.
		initial_scalebar_units (str, optional): Initial units of the scalebar. Defaults to 'nm'.
	Returns:
		img_cropped (Image): Cropped image
		scalebar_length (float): Length of the scalebar in pixels
		scalebar_units (str): Units of the scalebar
	'''

	if 'canvas_key' not in st.session_state.keys():
		new_canvas_key()

	img_resized, scalefactor = resize(img_original)
	col_original_img, col_img_settings = st.columns([3,1])

	with col_img_settings:
		scalebar_length = st.number_input("Scalebar length", value=initial_scalebar_length)
		scalebar_units = st.text_input("Scalebar units", value=initial_scalebar_units)
		st.markdown("<hr/>", unsafe_allow_html=True)

		# Set drawing mode
		drawing_action = st.selectbox("Drawing action", ['Move / Delete', 'Add ROI', 'Add scalebar'])

		if 'drawing_mode' not in st.session_state.keys():
			st.session_state['drawing_mode'] = 'transform'
		if drawing_action == 'Move / Delete':
			st.session_state['drawing_mode'] = 'transform'
		elif drawing_action == 'Add ROI':
			st.session_state['drawing_mode'] = 'rect'
		elif drawing_action == 'Add scalebar':
			st.session_state['drawing_mode'] = 'line'

	# Add a column to contain original image
	with col_original_img:
		initial_drawing = {
			'version': '4.4.0',
			'objects': [
				{
				'type': 'line',
				'x1': img_resized.width*initial_scalebar_pos[0], 'y1': img_resized.height*initial_scalebar_pos[1],
				'x2': img_resized.width*initial_scalebar_pos[2], 'y2': img_resized.height*initial_scalebar_pos[3],
				'fill': '#00000000', 'stroke': THEME_PRIMARY, 'strokeWidth': 4
				},
				{
				'type': 'rect',
				'left': img_resized.width*initial_roi_pos[0], 'top': img_resized.height*initial_roi_pos[1],
				'width': img_resized.width*initial_roi_pos[2], 'height': img_resized.height*initial_roi_pos[3],
				'fill': '#00000000', 'stroke': THEME_PRIMARY, 'strokeWidth': 4
				}
			]
		}

		canvas_result = st_canvas(
			key = st.session_state['canvas_key'],
			background_image = img_resized,
			height = img_resized.height,
			width = img_resized.width,
			drawing_mode = st.session_state['drawing_mode'],
			display_toolbar = False,
			initial_drawing = initial_drawing,
			fill_color = '#00000000',
			stroke_color = THEME_PRIMARY,
			stroke_width = 4
		)
		st.caption("Doubleclicking objects will remove them.")

	try:
		crop_rect = [d for d in canvas_result.json_data['objects'] if d['type']=='rect'][0]	
	except:
		st.error("Oops! You've removed your ROI, please add an ROI to continue.")
		return None
			
	crop_left = crop_rect['left']
	crop_top = crop_rect['top']
	crop_right = crop_left + crop_rect['width']*crop_rect['scaleX']
	crop_bottom = crop_top + crop_rect['height']*crop_rect['scaleY']
	if crop_left < 0:
		crop_left = 0
	if crop_top < 0:
		crop_top = 0
	if crop_right > img_resized.width:
		crop_right = img_resized.width
	if crop_bottom > img_resized.height:
		crop_bottom = img_resized.height
	
	try:
		scalebar_line = [d for d in canvas_result.json_data['objects'] if d['type']=='line'][0]
	except:
		st.error("Oops! You've removed your scalebar, please add a scalebar to continue.")
		return None
	
	scalebar_left = scalebar_line['left']
	scalebar_top = scalebar_line['top']
	scalebar_right = scalebar_left + scalebar_line['width']*scalebar_line['scaleX']
	scalebar_bottom = scalebar_top + scalebar_line['height']*scalebar_line['scaleY']
	scalebar_length_px = np.sqrt((scalebar_right-scalebar_left)**2 + (scalebar_bottom-scalebar_top)**2)

	img_cropped = img_original.crop((
		int((crop_left / img_resized.width) * img_original.width),
		int((crop_top / img_resized.height) * img_original.height),
		int((crop_right / img_resized.width) * img_original.width),
		int((crop_bottom / img_resized.height) * img_original.height)
	))

	return {
		'img_cropped': img_cropped,
		'scalefactor': scalefactor,
		'scalebar_length_px': scalebar_length_px,
		'scalebar_length': scalebar_length,
		'scalebar_units': scalebar_units
	}

	