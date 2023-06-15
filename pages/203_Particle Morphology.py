import streamlit as st
from streamlit_drawable_canvas import st_canvas

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import imutils
from uuid import uuid4

import plotly.graph_objects as go

from skimage import measure, morphology, segmentation
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage as ndi


import os
os.environ["OMP_NUM_THREADS"] = '1'

from helpers import sci_setup, sci_data
sci_setup.setup_page("Particle Morphology")

import warnings
warnings.filterwarnings('ignore')

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

img_test = "./assets/public_data/Particle Analysis - Test2.tif"


# ---- Functions ----

@st.cache_data(show_spinner=False)
def open_img(path):
	return Image.open(path)

@st.cache_data(show_spinner=False)
def resize_img(img: Image, max_height: int = 500, max_width: int = 500):
	# Resize the image to be a max of 500x500 by default
	ratio = 1
	if img.height > max_height:
		ratio = max_height / img.height
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	if img.width > max_width:
		ratio = max_width / img.width
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))

	return img, ratio

def new_canvas_key():
	st.session_state['canvas_key'] = f'canvas_{str(uuid4())}'

if 'canvas_key' not in st.session_state.keys():
	new_canvas_key()

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
	
	processed_images = {}

	# ---- Basic Layout ----

	# tab_setup, tab_process = st.tabs(["Setup", "Process"])
	# with tab_load:
	# 	col_setup_img, col_setup_settings = st.columns([3,1])
	# with tab_process:
	# 	col_process_img, col_process_settings = st.columns([3,1])

	# ---- Load Image ----

	with st.expander("Setup", expanded=True):
		img_file = st.file_uploader("Upload Image", type=FILETYPES_IMG, label_visibility="collapsed", on_change=new_canvas_key)

		if not img_file:
			img_file = img_test
		img_original = open_img(img_file)
		img_original = img_original.convert("RGB")
		img_resized = img_original.copy()
		img_resized, scalefactor = resize_img(img_original)

		col_original_img, col_img_settings = st.columns([3,1])

		with col_img_settings:
			scale_val = st.number_input("Scalebar length", value=20)
			scale_units_val = st.text_input("Scalebar units", value="nm")
			st.markdown("<hr/>", unsafe_allow_html=True)
			add_roi = st.checkbox("Add ROI", False)
			add_scalebar = st.checkbox("Add scalebar", False)

		drawing_mode = 'transform'
		if add_roi:
			drawing_mode = 'rect'
		if add_scalebar:
			drawing_mode = 'line'

		# Add a column to contain original image
		with col_original_img:
			initial_drawing = {
				'version': '4.4.0',
				'objects': [
					{
					'type': 'line',
					'x1': img_resized.width*0.66, 'y1': img_resized.height*0.85,
					'x2': img_resized.width*0.725, 'y2': img_resized.height*0.85,
					'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
					},
					{
					'type': 'rect',
					'left': img_resized.width*0.1, 'top': img_resized.height*0.1,
					'width': img_resized.width*0.85, 'height': img_resized.height*0.45,
					'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
					}
				]
			}

			canvas_result = st_canvas(
				key = st.session_state['canvas_key'],
				background_image = img_resized,
				height = img_resized.height,
				width = img_resized.width,
				drawing_mode = drawing_mode,
				display_toolbar = False,
				initial_drawing = initial_drawing,
				fill_color = '#00000000',
				stroke_color = PRIMARY_COLOR,
				stroke_width = 4
			)
			st.caption("Doubleclicking objects will remove them.")

		try:
			crop_rect = [d for d in canvas_result.json_data['objects'] if d['type']=='rect'][0]	
		except:
			st.write("Oops! You've removed your ROI, please add an ROI to continue.")
			return None
		crop_left = crop_rect['left']
		crop_top = crop_rect['top']
		crop_right = crop_left + crop_rect['width']*crop_rect['scaleX']
		crop_bottom = crop_top + crop_rect['height']*crop_rect['scaleY']
		img_crop =  img_original.crop((
			int((crop_left / img_resized.width) * img_original.width),
			int((crop_top / img_resized.height) * img_original.height),
			int((crop_right / img_resized.width) * img_original.width),
			int((crop_bottom / img_resized.height) * img_original.height)
			))

		try:
			scalebar_line = [d for d in canvas_result.json_data['objects'] if d['type']=='line'][0]
		except:
			st.write("Oops! You've removed your scalebar, please add a scalebar to continue.")
			return None

		scalebar_px = scalebar_line['width'] * scalebar_line['scaleX']

	# with col_setup_settings:
	# 	st.markdown("### Settings")
		
	# 	scale_val = st.number_input("Scalebar length", value=20, disabled=True)
	# 	scale_units_val = st.text_input("Scalebar units", value="nm", disabled=True)

	# 	st.markdown("<hr/>", unsafe_allow_html=True)

	# 	add_roi = st.checkbox("Add ROI", False)
	# 	add_scalebar = st.checkbox("Add scalebar", False)
		
	# 	if not img_file:
	# 		img_file = img_test

	# 	# Load image
	# 	img_orig = open_img(img_file)
	# 	img_orig = img_orig.convert('RGB')
	# 	# Normalize image to a width of 1000 pixels
	# 	img_resized = img_orig.copy()
	# 	img_resized, scalefactor = resize_img(img_orig)
	# 	# img_resized = np.array(img_orig)
	# 	# img_resized = imutils.resize(img_resized, width=500)
	# 	# img_resized = Image.fromarray(img_resized)
	# 	# img_resized, scalefactor = resize_img(img_orig)

	# 	drawing_mode = 'transform'
	# 	if add_roi:
	# 		drawing_mode = 'rect'
	# 	if add_scalebar:
	# 		drawing_mode = 'line'

	# # Add a column to contain original image
	# with col_setup_img:
	# 	# initial_drawing = {
	# 	# 	'version': '4.4.0',
	# 	# 	'objects': [
	# 	# 		{
	# 	# 		'type': 'line',
	# 	# 		'x1': img_resized.width*0.66, 'y1': img_resized.height*0.85,
	# 	# 		'x2': img_resized.width*0.725, 'y2': img_resized.height*0.85,
	# 	# 		'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
	# 	# 		},
	# 	# 		{
	# 	# 		'type': 'rect',
	# 	# 		'left': img_resized.width*0.1, 'top': img_resized.height*0.1,
	# 	# 		'width': img_resized.width*0.85, 'height': img_resized.height*0.45,
	# 	# 		'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
	# 	# 		}
	# 	# 	]
	# 	# }
	# 	initial_drawing = {
	# 		'version': '4.4.0',
	# 		'objects': [
	# 				{
	# 				'type': 'line', 'originX': 'left', 'originY': 'top',
	# 				'x1': img_resized.width*0.68, 'y1': img_resized.height*0.85,
	# 				'x2': img_resized.width*0.84, 'y2': img_resized.height*0.85,
	# 				'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
	# 				},
	# 				{
	# 				'type': 'rect', 'originX': 'left', 'originY': 'top',
	# 				'left': img_resized.width*0.1, 'top': img_resized.height*0.1,
	# 				'width': img_resized.width*0.8, 'height': img_resized.height*0.65,
	# 				'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
	# 				}
	# 		]
	# 	}		

	# 	canvas_result = st_canvas(
	# 		# key = st.session_state['canvas_key'],
	# 		key = 'canvas',
	# 		background_image = img_resized,
	# 		height = img_resized.height,
	# 		width = img_resized.width,
	# 		drawing_mode = drawing_mode,
	# 		display_toolbar = False,
	# 		initial_drawing = initial_drawing,
	# 		fill_color = '#00000000',
	# 		stroke_color = PRIMARY_COLOR,
	# 		stroke_width = 4
	# 	)
	# 	st.caption("Doubleclicking objects will remove them.")
	
	# 	try:
	# 		crop_rect = [d for d in canvas_result.json_data['objects'] if d['type']=='rect'][0]	
	# 	except:
	# 		st.error("Oops! You've removed your ROI, please add an ROI to continue.")
	# 		return None
				
	# 	crop_left = crop_rect['left']
	# 	crop_top = crop_rect['top']
	# 	crop_right = crop_left + crop_rect['width']*crop_rect['scaleX']
	# 	crop_bottom = crop_top + crop_rect['height']*crop_rect['scaleY']
	# 	if crop_left < 0:
	# 		crop_left = 0
	# 	if crop_top < 0:
	# 		crop_top = 0
	# 	if crop_right > img_resized.width:
	# 		crop_right = img_resized.width
	# 	if crop_bottom > img_resized.height:
	# 		crop_bottom = img_resized.height
		
	# 	try:
	# 		scalebar_line = [d for d in canvas_result.json_data['objects'] if d['type']=='line'][0]
	# 	except:
	# 		st.error("Oops! You've removed your scalebar, please add a scalebar to continue.")
	# 		return None
		
	# 	scalebar_left = scalebar_line['left']
	# 	scalebar_top = scalebar_line['top']
	# 	scalebar_right = scalebar_left + scalebar_line['width']*scalebar_line['scaleX']
	# 	scalebar_bottom = scalebar_top + scalebar_line['height']*scalebar_line['scaleY']
	# 	scalebar_px = np.sqrt((scalebar_right-scalebar_left)**2 + (scalebar_bottom-scalebar_top)**2)

	with st.expander("Process", expanded=True):
		col_process_img, col_process_settings = st.columns([3, 1])
		# ---- Image processing setting ----

		with col_process_settings:
			st.markdown("### Settings")
			apply_invert = st.checkbox("Invert Image", value=True, help="Image processing requires light particles on a dark background.")
			gaussian_kernel = st.slider("Gaussian Blur Kernel", 1, 100, 3, 2)
			threshold_value = st.slider("Threshold Value", 0, 255, 90, 1)
			threshold_histogram = st.container()
			particle_detection_method = st.selectbox(
				"Particle Detection Method", ["Segmentation", "Watershed"])
			remove_border_particles = st.checkbox("Remove Border Particles", value=True)
			# min_particle_area, max_particle_area = st.slider("Particle Area", 0, 10000, (10, 10000), 1)
			
			# col_bilateral_diameter, col_bilateral_sigma_color, col_bilateral_sigma_space = st.columns([1,1,1])
			# with col_bilateral_diameter:
			# 	bilateral_diameter = st.slider("Diameter", 1, 100, 11, 2)
			# with col_bilateral_sigma_color:
			# 	bilateral_sigma_color = st.slider("Sigma Color", 1, 100, 11, 2)
			# with col_bilateral_sigma_space:
			# 	bilateral_sigma_space = st.slider("Sigma Space", 1, 100, 11, 2)
			# apply_watershed = st.checkbox("Apply Watershed")
			# # apply_particle_analysis = st.checkbox("Apply Particle Analysis")


		# ---- Process image and save result at each step ----

		# Crop the image
		img_proc = img_original.crop((
			int((crop_left / img_resized.width) * img_original.width),
			int((crop_top / img_resized.height) * img_original.height),
			int((crop_right / img_resized.width) * img_original.width),
			int((crop_bottom / img_resized.height) * img_original.height)
		))
		img_proc = np.array(img_proc)
		# img_proc = img_proc[int(crop_top*img_orig_arr.shape[0]):int(crop_bottom*img_orig_arr.shape[0]), int(crop_left*img_orig_arr.shape[1]):int(crop_right*img_orig_arr.shape[1])]
		processed_images['Original'] = cv2.cvtColor(img_proc, cv2.COLOR_RGB2RGBA)

		# Convert to grayscale
		img_proc = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
		processed_images['Gray'] = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGBA)

		# Apply inversion if selected
		if apply_invert:
			img_proc = cv2.bitwise_not(img_proc)	
			processed_images['Invert'] = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGBA)
		
		# Apply Gaussian blur
		img_proc = cv2.GaussianBlur(img_proc, (gaussian_kernel, gaussian_kernel), 0)
		processed_images['Gaussian Blur'] = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGBA)

		# Apply bilateral filter
		# img_proc = cv2.bilateralFilter(img_proc, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
		# processed_images['Bilateral filtered'] = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGBA)
		
		# Apply threshold
		hist = cv2.calcHist([img_proc], [0], None, [256], [0, 256])
		fig = go.Figure(data=[go.Bar(x=np.arange(0, 256), y=hist[:,0])])
		fig.update_layout(dict(
			margin= dict(l=0, r=0, t=0, b=0),
			height=100,
			yaxis=dict(
				showticklabels=False,
				visible=False
			),
			# xaxis=dict(
			# 	showticklabels=False,
			# 	visible=False
			# ),
		))
		threshold_histogram.plotly_chart(fig, use_container_width=True)
		_, img_proc = cv2.threshold(img_proc, threshold_value, 255, cv2.THRESH_BINARY)
		# img_proc = cv2.threshold(img_proc, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		processed_images['Threshold'] = cv2.cvtColor(img_proc, cv2.COLOR_GRAY2RGBA)

		# Apply particle detection method
		if  particle_detection_method == "Segmentation":
			# Apply segmentation
			label_img = measure.label(img_proc)

		elif particle_detection_method == "Watershed":
			# Calculate distance transform
			distance = ndi.distance_transform_edt(img_proc)
			# Save normalized distance
			processed_images['Watershed - Distance'] = (distance / (distance.max() - distance.min()) * 255).astype(np.uint8)
			# Find local maxima in the distance image
			max_coords = peak_local_max(distance, labels=img_proc, footprint=np.ones((3, 3)))
			local_maxima = np.zeros_like(img_proc, dtype=bool)
			local_maxima[tuple(max_coords.T)] = True
			markers = ndi.label(local_maxima)[0]
			# Apply watershed
			label_img = watershed(-distance, markers, mask=img_proc)
		
		# Remove small particles
		label_img = morphology.remove_small_objects(label_img, min_size=10)

		# Remove particles touching the border
		if remove_border_particles:
			label_img = segmentation.clear_border(label_img)

		img_label_overlay = label2rgb(label_img, image=img_proc)
		alpha = np.full((img_label_overlay.shape[0], img_label_overlay.shape[1]), 1, dtype=np.uint8)
		img_label_overlay = np.dstack((img_label_overlay, alpha))
		processed_images[particle_detection_method] = img_label_overlay.copy()


		# ---- Display processed images ----

		with col_process_img:
			processed_image_selection = st.selectbox("View Processed Image", processed_images.keys(), index=len(processed_images)-1, label_visibility="collapsed")
			st.image(processed_images[processed_image_selection], use_column_width=True)
	

	# ---- Particle Analysis ----

	# Get particle properties
	regions = regionprops(label_img)
	if len(regions) == 0:
		st.error("No particles detected")
		return None

	properties = [
		'area', 'equivalent_diameter_area', 'perimeter',
		'major_axis_length', 'minor_axis_length',
		'solidity', 'eccentricity']

	df_particles = pd.DataFrame(columns=properties)
	for region in regions:
		df_particles = pd.concat([df_particles, pd.DataFrame([[getattr(region, prop) for prop in properties]], columns=properties)], ignore_index=True)
	
	with st.expander("Particle Data"):
		st.dataframe(df_particles, use_container_width=True)

	# Plot particle data as scatter plot where points are particles and x and y axis are properties
	col_copy_particles_from, col_x_axis, col_y_axis = st.columns(3)
	with col_copy_particles_from:
		copy_particles_from = st.selectbox("Particles From", processed_images.keys(), index=len(processed_images)-1)
		img_copy_particles_from = processed_images[copy_particles_from]
	with col_x_axis:
		x_axis = st.selectbox("X-Axis", properties, index=0)
	with col_y_axis:
		y_axis = st.selectbox("Y-Axis", properties, index=len(properties)-1)

	# Create new image
	new_image = np.zeros_like(img_copy_particles_from)
	# Triple the size of the image to make room for overlapping particles
	new_image = cv2.resize(new_image, (new_image.shape[1]*3, new_image.shape[0]*3))
	
	# Scale properties to range 0-1
	scaler = MinMaxScaler()
	df_particles_scaled = pd.DataFrame(scaler.fit_transform(df_particles), columns=df_particles.columns)

	# Paste each particle into new image at the position based on its properties
	for region_info, (scaled_x, scaled_y) in zip(regions, zip(df_particles_scaled[x_axis], df_particles_scaled[y_axis])):
		# Extract particle from original image within the bounding box
		minr, minc, maxr, maxc = region_info['bbox']
		particle_bbox = processed_images[copy_particles_from][minr:maxr, minc:maxc].copy()

		# Create a mask for the particle within the bounding box
		bbox_coords = region_info['coords'] - [minr, minc]
		mask = np.zeros_like(particle_bbox, dtype=bool)
		mask[bbox_coords[:, 0], bbox_coords[:, 1]] = True

		# Apply the mask to the bounding box to isolate the particle
		particle = np.where(mask, particle_bbox, 0)
		
		# Get particle size
		try:
			particle_height, particle_width = particle.shape
		except:
			particle_height, particle_width, _ = particle.shape

		# Calculate position in new image based on properties
		x_start = int(scaled_x * (new_image.shape[1] - particle_width))
		y_start = int(scaled_y * (new_image.shape[0] - particle_height))

		# Invert y-axis
		y_start = new_image.shape[0] - y_start - particle_height
		
		# Paste particle into new image without background
		new_image[y_start:y_start+particle_height, x_start:x_start+particle_width] = np.where(particle, particle, new_image[y_start:y_start+particle_height, x_start:x_start+particle_width])
		

	# col_l, col_c, col_r = st.columns([1,4,1])
	# with col_c:
	tab_particles_as_markers, tab_plot_data, tab_histogram  = st.tabs(["Particles as Markers", "Plot Data", "Histogram"])
	with tab_particles_as_markers:
		st.image(new_image, use_column_width=True, clamp=True)
	with tab_plot_data:
		fig = go.Figure()
		fig.update_layout(
			xaxis_title=x_axis,
			yaxis_title=y_axis,
		)
		fig.add_trace(go.Scatter(x=df_particles[x_axis], y=df_particles[y_axis], mode='markers'))
		st.plotly_chart(fig, use_container_width=True)
	with tab_histogram:
		fig = go.Figure()
		fig.update_layout(
			xaxis_title=x_axis,
			yaxis_title="Count",
		)
		fig.add_trace(go.Histogram(x=df_particles[x_axis]))
		st.plotly_chart(fig, use_container_width=True)
		

if __name__ == "__main__":
	main()