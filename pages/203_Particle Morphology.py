import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import imutils

import plotly.graph_objects as go

from skimage.segmentation import watershed
from skimage import measure
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.feature import peak_local_max
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

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
	
	col_img, col_settings = st.columns([1,1])

	processed_images = {}
	
	with col_settings:
		tab_load, tab_process = st.tabs(["Load", "Process"])
		with tab_load:
			img_path = st.file_uploader("Upload Image", label_visibility="collapsed")
			if not img_path:
				img_path = img_test

			st.write("Crop Image")
			col_top, col_left, col_bottom, col_right = st.columns([1,1,1,1])
			with col_top:
				crop_top = st.number_input("Top", value=0.11, min_value=0.0, max_value=1.0, step=0.01)
			with col_left:
				crop_left = st.number_input("Left", value=0.12, min_value=0.0, max_value=1.0, step=0.01)
			with col_bottom:
				crop_bottom = st.number_input("Bottom", value=0.56, min_value=0.0, max_value=1.0, step=0.01)
			with col_right:
				crop_right = st.number_input("Right", value=0.95, min_value=0.0, max_value=1.0, step=0.01)

			st.write("Calibration")
			col_startx, col_starty, col_endx, col_endy = st.columns([1,1,1,1])
			with col_startx:
				cal_start_x = st.number_input("Start X", value=0.664, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
			with col_starty:
				cal_start_y = st.number_input("Start Y", value=0.85, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
			with col_endx:
				cal_end_x = st.number_input("End X", value=0.724, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
			with col_endy:
				cal_end_y = st.number_input(
					"End Y", value=0.85, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")
			
			col_cal_value, col_cal_unit = st.columns([1,1])
			with col_cal_value:
				cal_value = st.number_input("Calibration Value", value=20.0, min_value=0.0, step=0.01, disabled=True)
			with col_cal_unit:
				cal_unit = st.selectbox("Calibration Unit", ["nm", "µm", "mm", "cm", "m"], disabled=True)

			# cal_length = np.sqrt((cal_end_x - cal_start_x)**2 + (cal_end_y - cal_start_y)**2)
			# cal_scale = cal_value / cal_length
			# st.write(f"Calibration Scale: {cal_scale:.2f} {cal_unit} per pixel")

		with tab_process:
			apply_invert = st.checkbox("Invert Image", value=True)
			gaussian_kernel = st.slider("Gaussian Blur Kernel", 1, 100, 3, 2)
			threshold_value = st.slider("Threshold Value", 0, 255, 90, 1)
			threshold_histogram = st.container()
			particle_detection_method = st.selectbox(
				"Particle Detection Method", ["Segmentation", "Watershed"])
			
			# col_bilateral_diameter, col_bilateral_sigma_color, col_bilateral_sigma_space = st.columns([1,1,1])
			# with col_bilateral_diameter:
			# 	bilateral_diameter = st.slider("Diameter", 1, 100, 11, 2)
			# with col_bilateral_sigma_color:
			# 	bilateral_sigma_color = st.slider("Sigma Color", 1, 100, 11, 2)
			# with col_bilateral_sigma_space:
			# 	bilateral_sigma_space = st.slider("Sigma Space", 1, 100, 11, 2)
			# apply_watershed = st.checkbox("Apply Watershed")
			# # apply_particle_analysis = st.checkbox("Apply Particle Analysis")



	# ---- Load and process image ----

	# Load image
	img_orig = open_img(img_path).convert("RGB")
	# Convert image to numpy array
	img_orig = np.array(img_orig)
	# Normalize image to a width of 1000 pixels
	img_orig = imutils.resize(img_orig, width=1000)
	# Create copies of the original image for annotation and processing
	img_annot = img_orig.copy()
	img_proc = img_orig.copy()

	# Add crop rectangle to the annotated image
	cv2.rectangle(img_annot, (int(crop_left*img_orig.shape[1]), int(crop_top*img_orig.shape[0])), (int(crop_right*img_orig.shape[1]), int(crop_bottom*img_orig.shape[0])), (100, 149, 237), 5)
	# Add calibration line to the annotated image
	cv2.line(img_annot, (int(cal_start_x*img_orig.shape[1]), int(cal_start_y*img_orig.shape[0])), (int(cal_end_x*img_orig.shape[1]), int(cal_end_y*img_orig.shape[0])), (100, 149, 237), 5)


	# ---- Process image and save result at each step ----

	# Crop the image
	img_proc = img_proc[int(crop_top*img_orig.shape[0]):int(crop_bottom*img_orig.shape[0]), int(crop_left*img_orig.shape[1]):int(crop_right*img_orig.shape[1])]
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

	img_label_overlay = label2rgb(label_img, image=img_proc)
	alpha = np.full((img_label_overlay.shape[0], img_label_overlay.shape[1]), 1, dtype=np.uint8)
	img_label_overlay = np.dstack((img_label_overlay, alpha))
	processed_images[particle_detection_method] = img_label_overlay.copy()


	# ---- Display images ----

	with col_img:
		tab_orig, tab_proc  = st.tabs(["Original", "Processed"])
		with tab_orig:
			st.image(img_annot, use_column_width=True)
		with tab_proc:
			processed_image_selection = st.selectbox("View Processed Image", processed_images.keys(), index=len(processed_images)-1)
			st.image(processed_images[processed_image_selection], use_column_width=True)
	

	# ---- Particle Analysis ----

	# Get particle properties
	regions = regionprops(label_img)
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
		mask = np.zeros_like(particle_bbox, dtype=np.bool)
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
		

	col_l, col_c, col_r = st.columns([1,4,1])
	with col_c:
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