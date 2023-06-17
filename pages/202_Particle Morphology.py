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
from skimage.measure import regionprops, regionprops_table
from skimage.segmentation import watershed
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage as ndi


import os
os.environ["OMP_NUM_THREADS"] = '1'

from helpers import sci_setup, sci_data, sci_image
sci_setup.setup_page("Particle Morphology")

import warnings
warnings.filterwarnings('ignore')

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

img_test = "./assets/public_data/Particle Analysis - Test2.tif"


# ---- Functions ----

@st.cache_data(show_spinner=False)
def open_img(img_file):
	img = Image.open(img_file)
	return img

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
	
	processed_images = {}

	# ---- Load Image ----

	with st.expander("Setup", expanded=True):
		img_file = st.file_uploader("Upload Image", type=FILETYPES_IMG, label_visibility="collapsed", on_change=sci_image.new_canvas_key)
		if not img_file:
			img_file = img_test
		img_original = open_img(img_file)
		img_original = img_original.convert("RGB")

		# Set initial values for ROI and scalebar
		initial_roi_pos = (0.1, 0.1, 0.85, 0.45)
		initial_scalebar_pos = (0.66, 0.85, 0.725, 0.85)
		initial_scalebar_length = 20
		initial_scalebar_units = "nm"
		# Perform cropping and calibration on original image
		crop_and_calibrate = sci_image.crop_and_calibrate(
			img_original, initial_roi_pos, initial_scalebar_pos, initial_scalebar_length, initial_scalebar_units)
		# Extract the output of the crop and calibrate function
		img_cropped = crop_and_calibrate['img_cropped']
		scalebar_length = crop_and_calibrate['scalebar_length']
		scalebar_units = crop_and_calibrate['scalebar_units']
		scalebar_length_px = crop_and_calibrate['scalebar_length_px']
		scalefactor = crop_and_calibrate['scalefactor']

	with st.expander("Process", expanded=True):
		col_process_img, col_process_settings = st.columns([3, 1])
		# ---- Image processing setting ----

		with col_process_settings:
			st.markdown("### Settings")
			apply_invert = st.checkbox("Invert Image", value=True, help="Image processing requires light particles on a dark background.")
			gaussian_kernel = st.slider("Gaussian Blur Kernel", 1, 100, 3, 2)
			threshold_value = st.slider("Threshold Value", 0, 255, 90, 1)
			threshold_histogram = st.container()
			particle_detection_method = st.selectbox("Particle Detection Method", ["Segmentation", "Watershed"])
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

		img_processed = crop_and_calibrate['img_cropped']
		img_processed = np.array(img_processed)
		# img_processed = img_processed[int(crop_top*img_orig_arr.shape[0]):int(crop_bottom*img_orig_arr.shape[0]), int(crop_left*img_orig_arr.shape[1]):int(crop_right*img_orig_arr.shape[1])]
		processed_images['Original'] = cv2.cvtColor(img_processed, cv2.COLOR_RGB2RGBA)

		# Convert to grayscale
		img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
		processed_images['Gray'] = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGBA)

		# Apply inversion if selected
		if apply_invert:
			img_processed = cv2.bitwise_not(img_processed)	
			processed_images['Invert'] = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGBA)
		
		# Apply Gaussian blur
		img_processed = cv2.GaussianBlur(img_processed, (gaussian_kernel, gaussian_kernel), 0)
		processed_images['Gaussian Blur'] = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGBA)

		# Apply bilateral filter
		# img_processed = cv2.bilateralFilter(img_processed, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)
		# processed_images['Bilateral filtered'] = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGBA)
		
		# Apply threshold
		hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
		fig = go.Figure(data=[go.Bar(x=np.arange(0, 256), y=hist[:,0])])
		fig.update_layout(dict(
			margin= dict(l=0, r=0, t=0, b=0),
			height=100,
			yaxis=dict(
				showticklabels=False,
				visible=False
			),
		))
		threshold_histogram.plotly_chart(fig, use_container_width=True)
		_, img_processed = cv2.threshold(img_processed, threshold_value, 255, cv2.THRESH_BINARY)
		# img_processed = cv2.threshold(img_processed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
		processed_images['Threshold'] = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGBA)

		# Apply particle detection method
		if  particle_detection_method == "Segmentation":
			# Apply segmentation
			label_img = measure.label(img_processed)

		elif particle_detection_method == "Watershed":
			# Calculate distance transform
			distance = ndi.distance_transform_edt(img_processed)
			# Save normalized distance
			processed_images['Watershed - Distance'] = (distance / (distance.max() - distance.min()) * 255).astype(np.uint8)
			# Find local maxima in the distance image
			max_coords = peak_local_max(distance, labels=img_processed, footprint=np.ones((3, 3)))
			local_maxima = np.zeros_like(img_processed, dtype=bool)
			local_maxima[tuple(max_coords.T)] = True
			markers = ndi.label(local_maxima)[0]
			# Apply watershed
			label_img = watershed(-distance, markers, mask=img_processed)
		
		# Remove small particles
		label_img = morphology.remove_small_objects(label_img, min_size=10)

		# Remove particles touching the border
		if remove_border_particles:
			label_img = segmentation.clear_border(label_img)

		img_label_overlay = label2rgb(label_img, image=img_processed)
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

	show_properties = [
		'area', 'equivalent_diameter_area', 'perimeter',
		'major_axis_length', 'minor_axis_length',
		'solidity', 'eccentricity']
	hide_properties = ['label', 'coords', 'bbox']
	
	properties = hide_properties + show_properties

	df_regions = pd.DataFrame(columns=properties)
	for region in regions:
		row = {}
		for prop in properties:
			if prop == 'coords':
				row[prop] = region.coords.tolist()
			elif prop == 'bbox':
				row[prop] = region.bbox
			else:
				row[prop] = getattr(region, prop)
		df_regions = df_regions.append(row, ignore_index=True)
	df_regions = df_regions.astype({'area': float})

	# for each region, extract the particle mask
	# particle_masks = []
	# for region in regions:
	# 	particle_mask = np.zeros_like(img_processed, dtype=bool)
	# 	particle_mask[tuple(region.coords.T)] = True
	# 	particle_masks.append(particle_mask)
	# 	df_regions.loc[df_regions['label'] == region.label, 'mask'] = [particle_mask]

	with st.expander("Particle Data"):
		st.markdown("### Raw Data")
		st.dataframe(df_regions[show_properties], use_container_width=True)
		st.markdown("### Statistics")
		st.dataframe(df_regions[show_properties].describe(percentiles=[0.1, 0.5, 0.9]), use_container_width=True)

	with st.expander("Data Visualization", expanded=True):
		st.markdown("### Settings")

		# Scale properties to range 0-1
		scaler = MinMaxScaler()
		df_regions_scaled = pd.DataFrame(scaler.fit_transform(df_regions[show_properties]), columns=show_properties)
		# add back in the hide properties
		df_regions_scaled = pd.concat([df_regions[hide_properties], df_regions_scaled], axis=1)

		col_copy_particles_from, col_variable_1, col_variable_2 = st.columns(3)
		with col_copy_particles_from:
			copy_particles_from = st.selectbox("Particles From", processed_images.keys(), index=len(processed_images)-1)
			img_copy_particles_from = processed_images[copy_particles_from]
		with col_variable_1:
			variable_1 = st.selectbox("Variable 1", show_properties, index=0)
		with col_variable_2:
			variable_2 = st.selectbox("Variable 2", show_properties, index=len(show_properties)-1)

		# Sort particles by variable 1 and 2
		df_regions_sorted = df_regions_scaled.sort_values(by=[variable_1, variable_2])
		
		st.markdown("<hr>", unsafe_allow_html=True)
		# col_l, col_c, col_r = st.columns([1,4,1])
		# with col_c:
		# tab_particle_grid,\
		# tab_particles_as_markers,\
		# tab_plot_data,\
		# tab_histogram,\
		# 	= st.tabs(["Particle grid", "Particles as Markers", "Plot Data", "Histogram"])

		data_visualization_selection = st.selectbox("Data Visualization Selection", ["Particle Grid", "Particle Plot", "Plot Data", "Histogram"], index=0)
		
		# with tab_particle_grid:
		if data_visualization_selection == "Particle Grid":
			img_grid_particles = regions_to_grid(img_copy_particles_from, df_regions_sorted)
			st.image(img_grid_particles, use_column_width=True, clamp=True)

		# with tab_particles_as_markers:
		elif data_visualization_selection == "Particle Plot":
			img_plot_particles = regions_to_plot(img_copy_particles_from, df_regions_sorted, variable_1, variable_2)
			st.image(img_plot_particles, use_column_width=True, clamp=True)
			
		# with tab_plot_data:
		elif data_visualization_selection == "Plot Data":
			fig = go.Figure()
			fig.update_layout(
				xaxis_title=variable_1,
				yaxis_title=variable_2,
			)
			fig.add_trace(go.Scatter(x=df_regions[variable_1], y=df_regions[variable_2], mode='markers'))
			st.plotly_chart(fig, use_container_width=True)


			import matplotlib.pyplot as plt
			from matplotlib.offsetbox import OffsetImage, AnnotationBbox
			fig, ax = plt.subplots()

			for x0, y0, img in zip(df_regions[variable_1], df_regions[variable_2], img_copy_particles_from[:50, :50, :]):
				ab = AnnotationBbox(OffsetImage(img, zoom=0.1), (x0, y0), frameon=False)
				ax.add_artist(ab)
				
			st.pyplot(fig)



		# with tab_histogram:
		elif data_visualization_selection == "Histogram":
			fig = go.Figure()
			fig.update_layout(
				xaxis_title=variable_1,
				yaxis_title="Count",
			)
			fig.add_trace(go.Histogram(x=df_regions[variable_1]))
			st.plotly_chart(fig, use_container_width=True)

@st.cache_data()
def regions_to_plot(img: Image, df_regions: pd.DataFrame, variable_1: str, variable_2: str):
	# Plot particle data as scatter plot where points are particles and x and y axis are properties
	# Create new image
	img_plot_particles = np.zeros_like(img)
	# Triple the size of the image to make room for overlapping particles
	img_plot_particles = cv2.resize(img_plot_particles, (img_plot_particles.shape[1]*3, img_plot_particles.shape[0]*3))

	# Paste each particle into new image at the position based on its properties
	for _, region in df_regions.iterrows():
		scaled_x = region[variable_1]
		scaled_y = region[variable_2]
		# Extract particle from original image within the bounding box
		minr, minc, maxr, maxc = region['bbox']
		particle_bbox = img[minr:maxr, minc:maxc].copy()

		# Create a mask for the particle within the bounding box
		bbox_coords = np.array(region['coords']) - [minr, minc]
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
		x_start = int(scaled_x * (img_plot_particles.shape[1] - particle_width))
		y_start = int(scaled_y * (img_plot_particles.shape[0] - particle_height))

		# Invert y-axis
		y_start = img_plot_particles.shape[0] - y_start - particle_height
		
		# Paste particle into new image without background
		img_plot_particles[y_start:y_start+particle_height, x_start:x_start+particle_width] = np.where(particle, particle, img_plot_particles[y_start:y_start+particle_height, x_start:x_start+particle_width])
		
	return img_plot_particles

@st.cache_data()
def regions_to_grid(img: Image, df_regions: pd.DataFrame):
	# Get max width and height of bounding boxes from df_regions['bbox']
	max_bbox_width = int(df_regions['bbox'].apply(lambda bbox: bbox[2] - bbox[0]).max())
	max_bbox_height = int(df_regions['bbox'].apply(lambda bbox: bbox[3] - bbox[1]).max())
	# Calculate number of rows and columns
	n_cols = 10
	n_rows = int(np.ceil(len(df_regions) / n_cols)) + 1
	# Create blank image to paste particles into allowing space for label under each particle
	img_grid_particles = np.zeros((n_rows * (max_bbox_height + 20), n_cols * max_bbox_width, 4))

	# Iterate over regions in df_regions
	i = 0
	for _, region in df_regions.iterrows():
		# Calculate row and column of particle
		row = i // n_cols
		col = i % n_cols
		# Extract particle from original image within the bounding box
		minr, minc, maxr, maxc = region['bbox']
		particle_bbox = img[minr:maxr, minc:maxc]
		# Create a mask for the particle within the bounding box
		bbox_coords = np.array(region['coords']) - [minr, minc]
		mask = np.zeros_like(particle_bbox, dtype=bool)
		mask[bbox_coords[:, 0], bbox_coords[:, 1]] = True
		# Apply the mask to the bounding box to isolate the particle
		particle = np.where(mask, particle_bbox, 0)
		
		# Paste particle into center of grid cell
		cell_center_x = col * max_bbox_width + max_bbox_width // 2
		cell_center_y = row * (max_bbox_height + 20) + max_bbox_height // 2
		x1 = cell_center_x - particle.shape[1] // 2
		x2 = x1 + particle.shape[1]
		y1 = cell_center_y - particle.shape[0] // 2
		y2 = y1 + particle.shape[0]
		img_grid_particles[y1:y2, x1:x2] = particle

		# Add label centered at bottom of grid cell
		label = str(region['label'])
		label_width, label_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)[0]
		label_x = cell_center_x - label_width // 2
		label_y = (row + 1) * (max_bbox_height + 20) - label_height // 2
		cv2.putText(img_grid_particles, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 255), 1, cv2.LINE_AA)
		i += 1

	return img_grid_particles

if __name__ == "__main__":
	main()