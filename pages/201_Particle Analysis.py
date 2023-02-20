import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from pprint import pprint
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
import math

import os
os.environ["OMP_NUM_THREADS"] = '1'

from helpers import sci_setup, sci_data
sci_setup.setup_page("Particle Analysis")

import warnings
warnings.filterwarnings('ignore')

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

img_test = "./assets/public_data/Particle Analysis - Test1.png"

# @st.experimental_singleton
# def add_to_session_state():
# 	st.session_state['add_roi'] = False
# 	st.session_state['add_scalebar'] = False
# add_to_session_state()

# ---- Functions ----

def img_segragation(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

	# noise removal
	kernel = np.ones((3,3),np.uint8)
	opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

	# sure background area
	sure_bg = cv2.dilate(opening,kernel,iterations=3)

	# Finding sure foreground area
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

	# Finding unknown region
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)

	# Marker labelling
	ret, markers = cv2.connectedComponents(sure_fg)

	# Add one to all labels so that sure background is not 0, but 1
	markers = markers+1

	# Now, mark the region of unknown with zero
	markers[unknown==255] = 0

	markers = cv2.watershed(img,markers)
	img[markers == -1] = [255,0,0]

st.cache_data(show_spinner=False)
def detect_particles(img, params):
	diameters = []

	if params['invert_val']:
		img = ImageOps.invert(img)
	# img = img.convert("L")
	img = np.array(img)
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	b = 1 + params["blur_val"]*2 
	img = cv2.GaussianBlur(img, (b,b), 0)

	img_output = img.copy()
	img_output = cv2.cvtColor(img_output, cv2.COLOR_GRAY2RGB)

	circles = cv2.HoughCircles(
		image=img,
		method=cv2.HOUGH_GRADIENT_ALT, 
		dp=params["dp_val"], 
		param1=params["param1_val"],
		param2=params["param2_val"], 
		minDist=params["min_dist_val"], 
		minRadius=int(params["diameter_val"][0]/2), 
		maxRadius=int(params["diameter_val"][1]/2),
		)
	if circles is not None:
		for circle in circles[0]:
			x, y, r = circle
			cv2.circle(img_output, (int(x), int(y)), int(r), (69, 137, 255), 2)
			cv2.circle(img_output, (int(x), int(y)), 2, (255, 255, 255), 2)
			diameters.append(2 * r)

	return img_output, circles, diameters

@st.cache_data(show_spinner=False)
def resize_img(img: Image, max_height: int=500, max_width: int=500):
	# Resize the image to be a max of 500x500 by default
	ratio = 1
	if img.height > max_height:
		ratio = max_height / img.height
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	if img.width > max_width:
		ratio = max_width / img.width
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	
	return img, ratio

st.cache_data(show_spinner=False)
def plot_mixture(gmm, X, show_legend=True, ax=None):
	if ax is None:
		ax = plt.gca()

		# Compute PDF of whole mixture
		upper = 10**(math.ceil(math.log10(max(X))))
		x = np.linspace(0, upper, 1000)
		print(10**math.ceil(math.log10(max(X))))
		logprob = gmm.score_samples(x.reshape(-1, 1))
		pdf = np.exp(logprob)

		# Compute PDF for each component
		responsibilities = gmm.predict_proba(x.reshape(-1, 1))
		pdf_individual = responsibilities * pdf[:, np.newaxis]
		# Plot data histogram
		ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4, label='Data')

		# Plot PDF of whole model
		ax.plot(x, pdf, '-k', label='Mixture PDF')

		# Plot PDF of each component
		ax.plot(x, pdf_individual, '--', label='Component PDF')
		ax.set_xlabel('$x$')
		ax.set_ylabel('$p(x)$')
		if show_legend:
			ax.legend()

@st.cache_data(show_spinner=False)
def open_img(path):
	return Image.open(path)

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)

	# container_left, container_center, container_right = st.columns([1,3,1])
	# with container_center:

	st.markdown("### Setup")
	
	with st.expander("Upload image, add ROI, and define scale", expanded=True):
		img_file = st.file_uploader(label='Upload image file', type=FILETYPES_IMG, label_visibility='collapsed')

		if not img_file:
			img_file = img_test
			st.caption("The example shown here is of silica coated gold nanoparticles. The analyzer distinguishes three distinct populations for core, shell, and contaminant silica particles.")
		img_original = open_img(img_file)
		img_original = img_original.convert("RGB")
		img = img_original.copy()
		img, scalefactor = resize_img(img_original)

		col_original_img, col_img_settings = st.columns([3,1])
		
		# Add a column to contain image settings
		with col_img_settings:
			scale_val = st.number_input("Scalebar length", value=500)
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
					'type': 'line', 'originX': 'left', 'originY': 'top',
					'x1': img.width*0.68, 'y1': img.height*0.85,
					'x2': img.width*0.84, 'y2': img.height*0.85,
					'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
					},
					{
					'type': 'rect', 'originX': 'left', 'originY': 'top',
					'left': img.width*0.1, 'top': img.height*0.1,
					'width': img.width*0.8, 'height': img.height*0.65,
					'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
					}
				]
			}

			canvas_result = st_canvas(
				key = "canvas",
				background_image = img,
				height = img.height,
				width = img.width,
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
			int((crop_left / img.width) * img_original.width),
			int((crop_top / img.height) * img_original.height),
			int((crop_right / img.width) * img_original.width),
			int((crop_bottom / img.height) * img_original.height)
			))
		
		try:
			scalebar_line = [d for d in canvas_result.json_data['objects'] if d['type']=='line'][0]
		except:
			st.write("Oops! You've removed your scalebar, please add a scalebar to continue.")
			return None

		scalebar_px = scalebar_line['width'] * scalebar_line['scaleX']

	st.markdown("<hr/>", unsafe_allow_html=True)
	
	st.markdown("### Detection")
	
	with st.expander("Detection settings", expanded=True):
		col_detected_particles, col_detection_settings = st.columns([3,1])

		with col_detection_settings:
			blur_val = st.slider("Blur", min_value=0, max_value=20, value=4,
			help="Level of Gaussian blur applied to image")

			invert_val = st.checkbox("Invert",
				help="Invert the image. Should not affect detection but can make viewing easier") 

			dp_val = st.slider("DP", min_value=1.0, max_value=2.0, value=1.2,
				help="Inverse ratio of the accumulator resolution to the image resolution.")

			param1_val = st.slider("Param 1", min_value=50, max_value=500, value=100,
				help="The higher threshold of the two passed to the Canny edge detector")

			param2_val = st.slider("Param 2", min_value=0.0, max_value=1.0, value=0.8,
				help="The circle 'perfectness' measure")

			# min_dist_val = st.slider(f"Min distance ({scale_units_val})", min_value=1, max_value=500, value=10,
			# 	help="Minimum distance between the centers of the detected circles.")

			# diameter_val = st.slider(f"Diameter ({scale_units_val})", min_value=1, max_value=500, value=(10,300),
			# 	help="Minimum and maximum circle diameter.")

			min_dist_val = st.number_input(
				f"Min distance ({scale_units_val})",
				min_value=float(scale_val/1000),
				max_value=float(scale_val*1000),
				value=float(scale_val/10),
				help="Minimum distance between the centers of the detected circles.")

			min_diameter_val = st.number_input(
				f"Min diameter ({scale_units_val})",
				min_value=float(scale_val/1000),
				max_value=float(scale_val*1000),
				value=float(scale_val/100),
				help="Minimum circle diameter.")

			max_diameter_val = st.number_input(
				f"Max diameter ({scale_units_val})",
				min_value=float(scale_val/1000),
				max_value=float(scale_val*1000),
				value=float(scale_val),
				help="Maximum circle diameter.")

		detection_settings = {
			"blur_val": blur_val,
			"invert_val": invert_val,
			"dp_val": dp_val,
			"param1_val": param1_val,
			"param2_val": param2_val,
			# "min_dist_val": min_dist_val,
			# "diameter_val": diameter_val,
			"min_dist_val": min_dist_val / (scalefactor * scale_val/scalebar_px),
			"diameter_val": [i / (scalefactor * scale_val/scalebar_px) for i in [min_diameter_val, max_diameter_val]],
		}
		
		with col_detected_particles:
			img_output, circles, diameters_px = detect_particles(img_crop, detection_settings)
			st.image(img_output)
			diameters_units = [i * (scale_val/scalebar_px) * scalefactor for i in diameters_px]

		st.caption("More information on circle detection parameters can be found [here](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d)")

		particle_count = len(diameters_units)
		# particle_mean = np.mean(diameters_units)
		# particle_std = np.std(diameters_units)

		st.markdown(f"""
			### Particles detected: {particle_count}
			""")

	xs, ys, rs = [],[],[]
	for circle in circles[0]:
		x, y, r = circle
		xs.append(x)
		ys.append(y)
		rs.append(r)
	with st.expander(label="Particles"):
		df = pd.DataFrame(
			{f'Diameter ({scale_units_val})': diameters_units,
			'x (px)':xs,
			'y (px)': ys}
			# columns=[f'Diameter ({scale_units_val})']
		)
		df.index += 1
		st.dataframe(df)

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Output")
	
	with st.expander("Distribution of detected particles", expanded=True):
		col_dist_plot, col_dist_plot_settings = st.columns([3,1])

		# Add a column to contain dist plot settings
		with col_dist_plot_settings:
			max_components_val = st.number_input("Max components", value=3,
			help="The AIC will be calculated for each population size up to 'max components'.")

		counts, bin_edges = np.histogram(diameters_units, bins=int(np.sqrt(len(diameters_units))))
		bin_widths = np.diff(bin_edges)
		X = np.array(diameters_units).reshape(-1, 1)

		k_arr = np.arange(max_components_val) + 1
		models = [GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X) for k in k_arr]

		# Compute metrics to determine best hyperparameter
		AIC = [m.aic(X) for m in models]
		# BIC = [m.bic(X) for m in models]
		N = np.argmin(AIC)
		gmm_best = models[N]

		# Compute PDF of whole mixture
		upper = 10**(math.ceil(math.log10(max(X))))
		x = np.linspace(0, upper, 1000)
		logprob = gmm_best.score_samples(x.reshape(-1, 1))
		pdf = np.exp(logprob)

		# Compute PDF for each component
		responsibilities = gmm_best.predict_proba(x.reshape(-1, 1))
		pdf_individual = responsibilities * pdf[:, np.newaxis]

		with col_dist_plot:
			# Plot distribution
			fig = ff.create_distplot(
				[diameters_units],
				group_labels=["Diameters"],
				bin_size=bin_widths,
				curve_type='normal',
				show_curve=False,
				# show_rug=False,
				colors=[PRIMARY_COLOR])
			# Plot Best GMM
			fig.add_trace(go.Scatter(x=x, y=pdf, name="Best GMM"))
			# Plot each component
			for i, y in enumerate(list(zip(*pdf_individual))):
				fig.add_trace(go.Scatter(x=x, y=y, name=f"Component {i+1}"))

			fig.layout.template = 'plotly_dark'
			fig.layout.xaxis.title.text = f'Particle diameter ({scale_units_val})'
			fig.layout.legend.traceorder = 'normal'
			fig.layout.margin = dict(l=20, r=20, t=20, b=20)
			st.plotly_chart(fig, use_container_width=True)
		
		st.markdown("<hr/>", unsafe_allow_html=True)
		
		means = gmm_best.means_
		covs = gmm_best.covariances_
		stds = [ np.sqrt(  np.trace(covs[i])/(N+1)) for i in range(0,N+1) ]

		df = pd.DataFrame(data={
			f"Mean ({scale_units_val})": [round(i[0],2) for i in means],
			f"Standard deviation ({scale_units_val})": [round(i,2) for i in stds],		
		})
		df.sort_values(f"Mean ({scale_units_val})", inplace=True)
		df.reset_index(drop=True, inplace=True)
		df.index += 1
		st.dataframe(df)

	with st.expander(label='Akaike information criterion'):
		st.markdown('''
		The Akaike Information Criterion (AIC) determines the optimum trade-off between model error and size.
		
		In this case, the 'size' is the number of populations.
		''')

		# Plot AIC
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=k_arr, y=AIC, name="AIC"))
		fig.layout.template = 'plotly_dark'
		fig.layout.xaxis.title.text = 'Number of compoents (k)'
		fig.layout.legend.traceorder = 'normal'
		fig.layout.margin = dict(l=20, r=20, t=20, b=20)
		st.plotly_chart(fig, use_container_width=True)

	st.markdown("<hr/>", unsafe_allow_html=True)


if __name__ == '__main__':
	main()
