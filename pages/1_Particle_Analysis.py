import streamlit as st
import cv2
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from pprint import pprint
import json
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import math
st.set_option('deprecation.showPyplotGlobalUse', False)

img_test = r"test_data\Particle analysis - Test 1.tif"

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']

PRIMARY_COLOR = "#4589ff"

st.title("Particle analysis")

def detect_particles(img, params):
	diameters = []
	img_output = img.copy()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	b = 1 + params["blur_val"]*2 
	img = cv2.GaussianBlur(img, (b,b), 0)

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

	return img_output, diameters

def resize_img(img: Image, max_height: int=600, max_width: int=600) -> Image:
    # Resize the image to be a max of 700x700 by default, or whatever the user 
    # provides. If streamlit has an attribute to expose the default width of a widget,
    # we should use that instead.
    if img.height > max_height:
        ratio = max_height / img.height
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    return img, ratio

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

detection_settings = {}

def main():

	global detection_settings

	img_file = st.sidebar.file_uploader(label='Upload an image to analyze', type=FILETYPES_IMG)

	with st.sidebar.expander("Image settings"):
		scale_val = st.number_input("Scalebar length", value=500)
		scale_units_val = st.text_input("Scalebar units", value="nm")


	with st.sidebar.expander("Circle detection settings"):
	# 	sb_col1, sb_col2 = st.columns(2)
	# 	with sb_col1:
	# 		detection_settings_file = st.file_uploader(label='Upload detection settings', type=["txt"])
	# 	with sb_col2:
	# 		st.download_button("Download detection settings", json.dumps(detection_settings))

		blur_val = st.slider("Blur", min_value=0, max_value=20, value=5)
		dp_val = st.slider("DP", min_value=1.0, max_value=2.0, value=1.5)
		min_dist_val = st.slider("Min distance", min_value=1, max_value=100, value=10)
		param1_val = st.slider("Param 1", min_value=50, max_value=500, value=85)
		param2_val = st.slider("Param 2", min_value=0.0, max_value=1.0, value=0.85)
		diameter_val = st.slider("Diameter (px)", min_value=1, max_value=500, value=(10,300))

	if not img_file:
		img_file = img_test
		# return None

	img_original = Image.open(img_file)
	img_original = img_original.convert("RGB")
	img = img_original.copy()
	img, scalefactor = resize_img(img_original)

	detection_settings = {
		"blur_val": blur_val,
		"dp_val": dp_val,
		"param1_val": param1_val,
		"param2_val": param2_val,
		"min_dist_val": min_dist_val,
		"diameter_val": diameter_val,
	}

	initial_drawing = {
		'version': '4.4.0',
		'objects': [
			{'type': 'rect', 'originX': 'left', 'originY': 'top',
			'left': img.width*0.25, 'top': img.height*0.25,
			'width': img.width*0.5, 'height': img.height*0.5,
			'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4,},

			{'type': 'line', 'originX': 'left', 'originY': 'top',
			'x1': img.width*0.67, 'y1': img.height*0.85,
			'x2': img.width*0.75, 'y2': img.height*0.85,
			'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4,}
			]}

	# col1, col2 = st.columns(2)

	# with col1:
	canvas_result = st_canvas(
	    key="canvas",
	    background_image = img,
	    height = img.height,
	    width = img.width,
	    drawing_mode = "transform",
	    display_toolbar = False,
	    initial_drawing = initial_drawing,
	)

	# canvas_result.json_data['objects'] = []
	# print(canvas_result.json_data)

	if not canvas_result.json_data:
		return None

	crop_rect = [d for d in canvas_result.json_data['objects'] if d['type']=='rect'][0]
	crop_left = crop_rect['left']
	crop_top = crop_rect['top']
	crop_right = crop_left + crop_rect['width']*crop_rect['scaleX']
	crop_bottom = crop_top + crop_rect['height']*crop_rect['scaleY']
	# (print(crop_left, crop_top, crop_right, crop_bottom))
	img_crop =  img_original.crop((
		int((crop_left / img.width) * img_original.width),
		int((crop_top / img.height) * img_original.height),
		int((crop_right / img.width) * img_original.width),
		int((crop_bottom / img.height) * img_original.height)
		))

	scalebar_line = [d for d in canvas_result.json_data['objects'] if d['type']=='line'][0]
	scalebar_px = scalebar_line['width'] * scalebar_line['scaleX']

	# with col2:
	img_crop = np.array(img_crop)
	img_output, diameters_px = detect_particles(img_crop, detection_settings)
	st.image(img_output)

	diameters_units = [i * (scale_val/scalebar_px) * scalefactor for i in diameters_px]

	counts, bin_edges = np.histogram(diameters_units, bins=int(np.sqrt(len(diameters_units))))
	bin_widths = np.diff(bin_edges)
	X = np.array(diameters_units).reshape(-1, 1)

	max_pops = 2

	k_arr = np.arange(max_pops) + 1
	models = [GaussianMixture(n_components=k).fit(X) for k in k_arr]

	# Compute metrics to determine best hyperparameter
	AIC = [m.aic(X) for m in models]
	BIC = [m.bic(X) for m in models]
	gmm_best = models[np.argmin(AIC)]

	# Compute PDF of whole mixture
	upper = 10**(math.ceil(math.log10(max(X))))
	x = np.linspace(0, upper, 1000)
	logprob = gmm_best.score_samples(x.reshape(-1, 1))
	pdf = np.exp(logprob)

	# Compute PDF for each component
	responsibilities = gmm_best.predict_proba(x.reshape(-1, 1))
	pdf_individual = responsibilities * pdf[:, np.newaxis]

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
		fig.add_trace(go.Scatter(x=x, y=y, name=f"Compoent {i}"))

	fig.layout.xaxis.title.text = f'Particle diameter ({scale_units_val})'
	fig.layout.legend.traceorder = 'normal'
	st.plotly_chart(fig, use_container_width=True)

	particle_mean = np.mean(diameters_units)
	particle_std = np.std(diameters_units)
	particle_count = len(diameters_units)

	st.markdown(f"""
		Particles detected: {particle_count} | Mean: {round(particle_mean,2)} ± {round(particle_std,2)} {scale_units_val}.
		""")
	st.caption("Mean and SD currently assume a single population")

if __name__ == '__main__':
	main()