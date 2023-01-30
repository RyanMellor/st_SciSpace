import streamlit as st
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
from pprint import pprint
from colorutils import rgb_to_hsv, rgb_to_hex
from simplification.cutil import simplify_coords
import extcolors

from helpers import setup
setup.setup_page("Data Extractor")

img_test = Image.open("./assets/public_data/Data Extractor - Test1.png")

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

# ---- Functions ----


def map_data(data, inmin, inmax, outmin, outmax):
	return outmin + (outmax - outmin)*(data - inmin) / (inmax - inmin)


def lerp_gaps(data):
	for i in range(len(data)):
		if data[i]:
			continue
		for j in range(i, len(data)):

			if not(data[j]):
				continue
			start = data[i-1]
			if start:
				end = data[j]
				points = j-i
				gap = (end-start)/(points+1)
				data[i:j] = [start + (k+1)*gap for k in range(points)]
				break
			continue
	return data


def resize_img(img: Image, max_height: int = 600, max_width: int = 600):
	# Resize the image to be a max of 600x600 by default, or whatever the user
	# provides. If streamlit has an attribute to expose the default width of a widget,
	# we should use that instead.
	ratio = 1
	if img.height > max_height:
		ratio = max_height / img.height
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	if img.width > max_width:
		ratio = max_width / img.width
		img = img.resize((int(img.width * ratio), int(img.height * ratio)))
	return img, ratio


def main():

	st.markdown("<hr/>", unsafe_allow_html=True)

	img_file = st.file_uploader(
		label='Upload an image to analyze', type=FILETYPES_IMG)

	if not img_file:
		img_original = img_test
	else:
		img_original = Image.open(img_file)
	img = img_original.copy()
	img.convert('RGB')
	img, scalefactor = resize_img(img_original)

	# cvimg_gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
	# cvimg_hsv = cv2.cvtColor(cvimg, cv2.COLOR_BGR2HSV)
	# cvimg_rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

	# col_original_img, col_img_settings = st.columns([3,1])

	# Add a column to contain image settings
	# with col_img_settings:

	col_settings_labels, col_settings_axes, col_settings_detect = st.columns(3)

	with col_settings_labels:
		st.write("### Labels")
		chart_title = st.text_input('Chart title', "title")
		x_title = st.text_input('x-axis label', "x-axis")
		x_units = st.text_input('x-axis units', "")
		y_title = st.text_input('y-axis label', "y-axis")
		y_units = st.text_input('y-axis units', "")

	with col_settings_axes:
		st.write("### Axes")
		xmin = st.number_input('xmin', 0)
		xmax = st.number_input('xmax', 20)
		ymin = st.number_input('ymin', 0)
		ymax = st.number_input('ymax', 1)
		xlog = st.checkbox('x log', False)
		ylog = st.checkbox('y log', False)

	with col_settings_detect:
		st.write("### Detection")
		# minimalX = False # If True, only export Y values which have at least one associated Y
		epsilon = st.number_input('Epsilon', 0.0001, format="%.4f",
								  help="Epsilon parameter of Ramer–Douglas–Peucker algorithm iterative end-point fit algorithm. Increase for fewer data points")
		# [0, 1] Vertical slice of img used for series finding, should be a point where series do not overlap
		abscissa = st.number_input('Abscissa', 0.45)
		# Set how different two colours must be to be considered different series, default [1,60,30]
		hsv_tol = [1, 60, 3]
		# Set how different two colours must be to be considered different series, default []
		rgb_tol = [6]*3

	# Add a column to contain original image
	# with col_original_img:
	initial_drawing = {
		'version': '4.4.0',
		'objects': [
			{
				'type': 'rect', 'originX': 'left', 'originY': 'top',
				'left': img.width*0.06, 'top': img.height*0.03,
				'width': img.width*0.93, 'height': img.height*0.83,
				'fill': '#00000000', 'stroke': PRIMARY_COLOR, 'strokeWidth': 4
			}
		]
	}

	canvas_result = st_canvas(
		# key="canvas",
		background_image=img,
		height=img.height,
		width=img.width,
		drawing_mode="transform",
		display_toolbar=False,
		initial_drawing=initial_drawing,
	)
	st.caption(
		"Warning: Doubleclicking objects will remove them and you will have to refresh the page")

	if not canvas_result.json_data:
		return None
	try:
		crop_rect = [d for d in canvas_result.json_data['objects']
					 if d['type'] == 'rect'][0]
	except:
		st.write("Oops! You've removed your ROI, please refresh the page")
		return None
	crop_left = crop_rect['left']
	crop_top = crop_rect['top']
	crop_right = crop_left + crop_rect['width']*crop_rect['scaleX']
	crop_bottom = crop_top + crop_rect['height']*crop_rect['scaleY']
	img_crop = img_original.crop((
		int((crop_left / img.width) * img_original.width),
		int((crop_top / img.height) * img_original.height),
		int((crop_right / img.width) * img_original.width),
		int((crop_bottom / img.height) * img_original.height)
	))

	cvimg = np.array(img_crop)
	cvimg_height, cvimg_width = cvimg.shape[:2]
	st.image(cvimg)

	series = {}

	if xlog:
		xaxis = list(np.logspace(start=np.log10(xmin), stop=np.log10(xmax), num=cvimg_width))
	else:
		xaxis = [xmin+i*(xmax-xmin)/cvimg_width for i in range(cvimg_width)]
	if ylog:
		yaxis = np.logspace(start=np.log10(ymin), stop=np.log10(ymax), num=cvimg_height)
	else:
		yaxis = [ymin+i*(ymax-ymin)/cvimg_height for i in range(cvimg_height)]

	x_label = x_title
	if x_units != "":
		x_label += " (" + x_units + ")"
	y_label = y_title
	if y_units != "":
		y_label += " (" + y_units + ")"

	st.markdown("<hr/>", unsafe_allow_html=True)

	tolerance = st.number_input('Tolerance', 0, 100, 15)
	limit = st.number_input('Limit', 1, 30, 5)

	colors, pixel_count = extcolors.extract_from_image(
		img_crop, tolerance=tolerance, limit=limit)
	rgb_unique = [np.array(i[0]) for i in colors]

	for rgb_val in rgb_unique:
		if list(rgb_val) != [255, 255, 255]:
			line = len(series)
			series[line] = {}
			series[line]['rgb'] = rgb_val
			series[line]['hex'] = rgb_to_hex(rgb_val)
			series[line]['data'] = [[i, None] for i in xaxis]

	for line in series:
		line_rgb = (series[line]['rgb'])
		lower = np.array(line_rgb - rgb_tol)
		for a in range(len(lower)):
			if lower[a] < 0:
				lower[a] = 0
		upper = np.array(line_rgb + rgb_tol)
		for a in range(len(upper)):
			if upper[a] > 255:
				upper[a] = 255
		mask = cv2.inRange(cvimg, lower, upper)
		# res = cv2.bitwise_and(cvimg, cvimg, mask=mask)

		series[line]['mask'] = mask

		data = series[line]['data']
		h = len(mask)
		w = len(mask[0])

		for y in range(h):
			for x in range(w):
				if mask[y][x] == 0:
					continue
				elif data[x][1] == None:
					data[x][1] = [h - y]
					continue
				data[x][1] += [h - y]

		for x in range(w):
			if data[x][1] != None:
				data[x][1] = np.average(data[x][1])

		data = [[x, map_data(y, 0, cvimg_height, ymin, ymax)] for [x, y] in data if y]
		series[line]['data'] = data
		data_rdp = [i for i in data if i[1]]
		data_rdp = simplify_coords(data_rdp, epsilon)
		series[line]['data_rdp'] = data_rdp

	fig_raw_data = go.Figure()
	extracted_data = pd.DataFrame()

	for i, line in enumerate(series):
		col = series[line]['hex']
		data = series[line]['data_rdp']
		data = np.array([data])
		data_t = data.T
		x = [float(i) for i in data_t[0]]
		y = [float(i) for i in data_t[1]]
		fig_raw_data.add_trace(go.Scatter(x=x, y=y, marker_color=col))
		temp_df = pd.DataFrame(index=x, data=y, columns=[i])
		temp_df
		extracted_data = pd.concat([extracted_data, temp_df])

	fig_raw_data.layout.template = 'plotly_dark'
	fig_raw_data.layout.legend.traceorder = 'normal'
	fig_raw_data.layout.margin = dict(l=20, r=20, t=20, b=20)
	fig_raw_data.layout.xaxis.title.text = x_label
	fig_raw_data.layout.yaxis.title.text = y_label
	fig_raw_data.layout.legend.title.text = chart_title
	st.plotly_chart(fig_raw_data, use_container_width=True)

	extracted_data.sort_index(inplace=True)

	st.markdown("<hr/>", unsafe_allow_html=True)

	with st.expander("Extracted data"):
		st.download_button(
			label = "Download extracted data",
			data = extracted_data.to_csv().encode("utf-8"),
			file_name = "extracted_data.csv")
		st.dataframe(extracted_data)


if __name__ == '__main__':
	main()