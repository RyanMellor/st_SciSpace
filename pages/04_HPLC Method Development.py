import streamlit as st
from streamlit_drawable_canvas import st_canvas
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, AgGridTheme
from st_aggrid.grid_options_builder import GridOptionsBuilder

import os
import cv2
import json
import numpy as np
import pandas as pd
from PIL import Image
from pprint import pprint
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from lmfit import models
from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.integrate import trapz
import requests
from io import BytesIO
import urllib.request
import math
import copy


data_test = r"http://sci-space.co.uk//test_data/HPLC%20Method%20Development%20-%20Test%20data.xlsx"

data_template = r"http://sci-space.co.uk//test_data/HPLC%20Method%20Validation%20-%20Template.xlsx"
df_data_template =  pd.read_excel(data_template, index_col=0)
buffer_data_template = BytesIO()
with pd.ExcelWriter(buffer_data_template, engine='xlsxwriter') as writer:
	df_data_template.to_excel(writer)

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

# ---- Page setup ----
fav_url = r"http://sci-space.co.uk//favicon.ico"
fav_response = requests.get(fav_url)
fav = Image.open(BytesIO(fav_response.content))
st.set_page_config(
    page_title="HPLC Method Validation",
    page_icon=fav,
)

st.title("HPLC Method Validation")

page_setup = """
	<div>
		<a href="http://sci-space.co.uk/" target="_blank">
			<img src="http://sci-space.co.uk//scispace.png">
		</a>
		</p>
		<a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
			<img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
		</a>
	</div>
	<hr/>
	<style>
		footer {visibility: hidden;}
		thead tr th:first-child {display:none}
        tbody th {display:none}
	</style>
	"""
st.sidebar.markdown(page_setup, unsafe_allow_html=True,)

st.sidebar.download_button(
	'Download Data Template',
	data = buffer_data_template,
	file_name = 'SciSpace - HPLC Validation Template.xlsx'
	)

@st.cache()
def read_data(data_file, ext=None):
	if isinstance(data_file, st.runtime.uploaded_file_manager.UploadedFile):
		ext = os.path.splitext(data_file.name)[-1][1:]
	else:
		ext = os.path.splitext(data_file)[-1][1:]

	# TODO add options for pd.read_XXX to sidebar
	if ext in ['xls', 'xlsx']:
		return pd.read_excel(data_file, index_col=0)
	elif ext in ['csv', 'txt']:
		return pd.read_csv(data_file, index_col=0)
	else:
		return None

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)

	data_file = st.file_uploader(
		label='Upload raw data',
		type=['txt', 'csv', 'xls', 'xlsx'])
	if not data_file:
		data_file = data_test

	df_data_read = read_data(data_file)
	if df_data_read is None:
		return None

	# sample_names = [i for i in df_data_read.columns if i != "Baseline"]
	# runs = {}

	# for col in 

	# if subtract_baseline:
	# 	# st.markdown("Doing good science :thumbsup:")
	# 	df_data = df_data_read[sample_names].sub(df_data_read["Baseline"], axis=0)
	# else:
	# 	df_data = df_data_read
	df_baseline = df_data_read[[i for i in df_data_read.columns if 'Baseline' in i]]
	df_baseline.columns = ["_".join(i.split('_')[:-1]) for i in df_baseline.columns]
	df_data = df_data_read[[i for i in df_data_read.columns if not 'Baseline' in i]]
	df_data.columns = ["_".join(i.split('_')[:-1]) for i in df_data.columns]

	df_data = df_data.sub(df_baseline)
	
	samples = pd.DataFrame(columns=['samples'], data=df_data.columns)
	col_dataselector, col_plotraw = st.columns([1, 3])

	with col_dataselector:
		ob_samples = GridOptionsBuilder.from_dataframe(samples)
		ob_samples.configure_selection(selection_mode='multiple', use_checkbox=True)
		ob_samples.configure_column('samples', suppressMenu=True, sortable=False)
		ag_samples = AgGrid(samples,
							ob_samples.build(),
							height=600,
							update_mode=GridUpdateMode.SELECTION_CHANGED,
							theme=AgGridTheme.ALPINE)
		selected_samples = [i['_selectedRowNodeInfo']['nodeRowIndex'] for i in ag_samples.selected_rows]

	with col_plotraw:
		# TODO add options for labels to sidebar
		fig_raw_data = px.line(df_data[df_data.columns[selected_samples]], color_discrete_sequence=px.colors.sequential.Blues)
		fig_raw_data.layout.template = 'plotly_dark'
		fig_raw_data.layout.legend.traceorder = 'normal'
		fig_raw_data.layout.margin = dict(l=20, r=20, t=20, b=20)
		fig_raw_data.layout.xaxis.title.text = 'Time (min)'
		fig_raw_data.layout.yaxis.title.text = 'Response'
		fig_raw_data.layout.legend.title.text = 'Sample'
		st.plotly_chart(fig_raw_data, use_container_width=True)


if __name__ == '__main__':
	main()