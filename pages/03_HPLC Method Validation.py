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
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.integrate import trapz
import requests
from io import BytesIO
import urllib.request
import math

data_test = r"http://sci-space.co.uk//test_data/HPLC%20Method%20Validation%20-%20Test%20data.xlsx"

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

logo_url = r"http://sci-space.co.uk//scispace.png"
logo_response = requests.get(logo_url)
logo = Image.open(BytesIO(logo_response.content))

st.sidebar.image(logo)

page_setup = """
	<div>
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

	st.markdown(r"""
		### How to use this page
		1. Download the raw data template provided in the sidebar.
		2. Populate the spreadsheet with your raw data.
			- Lin (linearity): triplicate preparations at 25, 50, 75, 100, 150, and 200% of the target value.
			- Acc (accuracy): triplicate preparations at 50, 100, and 100% of the target value.
			- Rep (repeatability): 10 injections of the same sample at 100% of the target value.
		3. Upload the populated document to the next section.
		4. Select whether or not to subtract baseline.
		5. Define analyte integration ranges.

	""", unsafe_allow_html=True)

	st.markdown("<hr/>", unsafe_allow_html=True)

	data_file = st.file_uploader(
		label='Upload raw data',
		type=['txt', 'csv', 'xls', 'xlsx'])
	if not data_file:
		data_file = data_test

	df_data_read = read_data(data_file)
	if df_data_read is None:
		return None

	sample_names = [i for i in df_data_read.columns if i != "Baseline"]

	subtract_baseline = st.checkbox('Subtract baseline')
	if subtract_baseline:
		st.markdown("Doing good science :thumbsup:")
		# print([i for i in df_data.columns if i != "Baseline"])
		df_data = df_data_read[sample_names].sub(df_data_read["Baseline"], axis=0)
	else:
		df_data = df_data_read

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

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("## Analytes")
	analytes = [
		{
		'analyte': 'Analyte1',
		'from': 4,
		'to': 4.4,
		'target concentration': 200,
		},
		{
		'analyte': 'Analyte2',
		'from': 5.1,
		'to': 5.4,
		'target concentration': 200,
		},
	]
	df_analytes = pd.DataFrame(analytes)
	ob_analytes = GridOptionsBuilder.from_dataframe(df_analytes)
	ob_analytes.configure_column('analyte', suppressMenu=True, sortable=False, editable=True)
	ob_analytes.configure_column('from', suppressMenu=True, sortable=False, editable=True)
	ob_analytes.configure_column('to', suppressMenu=True, sortable=False, editable=True)
	ob_analytes.configure_column('target concentration', suppressMenu=True, sortable=False, editable=True)
	ag_analytes = AgGrid(
		df_analytes,
		ob_analytes.build(),
		columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
		theme=AgGridTheme.ALPINE,
	)
	analytes = ag_analytes.data.to_dict('records')

	study_abv = {
		'Lin': 'Linearity',
		'Acc': 'Accuracy',
		'Rep': 'Repeatability',
	}

	df_calcs = pd.DataFrame(columns=['sample', 'study', 'level', 'rep'])
	for sample in sample_names:
		split = sample.split('_')
		df = pd.DataFrame([{
			'sample': sample,
			'study': study_abv[split[0]],
			'level': int(split[1]),
			'rep': int(float(split[2])),
		}])
		df_calcs = pd.concat([df_calcs, df], ignore_index=True)

	for a in analytes:
		analyte = a['analyte']
		df_calcs[f'Nominal_{analyte}'] = pd.Series(dtype='float')
		df_calcs[f'AUC_{analyte}'] = pd.Series(dtype='float')
		df_calcs[f'Calc_{analyte}'] = pd.Series(dtype='float')
		df_calcs[f'Recovery_{analyte}'] = pd.Series(dtype='float')

		mask = (df_data.index > a['from']) & (df_data.index <= a['to'])
		df = df_data.loc[mask]
		for s in sample_names:
			auc = trapz(df[s], df.index)
			df_calcs[f'AUC_{analyte}'][(df_calcs['sample']==s)] = auc
			# df_calcs[f'Nominal_{analyte}'][(df_calcs['sample']==s)] = a['target concentration'] * df_calcs['level'][(df_calcs['sample']==s)]/100 

		df_calcs[f'AUC_{analyte}'] = df_calcs[f'AUC_{analyte}'].astype(float)

		df_calcs[f'Nominal_{analyte}'] = a['target concentration'] * df_calcs['level']/100
		df_calcs[f'Nominal_{analyte}'] = df_calcs[f'Nominal_{analyte}'].astype(float)

	st.markdown("<hr/>", unsafe_allow_html=True)

	log_x = False
	log_y = False

	for a in analytes:
		analyte = a['analyte']
		st.markdown(f"### {analyte}", unsafe_allow_html=True)

		col_lin, col_acc, col_rep= st.columns([1,1,1])

		with col_lin:
			df_lin = df_calcs[df_calcs['study']=='Linearity']
			df_lin_desc = df_lin.groupby(f'Nominal_{analyte}')[f'AUC_{analyte}'].describe()

			fig_lin = px.scatter(
				x=df_lin_desc.index,
				y=df_lin_desc['mean'],
				error_y=df_lin_desc['std'],
				trendline="ols",
				trendline_options=dict(log_x=log_x, log_y=log_y)
			)
			fig_lin.layout.template = 'plotly_dark'
			fig_lin.layout.legend.traceorder = 'normal'
			fig_lin.layout.margin = dict(l=10, r=10, t=30, b=30)
			fig_lin.layout.title = 'Linearity'
			fig_lin.layout.xaxis.title.text = 'Concentration'
			fig_lin.layout.yaxis.title.text = 'AUC'
			st.plotly_chart(fig_lin, use_container_width=True)

			x = np.array(df_lin[f'Nominal_{analyte}'])
			y = np.array(df_lin[f'AUC_{analyte}'])

			fit = np.polyfit(x, y, deg=1)
			n = len(x)
			slope = fit[0]
			intercept = fit[1]
			r = np.corrcoef(x, y)[0,1]
			r2 = r**2
			y_pred = slope * x + intercept
			steyx = (((y-y_pred)**2).sum()/(n-2))**0.5
			loq = 10 * (steyx / slope)
			lod = 3.3 * (steyx / slope)

			df_calcs[f'Calc_{analyte}'] = (df_calcs[f'AUC_{analyte}'] - intercept) / slope
			df_calcs[f'Recovery_{analyte}'] = 100 * (df_calcs[f'Calc_{analyte}'] / df_calcs[f'Nominal_{analyte}'])
			df_calcs[f'Calc_{analyte}'] = df_calcs[f'Calc_{analyte}'].astype(float)
			df_calcs[f'Recovery_{analyte}'] = df_calcs[f'Recovery_{analyte}'].astype(float)

		with col_acc:
			df_acc = df_calcs[df_calcs['study']=='Accuracy']
			df_acc_desc = df_acc.groupby('level')[f'Recovery_{analyte}'].describe()
			fig_acc = px.bar(
				x=df_acc_desc.index,
				y=df_acc_desc['mean'],
				error_y=df_acc_desc['std']
			)
			fig_acc.layout.template = 'plotly_dark'
			fig_acc.layout.legend.traceorder = 'normal'
			fig_acc.layout.margin = dict(l=10, r=10, t=30, b=30)
			fig_acc.layout.title = 'Accuracy'
			fig_acc.layout.xaxis.title.text = 'Level (%of target)'
			fig_acc.layout.yaxis.title.text = 'Recovery (%)'
			fig_acc.update_xaxes(type='category')
			st.plotly_chart(fig_acc, use_container_width=True)
	
		with col_rep:
			df_rep = df_calcs[df_calcs['study']=='Repeatability']
			df_rep_desc = df_rep.groupby('level')[f'Recovery_{analyte}'].describe()
			# print(df_rep_desc)
			fig_rep = px.bar(
				x=df_rep_desc.index,
				y=df_rep_desc['mean'],
				error_y=df_rep_desc['std']
			)
			fig_rep.layout.template = 'plotly_dark'
			fig_rep.layout.legend.traceorder = 'normal'
			fig_rep.layout.margin = dict(l=10, r=10, t=30, b=30)
			fig_rep.layout.title = 'Repeatability'
			fig_rep.layout.xaxis.title.text = 'Level (%of target)'
			fig_rep.layout.yaxis.title.text = 'Recovery (%)'
			fig_rep.update_xaxes(type='category')
			st.plotly_chart(fig_rep, use_container_width=True)
		
		st.dataframe(pd.DataFrame([{
			'Slope': slope,
			'Intercept': intercept,
			'R²': r2,
			'STEYX': steyx,
			'LOQ': loq,
			'LOD': lod,
		}]))
		# st.markdown(f'''
		# 	Slope = {round(slope,2)} | 
		# 	Intercept = {round(intercept,2)} | 
		# 	R² = {round(r2,4)} | 
		# 	STEYX = {round(steyx,2)} | 
		# 	LOQ = {round(loq,2)} | 
		# 	LOD = {round(lod,2)}
		# ''')
		mask = (df_data.index > a['from']) & (df_data.index <= a['to'])
		df_peak = df_data.loc[mask]
		peaks, properties = find_peaks(df_peak['Lin_100_01'], height=5, prominence=10, width=5)

		peak_idx = peaks
		peak_x = df_peak.index[peak_idx]
		peak_y = df_peak['Lin_100_01'][peak_x]

		left_ip_idx = math.floor(properties['left_ips'])
		left_ip_x = df_peak.index[left_ip_idx]
		left_ip_y = df_peak['Lin_100_01'][left_ip_x]

		right_ip_idx = math.ceil(properties['right_ips'])
		right_ip_x = df_peak.index[right_ip_idx]
		right_ip_y = df_peak['Lin_100_01'][right_ip_x]


		fig_peak = px.area(df_peak['Lin_100_01'])
		fig_peak.add_trace(go.Scatter(x=peak_x, y=peak_y))
		fig_peak.add_trace(go.Scatter(x=[left_ip_x], y=[left_ip_y]))
		fig_peak.add_trace(go.Scatter(x=[right_ip_x], y=[right_ip_y]))
		fig_peak.layout.template = 'plotly_dark'
		fig_peak.layout.legend.traceorder = 'normal'
		fig_peak.layout.margin = dict(l=20, r=20, t=20, b=20)
		fig_peak.layout.xaxis.title.text = 'Time (min)'
		fig_peak.layout.yaxis.title.text = 'Response'
		fig_peak.layout.legend.title.text = 'Sample'
		st.plotly_chart(fig_peak, use_container_width=True)


		st.markdown("<hr/>", unsafe_allow_html=True)

	with st.expander('Calculated values'):
		st.dataframe(df_calcs)

	st.markdown("<hr/>", unsafe_allow_html=True)

if __name__ == '__main__':
	main()