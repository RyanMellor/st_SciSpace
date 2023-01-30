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
from scipy.signal import savgol_filter
import requests
from io import BytesIO
import urllib.request

from helpers import setup
setup.setup_page("Deconvolution")

# data_test = r"http://sci-space.co.uk//test_data/Deconvolution%20-%20AuNCs.xlsx"
# model_test_url = r"http://sci-space.co.uk//test_data/Deconvolution%20-%20AuNCs.txt"
# model_test = urllib.request.urlopen(model_test_url)

data_test = "./assets/public_data/Deconvolution - Data - Test1.xlsx"
model_test = "./assets/public_data/Deconvolution - Model - Test1.txt"

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
PRIMARY_COLOR = "#4589ff"

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
		
	st.markdown("### Data selection")

	with st.expander("Upload data and select signal to be deconvoluted", expanded=True):

		data_file = st.file_uploader(
			label='Upload raw data',
			type=['txt', 'csv', 'xls', 'xlsx'])

		if not data_file:
			data_file = data_test

		ext = os.path.splitext(data_file)[-1][1:]

		# TODO add options for pd.read_XXX to sidebar
		if ext in ['xls', 'xlsx']:
			df_data = pd.read_excel(data_file, index_col=0)
		elif ext in ['csv', 'txt']:
			df_data = pd.read_csv(data_file, index_col=0)
		else:
			return None

		samples = pd.DataFrame(columns=['samples'], data=df_data.columns)
		col_dataselector, col_plotraw = st.columns([1, 3])

		with col_dataselector:
			ob_samples = GridOptionsBuilder.from_dataframe(samples)
			ob_samples.configure_selection(selection_mode='multiple', use_checkbox=True, pre_selected_rows=[0])
			ob_samples.configure_column('samples', suppressMenu=True, sortable=False)
			ag_samples = AgGrid(samples,
								ob_samples.build(),
								height=600,
								update_mode=GridUpdateMode.SELECTION_CHANGED,
								theme=AgGridTheme.ALPINE)
			selected_samples = [i['_selectedRowNodeInfo']['nodeRowIndex'] for i in ag_samples.selected_rows]

		with col_plotraw:
			# TODO add options for labels to sidebar
			fig_raw_data = px.line(df_data[df_data.columns[selected_samples]])
			fig_raw_data.layout.template = 'plotly_dark'
			fig_raw_data.layout.legend.traceorder = 'normal'
			fig_raw_data.layout.margin = dict(l=20, r=20, t=20, b=20)
			fig_raw_data.layout.xaxis.title.text = 'Wavelength (nm)'
			fig_raw_data.layout.yaxis.title.text = 'Absorbance'
			fig_raw_data.layout.legend.title.text = 'Sample'
			st.plotly_chart(fig_raw_data, use_container_width=True)

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Deconvolution setup")

	with st.expander("Upload deconvolution model", expanded=True):

		model_file = st.file_uploader(
			label='Upload deconvolution model',
			type=['txt'])
		if not model_file:
			model_file = model_test
		with open(model_file, 'r') as f:
			deconvolution_setup = json.load(f)

		# deconvolution_setup = [
		# 	{
		# 		'feature': '',
		# 		'model': '',
		# 		'color': '',
		# 		'parameters': [
		# 			{'parameter': 'amplitude', 'value': None, 'min': None, 'max': None, 'vary': None},
		# 			{'parameter': 'center', 'value': None, 'min': None, 'max': None, 'vary': None},
		# 			{'parameter': 'sigma', 'value': None, 'min': None, 'max': None, 'vary': None},
		# 		]
		# 	}
		# ]
		# with open(r'test_data\Deconvolution - AuNPs - Interacting.txt', 'w') as f:
		# 	json.dump(deconvolution_setup, f, indent=4)

		df_deconvolution_setup = pd.DataFrame()
		for f in deconvolution_setup:
			data = {}
			data['feature'] = f['feature']
			data['model'] = f['model']
			data['color'] = f['color']
			for p in f['parameters']:
				temp_data = data | p
				df = pd.DataFrame([temp_data], columns=temp_data.keys())
				df_deconvolution_setup = pd.concat([df_deconvolution_setup, df])
		df_deconvolution_setup.reset_index(drop=True, inplace=True)

		col_feature_setup, col_feature_parameters = st.columns([1,2])
		with col_feature_setup:
			st.markdown("### Feature setup")
			df_features = df_deconvolution_setup[['feature', 'model', 'color']].drop_duplicates()
			ob_features = GridOptionsBuilder.from_dataframe(df_features)
			ob_features.configure_column('feature', suppressMenu=True, sortable=False, editable=True)
			ob_features.configure_column('model', suppressMenu=True, sortable=False, editable=True)
			ob_features.configure_column('color', suppressMenu=True, sortable=False, editable=True)
			ag_features = AgGrid(
				df_features,
				ob_features.build(),
				columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
				theme=AgGridTheme.ALPINE,
			)
		with col_feature_parameters:
			st.markdown("### Feature parameters")
			ob_parameters = GridOptionsBuilder.from_dataframe(df_deconvolution_setup)
			ob_parameters.configure_column('feature', suppressMenu=True, sortable=False)
			ob_parameters.configure_column('model', hide=True)
			ob_parameters.configure_column('color', hide=True)
			ob_parameters.configure_column('parameter', suppressMenu=True, sortable=False)
			ob_parameters.configure_column('value', suppressMenu=True, sortable=False, editable=True, type=["numericColumn","numberColumnFilter"])
			ob_parameters.configure_column('min', suppressMenu=True, sortable=False, editable=True, type=["numericColumn","numberColumnFilter"])
			ob_parameters.configure_column('max', suppressMenu=True, sortable=False, editable=True, type=["numericColumn","numberColumnFilter"])
			ob_parameters.configure_column('vary', suppressMenu=True, sortable=False, editable=True)
			ag_parameters = AgGrid(
				df_deconvolution_setup,
				ob_parameters.build(),
				columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
				theme=AgGridTheme.ALPINE,
			)

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Output")

	with st.expander("Deconvolution result", expanded=True):

		col_deconvolution_plot, col_deconvloution_values = st.columns([2,1])
		with col_deconvolution_plot:
			composite_model = None
			params = None
			components = ag_parameters.data.groupby('feature')
			for component in components:
				name = component[0]
				df = component[1]
				prefix = name + '_'
				model = getattr(models, df.iloc[0]['model'])(prefix=prefix)
				for i, row in df.iterrows():
					par = row['parameter']
					for var in ['value', 'min', 'max', 'vary']:
						try:
							val = row[var]
							if not pd.isnull(val):
								model.set_param_hint(par, **{var: val})
						except:
							pass
				model_params = model.make_params()
				if params is None:
					params = model_params
				else:
					params.update(model_params)
				if composite_model is None:
					composite_model = model
				else:
					composite_model = composite_model + model

			#TODO add options for data_range to sidebar
			data_range = [400,1000]
			df_data = df_data[df_data.index >= data_range[0]]
			df_data = df_data[df_data.index <= data_range[1]]
			x_data = np.array(df_data.index)
			try:
				y_data = np.array(df_data[df_data.columns[selected_samples[0]]])
			except:
				return None

			#TODO add options for savgol_filter to sidebar
			perform_filter = True
			if perform_filter:
				y_data = savgol_filter(y_data, 11, 2)

			output = composite_model.fit(y_data, params, x=x_data)
			eval_components = output.eval_components(x=x_data)
			fit = output.best_fit
			res = fit - y_data
			r2 = r2_score(y_data, fit)

			fig_deconvolution = go.Figure()
			fig_deconvolution.add_trace(go.Scatter(x=x_data, y=y_data, name='Data', line_color='silver'))
			fig_deconvolution.add_trace(go.Scatter(x=x_data, y=fit, name=f'Fit: R2={r2:.4f}', line_color='gold'))
			for name, y_data in eval_components.items():
				name = name[:-1]
				df = df_deconvolution_setup[df_deconvolution_setup['feature']==name]
				color = df.iloc[0]['color']
				fig_deconvolution.add_trace(go.Scatter(x=x_data, y=y_data, name=name, line_color=color))

			fig_deconvolution.layout.template = 'plotly_dark'
			fig_deconvolution.layout.legend.traceorder = 'normal'
			fig_deconvolution.layout.margin = dict(l=20, r=20, t=20, b=20)
			fig_deconvolution.layout.xaxis.title.text = 'Wavelength (nm)'
			fig_deconvolution.layout.yaxis.title.text = 'Absorbance'
			st.plotly_chart(fig_deconvolution, use_container_width=True)

		with col_deconvloution_values:
			st.markdown("### Best fit values")
			st.write("")
			for key, val in output.best_values.items():
				st.write(key, round(val,2))

	st.markdown("<hr/>", unsafe_allow_html=True)


if __name__ == '__main__':
	main()