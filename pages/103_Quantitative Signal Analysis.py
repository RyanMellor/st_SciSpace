# ---- Standard imports ----
import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, AgGridTheme
from st_aggrid.grid_options_builder import GridOptionsBuilder
import os
import pandas as pd
from pprint import pprint
import plotly.graph_objects as go
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from math import log10, floor
from scipy.signal import savgol_filter
from scipy.integrate import trapz
from collections import OrderedDict

# ---- Custom imports ----
from helpers import sci_setup, sci_data
sci_setup.setup_page("Quantitative Signal Analysis")

data_test = "./assets/public_data/Quantitative Signal Analyser - Data - Test1.xlsx"
model_test = "./assets/public_data/Quantitative Signal Analyser - Model - Test1.xlsx"

PRIMARY_COLOR = "#4589ff"

plot_layout = {
	"template": 'plotly_dark',
	"legend": {
		'traceorder':'normal',
		'yanchor': "top",
		'y': 0.99,
		'xanchor': "left",
		'x': 0.01},
	"margin": dict(l=20, r=20, t=20, b=20),
	"xaxis": {},
	"yaxis": {},
	# "uirevision": "foo",
	"height":300
}
def round_to_n(x, n):
	return round(x, -int(floor(log10(abs(x)))) + (n - 1))

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
		
	st.markdown("### Data selection")

	with st.expander("Setup"):
		col_x_axis, col_y_axis, col_oth = st.columns(3)
		with col_x_axis:
			st.markdown("### X axis")
			x_title = st.text_input("X title", "Chemical shift (ppm)")
			x_min = st.number_input("X min", value=0.0)
			x_max = st.number_input("X max", value=6.0)
			x_reversed = st.checkbox('x reversed', True)
			plot_layout['xaxis']['autorange'] = 'reversed' if x_reversed  else True
			plot_layout['xaxis']['title'] = x_title
		with col_y_axis:
			st.markdown("### Y axis")
			y_title = st.text_input("Y title", "Intensity")
			y_min = st.number_input("Y min", value=0.0)
			y_max = st.number_input("Y max", value=2_000_000)
			plot_layout['yaxis']['title'] = y_title

	with st.expander("Upload raw data and select signal to be processed", expanded=True):

		data_files = st.file_uploader(
			label='Upload raw data',
			label_visibility='collapsed',
			type=['xls', 'xlsx', 'csv', 'txt'],
			accept_multiple_files=True)

		st.caption("Currently expects file to have two columns with headers, one for x_data and one for y_data.")

		if not data_files:
			data_files = [data_test]

		samples = []
		all_spectra = OrderedDict()

		for file in data_files:
			if isinstance(file, st.runtime.uploaded_file_manager.UploadedFile):
				sample = file.name
			else:
				sample = '.'.join(file.split('.')[0:-1])
			samples.append(sample)

			data_df = sci_data.file_to_df(file)
			data_df.columns = ['x_data', 'y_data']
			data_df = data_df[data_df['x_data'] > x_min]
			data_df = data_df[data_df['x_data'] <= x_max]
			data_df = data_df[data_df['y_data'] > y_min]
			data_df = data_df[data_df['y_data'] <= y_max]
			all_spectra[sample] = data_df

		current_idx = 0
		current_sample = samples[current_idx]
		current_spectrum = all_spectra[current_sample]

		x = current_spectrum['x_data']

		fig_raw_data = go.Figure(layout=plot_layout)
		fig_raw_data.add_trace(go.Scatter(x=x, y=current_spectrum['y_data'], name="Raw"))
		st.plotly_chart(fig_raw_data, use_container_width=True)


	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Integration Regions")
	with st.expander("Define regions to integrate", expanded=True):
		model_file = st.file_uploader(
			label='Upload model file',
			label_visibility='collapsed',
			type=['xls', 'xlsx', 'csv', 'txt'])
		if not model_file:
			model_file = model_test

		feature_df = sci_data.file_to_df(model_file)

		df_features = feature_df
		ob_features = GridOptionsBuilder.from_dataframe(df_features)
		ob_features.configure_selection(use_checkbox=True, pre_selected_rows=[0])
		ob_features.configure_column('feature', suppressMenu=True, sortable=False, editable=True)
		ob_features.configure_column('from', suppressMenu=True, sortable=False, editable=True)
		ob_features.configure_column('to', suppressMenu=True, sortable=False, editable=True)
		ob_features.configure_column('weighting', suppressMenu=True, sortable=False, editable=True)
		ag_features = AgGrid(
			df_features,
			ob_features.build(),
			columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
			theme=AgGridTheme.ALPINE,
		)
		st.caption("Use the checkbox to identify which feature to normalize against.")
		ag_features.data.sort_values(
			by='to' if x_reversed else 'from',
			ascending=False if x_reversed else True,
			inplace=True)
		features = ag_features.data.to_dict('records')
		try:
			norm_feature = ag_features.selected_rows[0]['feature']
		except:
			st.warning("Please select a feature to normalize against")
			return None

		st.markdown("<hr/>", unsafe_allow_html=True)

		col_sg_window, col_sg_order = st.columns(2)
		with col_sg_window:
			sg_window = st.number_input("Savgol filter window", 3, 100, 20, 1)
		with col_sg_order:
			sg_order = st.number_input("Savgol filter order", 2, 10, 3, 1)
		st.caption("Savgol parameters are used for smoothing of the derivative, they do not affect integration.")
		tabs = st.tabs([f['feature'] for f in features])
		for i, tab in enumerate(tabs):
			f = features[i]
			w = f['to'] - f['from']
			mask = (current_spectrum['x_data'] > f['from'] - w/10) & (current_spectrum['x_data'] <= f['to'] + w/10)
			feature_data = current_spectrum[mask]

			highlight = dict(x0=f['from'],
							x1=f['to'],
							fillcolor="white",
							opacity=0.1,
							line_width=0)
			
			with tab:
				st.markdown("### Raw data")
				fig_raw_data = go.Figure(layout=plot_layout)
				fig_raw_data.add_trace(go.Scatter(x=feature_data['x_data'], y=feature_data['y_data'], name="Raw"))
				fig_raw_data.add_vrect(**highlight)
				st.plotly_chart(fig_raw_data, use_container_width=True)
				
				st.markdown("### First derivative")
				
				try:
					feature_data['deriv'] = savgol_filter(feature_data['y_data'], sg_window, sg_order, 1)
				except:
					st.warning("Savgol filter order must be less than Savgol filter window.")
					return None
				fig_deriv = go.Figure(layout=plot_layout)
				fig_deriv.add_trace(go.Scatter(x=feature_data['x_data'], y=feature_data['deriv'], name="Deriv"))
				fig_deriv.add_vrect(**highlight)
				st.plotly_chart(fig_deriv, use_container_width=True)

			f['integration'] = trapz(feature_data['y_data'], feature_data['x_data'])
			if x_reversed:
				f['integration'] = -f['integration']
			f['weighted'] = f['integration'] / f['weighting']
			if f['feature'] == norm_feature:
				norm_int_weight = f['weighted']
		for f in features:
			f['normalized'] = f['weighted']/norm_int_weight

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Output")

	with st.expander("Integrations", expanded=True):
		fig_output = go.Figure(layout=plot_layout)
		fig_output.add_trace(go.Scatter(x=x, y=current_spectrum['y_data'], name="Raw"))
		for f in features:
			highlight = dict(x0=f['from'],
							x1=f['to'],
							fillcolor="white",
							opacity=0.1,
							line_width=0,
							annotation_text=f['feature'],
							annotation_position="top"
							)
			fig_output.add_vrect(**highlight)
		st.plotly_chart(fig_output, use_container_width=True)
		st.dataframe(pd.DataFrame().from_dict(features)[['feature', 'normalized']])

if __name__ == '__main__':
	main()