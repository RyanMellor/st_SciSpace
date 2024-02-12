import streamlit as st
from streamlit_drawable_canvas import st_canvas
# from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, AgGridTheme
# from st_aggrid.grid_options_builder import GridOptionsBuilder
# from st_aggrid.shared import JsCode

import json
import numpy as np
import pandas as pd
from pprint import pprint
import plotly.graph_objects as go
import plotly.express as px
from lmfit import models
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

from helpers import sci_setup, sci_data, sci_utils
sci_setup.setup_page("Deconvolution")

data_test = "./assets/public_data/Deconvolution - Data - Test1.xlsx"
model_test = "./assets/public_data/Deconvolution - Model - Test1.txt"

PRIMARY_COLOR = "#4589ff"

def sample_selection(samples:list):
	selected = [False] * len(samples)
	selected[0] = True
	df = pd.DataFrame({'✔': selected, 'Samples': samples})
	edited_df = st.data_editor(
		df,
		disabled=['Samples'],
		hide_index=True,
		use_container_width=True)
	return edited_df[edited_df['✔']]['Samples'].tolist()

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
		
	st.markdown("### Data selection")

	with st.expander("Upload raw data and select signal to be deconvoluted", expanded=True):

		data_file = st.file_uploader(
			label='Upload raw data',
			label_visibility='collapsed',
			type=['txt', 'csv', 'xls', 'xlsx'])

		if not data_file:
			data_file = data_test

		df_data = sci_data.file_to_df(data_file, index_col=0)

		col_dataselector, col_plotraw = st.columns([1, 3])

		with col_dataselector:
			tab_selected_samples, tab_settings = st.tabs(['Select samples', 'Settings'])
			with tab_selected_samples:
				selected_samples = sample_selection(df_data.columns.tolist())
			with tab_settings:

				# setup axis labels
				x_axis_title = st.text_input('X axis title', value='Wavelength')
				x_axis_units = st.text_input('X axis units', value='nm')
				y_axis_title = st.text_input('Y axis title', value='Absorbance')
				y_axis_units = st.text_input('Y axis units', value='')
				x_axis_label = f'{x_axis_title}' if x_axis_units == "" else f'{x_axis_title} ({x_axis_units})'
				y_axis_label = f'{y_axis_title}' if y_axis_units == "" else f'{y_axis_title} ({y_axis_units})'

				# apply data range
				min_x = int(df_data.index.min())
				max_x = int(df_data.index.max())
				data_range = st.slider('Data range', min_value=min_x, max_value=max_x, value=(min_x, max_x))
				df_data = df_data[df_data.index >= data_range[0]]
				df_data = df_data[df_data.index <= data_range[1]]
				x_data = np.array(df_data.index)

				# apply savgol filter
				sg_filter = st.checkbox('Apply Savitzky-Golay filter')
				if sg_filter:
					sg_filter_window = st.number_input('Savitzky-Golay filter window', min_value=1, max_value=100, value=11)
					sg_filter_order = st.number_input('Savitzky-Golay filter order', min_value=1, max_value=10, value=2)
					for s in df_data.columns:
						try:
							df_data[s] = savgol_filter(df_data[s], sg_filter_window, sg_filter_order)
						except Exception as e:
							st.error(e)
							return None

		with col_plotraw:
			fig = go.Figure()
			for s in selected_samples:
				fig.add_trace(go.Scatter(x=df_data.index, y=df_data[s], name=s))	
			fig.update_layout(
				template='plotly_dark',
		    	legend=dict(traceorder='normal'),
				margin=dict(l=20, r=20, t=20, b=20),
				xaxis_title=x_axis_label,
				yaxis_title=y_axis_label,
				legend_title='Sample'
				)
			st.plotly_chart(fig, use_container_width=True)

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Deconvolution setup")

	with st.expander("Upload deconvolution model", expanded=True):

		model_file = st.file_uploader(
			label='Upload deconvolution model',
			label_visibility='collapsed',
			type=['txt'])
		if not model_file:
			model_file = model_test
		with open(model_file, 'r') as f:
			deconvolution_setup = json.load(f)
		df_deconvolution_setup = pd.DataFrame()
		for i, f in enumerate(deconvolution_setup):
			data = {}
			data['id'] = i + 1
			data['feature'] = f['feature']
			data['model'] = f['model']
			data['color'] = f['color']
			for p in f['parameters']:
				temp_data = data | p
				df = pd.DataFrame([temp_data], columns=temp_data.keys())
				df_deconvolution_setup = pd.concat([df_deconvolution_setup, df])
		df_deconvolution_setup.reset_index(drop=True, inplace=True)

		from lmfit.models import VoigtModel, GaussianModel, LorentzianModel, PseudoVoigtModel, ExponentialModel, LinearModel
		built_in_models = {
			'VoigtModel': VoigtModel,
			'GaussianModel': GaussianModel,
			'LorentzianModel': LorentzianModel,
			'PseudoVoigtModel': PseudoVoigtModel,
			# 'ExponentialModel': ExponentialModel,
			# 'LinearModel': LinearModel
		}

		st.markdown("### Feature setup")
		df_features = df_deconvolution_setup[['id', 'feature', 'model', 'color']].drop_duplicates()
		edited_df_features = st.data_editor(
			df_features,
			disabled=['id'],
			hide_index=True,
			num_rows="dynamic",
			column_config={
				'feature': st.column_config.TextColumn(),
				'model': st.column_config.SelectboxColumn(
					options=built_in_models.keys(),
				),
				'color': st.column_config.TextColumn(
					help='Color of the feature in the plot. Use hex code or color name.',
				),
			}
		)

		# for f in edited_df_features.iterrows():
		# 	# list the parameters of the model
		# 	temp_model = built_in_models[f[1]['model']]()
		# 	temp_parameters = temp_model.param_names
		# 	st.markdown('---', unsafe_allow_html=True)
		# 	st.markdown(f"Feature: {f[1]['feature']}")
		# 	grid = sci_utils.make_grid(len(temp_parameters)+1, 5)
		# 	grid[0][1].markdown(f"Value")
		# 	grid[0][2].markdown(f"Min")
		# 	grid[0][3].markdown(f"Max")
		# 	grid[0][4].markdown(f"Vary")
		# 	for i, p in enumerate(temp_parameters):
		# 		grid[i+1][0].markdown(f"{p}")
		# 		temp_value = grid[i+1][1].number_input(f"Value", value=0.0, key=f"{f[1]['id']}_{p}_value", label_visibility="collapsed")
		# 		temp_min = grid[i+1][2].number_input(f"Min", value=0.0, key=f"{f[1]['id']}_{p}_min", label_visibility="collapsed")
		# 		temp_max = grid[i+1][3].number_input(f"Max", value=0.0, key=f"{f[1]['id']}_{p}_max", label_visibility="collapsed")
		# 		temp_vary = grid[i+1][4].checkbox(f"Vary", value=True, key=f"{f[1]['id']}_{p}_vary", label_visibility="collapsed")
				
		for f in edited_df_features.iterrows():
			df_deconvolution_setup.loc[df_deconvolution_setup['id'] == f[1]['id'], 'feature'] = f[1]['feature']
			df_deconvolution_setup.loc[df_deconvolution_setup['id'] == f[1]['id'], 'model'] = f[1]['model']
			df_deconvolution_setup.loc[df_deconvolution_setup['id'] == f[1]['id'], 'color'] = f[1]['color']

		show_colorpicker = st.checkbox("Show color picker")
		if show_colorpicker:
			col_colorpicker, col_nearestnamedcolor = st.columns([1, 4])
			with col_colorpicker:
				color_hex = st.color_picker("Color picker", "#6495ed")
			with col_nearestnamedcolor:
				nearest_named_color = sci_utils.nearest_named_color(color_hex)
				st.markdown(f"""
				Selected color: {color_hex}  
				Nearest named color: {nearest_named_color[0]}  
				Hex code: {nearest_named_color[1]}  
				<span style='background-color:{nearest_named_color[1]};color:{nearest_named_color[1]}'>color</span>
				""", unsafe_allow_html=True)

		st.markdown("### Feature parameters")
		edited_df_parameters = st.data_editor(
			df_deconvolution_setup,
			disabled=['feature', 'model', 'color', 'parameter', ],
			hide_index=True,
			use_container_width=True)

		st.markdown("### Model preview")
		composite_model = None
		params = None
		components = edited_df_parameters.groupby('feature')
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

		try:
			y_data = np.array(df_data[selected_samples[0]])
		except:
			return None
		
		# evaluate the model with the initial parameters
		initial_model = composite_model.eval(params, x=x_data)
		# initial_params = composite_model.guess(y_data, x=x_data)
		initial_comps = composite_model.eval_components(x=x_data)
		r2 = r2_score(y_data, initial_model)

		fig_model_preview = go.Figure()
		fig_model_preview.add_trace(go.Scatter(x=x_data, y=y_data, name='Data', line_color='silver'))
		fig_model_preview.add_trace(go.Scatter(x=x_data, y=initial_model, name=f'Initial: R2={r2:.4f}', line_color='gold'))
		for m in composite_model.components:
			name = m.prefix[:-1]
			df = df_deconvolution_setup[df_deconvolution_setup['feature']==name]
			color = df.iloc[0]['color']
			fig_model_preview.add_trace(go.Scatter(
				x=x_data, y=m.eval(params, x=x_data), name=name, line_color=color))
		# for name, comp_y_data in initial_comps.items():
		# 	name = name[:-1]
		# 	df = df_deconvolution_setup[df_deconvolution_setup['feature']==name]
		# 	color = df.iloc[0]['color']
		# 	fig_model_preview.add_trace(go.Scatter(x=x_data, y=comp_y_data, name=name, line_color=color))
		st.plotly_chart(fig_model_preview, use_container_width=True)
		
	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Output")

	with st.expander("Deconvolution result", expanded=True):

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
		fig_deconvolution.update_layout(
			template='plotly_dark',
			legend=dict(traceorder='normal'),
			margin=dict(l=20, r=20, t=20, b=20),
			xaxis_title=x_axis_label,
			yaxis_title=y_axis_label,
		)
		st.plotly_chart(fig_deconvolution, use_container_width=True)

	with st.expander("Best fit values"):
		st.write("")
		best_fit_values_df = pd.DataFrame().from_dict(output.best_values, orient='index')
		best_fit_values_df.columns=['Value']
		st.dataframe(best_fit_values_df)

	st.markdown("<hr/>", unsafe_allow_html=True)


if __name__ == '__main__':
	main()