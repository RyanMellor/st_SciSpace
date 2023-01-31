import streamlit as st
from st_aggrid import AgGrid, GridUpdateMode, ColumnsAutoSizeMode, AgGridTheme
from st_aggrid.grid_options_builder import GridOptionsBuilder

import os
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
from io import BytesIO, StringIO
import urllib.request

import numpy as np
import pandas as pd
from math import log10, floor

from scipy import sparse
from scipy.signal import savgol_filter, find_peaks
from scipy.sparse.linalg import spsolve
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score

from collections import OrderedDict

from lmfit import Parameters, Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, PseudoVoigtModel

from helpers import setup
setup.setup_page("Raman Signal Processor")

data_test = "./assets/public_data/Raman Signal Processor - Test1.csv"
model_test = "./assets/public_data/Deconvolution - Model - Test1.txt"

PRIMARY_COLOR = "#4589ff"

def round_to_n(x, n):
	return round(x, -int(floor(log10(abs(x)))) + (n - 1))

class Spectrum:
	def __init__(self, x_data, y_data):
		self.x_data = x_data
		self.raw = y_data
		self.smooth = None
		self.baseline = None
		self.flat = None
		self.peaks = None
		self.flat_fit = None
		self.flat_fit_r2 = 0

	def idx_to_x(self, idx):
		if int(idx) == idx:
			return self.x_data[idx]
		else:
			x = [int(np.floor(idx)), int(np.ceil(idx))]
			y = self.x_data[x]
			return np.interp(idx, x, y)

	def do_smooth(self, window, order):
		# Savitzky–Golay
		self.smooth = savgol_filter(self.raw, window, order)

	def do_flat(self, lam, p, niter):
		self.baseline = baseline(self.smooth, lam, p, niter)
		self.flat = self.smooth - self.baseline

	def do_peaks(self, height, threshold, distance, prominence, width):
		# Find peaks in baseline subtracted spectra
		peaks, peak_props = find_peaks(
			self.flat,
			height=height,
			threshold=threshold,
			distance=distance,
			prominence=prominence,
			width=width)
		peak_props['peaks'] = peaks


		self.peaks = {}
		for i, peak in enumerate(peaks):
			self.peaks[peak] = {prop:val[i] for (prop,val) in peak_props.items()}
			
			for prop in ['peaks', 'left_bases','left_ips', 'right_bases', 'right_ips']:
				idx = peak_props[prop][i]
				self.peaks[peak][prop+'_x'] = self.idx_to_x(idx)

	def do_cs(self):
		for peak in self.peaks:
			x = self.x_data[peak-1:peak+2]
			y = self.flat[peak-1:peak+2]
			cs = CubicSpline(x, y)
			xs = np.arange(x[0], x[-1], 0.1)
			der_roots = cs.derivative().roots()
			der_roots = der_roots[x[0] < der_roots]
			der_roots = der_roots[der_roots < x[-1]]
			self.peaks[peak]['peak_xs'] = xs
			self.peaks[peak]['peak_ys'] = cs(xs)
			self.peaks[peak]['peak_max_x'] = der_roots[0]
			self.peaks[peak]['peak_max_y'] = cs(der_roots[0])

	def do_voigt(self):
		if len(self.peaks)!=0:

			n=10
			i = 0
			model = 0

			for peak in self.peaks:
				x = self.x_data[peak-n:peak+1+n]
				y = self.flat[peak-n:peak+1+n]
				peak_model = PseudoVoigtModel(prefix='p%d_' % (i+1))
				peak_params = peak_model.guess(y, x=x)
				if model != 0:
					model += peak_model
					params += peak_params
				else:
					model = peak_model 
					params = peak_params
				i += 1

			result = model.fit(self.flat, params, x=self.x_data)
			self.flat_fit = result.eval(x=self.x_data)
			self.flat_fit_r2 = r2_score(self.flat, self.flat_fit)

			x = self.x_data
			xs = np.arange(x[0], x[-1], 0.1)
			i = 0
			comps = result.eval_components(x=xs)
			for peak in self.peaks:
				comp = comps['p%d_'% (i+1)]
				x_start = self.x_data[peak-4*n]
				x_end = self.x_data[peak+1+4*n]
				peak_xs = []
				peak_ys = []
				for j in range(len(xs)):
					x_current = xs[j]
					if x_current >= x_start and x_current <= x_end:
						peak_xs.append(x_current)
						peak_ys.append(comp[j])

				self.peaks[peak]['peak_xs'] = peak_xs
				self.peaks[peak]['peak_ys'] = peak_ys
				self.peaks[peak]['peak_max_x'] = result.params['p%d_center'% (i+1)].value
				self.peaks[peak]['peak_max_y'] = result.params['p%d_height'% (i+1)].value

				i += 1

	def do_simple_voigt(self):
		n=3

		for peak in self.peaks:
			x = self.x_data[peak-n:peak+1+n]
			y = self.flat[peak-n:peak+1+n]

			model = PseudoVoigtModel()
			params = model.guess(y, x=x)
			result = model.fit(y, params, x=x)

			xs = np.arange(x[0], x[-1], 0.1)
			self.peaks[peak]['peak_xs'] = xs
			self.peaks[peak]['peak_ys'] = result.eval(x=xs)
			self.peaks[peak]['peak_max_x'] = result.params['center'].value
			self.peaks[peak]['peak_max_y'] = result.params['height'].value

def baseline(y, lam, p, niter):
	# Asymmetric Least Squares Smoothing
	L = len(y)
	D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
	D = lam * D.dot(D.transpose())
	w = np.ones(L)
	W = sparse.spdiags(w, 0, L, L)
	for i in range(niter):
		W.setdiag(w)
		Z = W + D
		z = spsolve(Z, w*y)
		w = p * (y > z) + (1-p) * (y < z)
	return z

def csv_header_rows(file, header_contains):
	rows = []
	# if isinstance(file, st.runtime.uploaded_file_manager.UploadedFile):
	# 	for i, line in enumerate(file.readlines()):
	# 		if header_contains in line.split(','):
	# 			rows.append(i)
	# else:
	with open(file) as f:
		for i, line in enumerate(f.readlines()):
			if header_contains in line.split(','):
				rows.append(i)
	return rows

def file_to_buffer(filepath):
    """
    Takes data from file under given path
    :param filepath: Str
    :return: StringIO object
    """
    with open(filepath) as f:
        content = f.read()
    buffer = StringIO(content)
    buffer.name = os.path.basename(filepath)
    return buffer

def main():

	st.markdown("<hr/>", unsafe_allow_html=True)
		
	st.markdown("### Data selection")

	with st.expander("Preprocess"):
		min_x = st.number_input("Min x", value=200)
		max_x = st.number_input("Max x", value=2000)
	with st.expander("Upload raw data and select signal to be processed", expanded=True):

		data_files = st.file_uploader(
			label='Upload raw data',
			label_visibility='collapsed',
			type=['csv'],
			accept_multiple_files=True)

		if not data_files:
			data_files = [data_test]

		samples = []
		all_spectra = OrderedDict()

		header_contains = 'Dark Subtracted #1'
		index_col = 'Raman Shift'
		extract_col = 'Dark Subtracted #1'

		for file in data_files:
			# if file.split('.')[-1].lower() == 'csv':
			if isinstance(file, st.runtime.uploaded_file_manager.UploadedFile):
				sample = file.name
			else:
				sample = '.'.join(file.split('.')[0:-1])
			samples.append(sample)

			header_row = 105#csv_header_rows(file, header_contains)[0]
			all_cols = pd.read_csv(file, header=header_row, index_col=index_col)
			all_cols = all_cols[all_cols.index > min_x]
			all_cols = all_cols[all_cols.index < max_x]

			all_spectra[sample] = Spectrum(x_data=all_cols.index, y_data=all_cols[extract_col])

		current_idx = 0
		current_sample = samples[current_idx]
		current_spectrum = all_spectra[current_sample]

	st.markdown("<hr/>", unsafe_allow_html=True)

	st.markdown("### Signal Processing")

	plot_layout = {
		"template": 'plotly_dark',
		"legend": {
			'traceorder':'normal',
			'yanchor': "top",
			'y': 0.99,
			'xanchor': "left",
			'x': 0.01},
		"margin": dict(l=20, r=20, t=20, b=20),
		"xaxis": {'title':{'text':'Raman Shift (cm-1)'}},
		"uirevision": "foo",
		"height":300
	}

	x = current_spectrum.x_data

	with st.expander("Smooth", expanded=True):
		col_smooth_plot, col_smooth_parameters = st.columns([3,1])
		with col_smooth_parameters:
			smooth_window = st.slider("Window", min_value=3, max_value=101, value=5, step=2)
			smooth_order = st.slider("Order", min_value=2, max_value=10, value=3, step=1)
			current_spectrum.do_smooth(smooth_window, smooth_order)
		with col_smooth_plot:
			fig_processed_data = go.Figure(layout=plot_layout)
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.raw, name="Raw", line_color="firebrick"))
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.smooth, name="Smooth", line_color="cornflowerblue"))
			st.plotly_chart(fig_processed_data, use_container_width=True)

	with st.expander("Baseline", expanded=True):
		col_baseline_plot, col_baseline_parameters = st.columns([3,1])
		with col_baseline_parameters:
			baseline_lam_options = [round_to_n(i,2) for i in list(np.logspace(0, 9, 100))]
			baseline_lam = st.select_slider("Smoothness (λ)", baseline_lam_options)
			baseline_p_options =   [round_to_n(i,2) for i in list(np.logspace(-5, 0, 100))]
			baseline_p = st.select_slider("Asymmetry (p)", baseline_p_options)
			baseline_niter = st.slider("Iterations", min_value=1, max_value=50, value=10, step=1)
			current_spectrum.do_flat(baseline_lam, baseline_p, baseline_niter)
		with col_baseline_plot:
			fig_processed_data = go.Figure(layout=plot_layout)
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.smooth, name="Smooth", line_color="cornflowerblue"))
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.baseline, name="Baseline", line_color="grey"))
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.flat, name="Flat", line_color="goldenrod"))
			st.plotly_chart(fig_processed_data, use_container_width=True)

	with st.expander("Peaks", expanded=True):
		colp_eaks_plot, colp_eaks_parameters = st.columns([3,1])
		with colp_eaks_parameters:
			peaks_height = st.slider('Height', min_value=0, max_value=100000, value=10000, step=1)
			peaks_threshold = st.slider('Threshold', min_value=0, max_value=10000, value=0, step=1)
			peaks_distance = st.slider('Distance', min_value=1, max_value=100, value=1, step=1)
			peaks_prominence = st.slider('Prominence', min_value=0, max_value=100, value=1, step=1)
			peaks_width = st.slider('Width', min_value=0, max_value=100, value=0, step=1)
			current_spectrum.do_peaks(peaks_height, peaks_threshold, peaks_distance, peaks_prominence, peaks_width)
			decon = st.checkbox("Deconvolution", False)
			if decon:
				current_spectrum.do_voigt()
			else:
				current_spectrum.do_simple_voigt()
				current_spectrum.flat_fit_r2 = 0
		with colp_eaks_plot:
			fig_processed_data = go.Figure(layout=plot_layout)
			fig_processed_data.add_trace(go.Scatter(x=x, y=current_spectrum.flat, name="Flat", line_color="goldenrod"))
			# peak_xs = []
			# peak_ys = []
			for i, peak in enumerate(current_spectrum.peaks.values()):
				showlegend = True if i==1 else False
				# peak_xs.append(peak['peaks_x'])
				# peak_ys.append(peak['peak_heights'])
				fig_processed_data.add_trace(go.Scatter(x=peak['peak_xs'], y=peak['peak_ys'], name='Peaks', line_color="firebrick", showlegend=showlegend, fill='tozeroy'))
			# fig_processed_data.add_trace(go.Scatter(x=peak_xs, y=peak_ys, name="Peaks", line_color="firebrick", mode="markers"))
			st.plotly_chart(fig_processed_data, use_container_width=True)


if __name__ == '__main__':
	main()