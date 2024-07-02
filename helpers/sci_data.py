import streamlit as st
import pandas as pd
import os

@st.cache_data()
def file_to_df(data_file, delimiter=",", index_col=None, header=None):
	ext = ""
	if isinstance(data_file, st.runtime.uploaded_file_manager.UploadedFile):
		ext = os.path.splitext(data_file.name)[-1][1:]
	else:
		ext = os.path.splitext(data_file)[-1][1:]

	if ext in ['xls', 'xlsx']:
		if header is None:
			header = 0
		return pd.read_excel(data_file, index_col=index_col, header=header)
	elif ext in ['csv', 'txt']:
		if header is None:
			header = 'infer'
		return pd.read_csv(data_file, index_col=index_col, header=header, delimiter=delimiter)
	else:
		print("Unsupported file format")
		return None
	
def sample_selection(samples:list, return_index=False):
	"""
	Display a list of samples and return the selected samples.
	
	Parameters
	----------
	samples : list
		List of samples to be displayed.
		return_index : bool
		If True, return the index of the selected samples. If False, return the samples.
	"""
	selected = st.dataframe(
		pd.DataFrame(samples, columns=['Samples']),
		on_select="rerun",
		selection_mode="multi-row",
		hide_index=True,
		use_container_width=True
	)
	rows = selected.selection.rows
	if return_index:
		return rows
	else:
		return [samples[i] for i in rows]
	