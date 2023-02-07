import streamlit as st
import pandas as pd
import os

@st.cache()
def file_to_df(data_file, delimiter=",", index_col=None, header='infer'):
	ext = ""
	if isinstance(data_file, st.runtime.uploaded_file_manager.UploadedFile):
		ext = os.path.splitext(data_file.name)[-1][1:]
	else:
		ext = os.path.splitext(data_file)[-1][1:]

	if ext in ['xls', 'xlsx']:
		return pd.read_excel(data_file, index_col=index_col, header=header)
	elif ext in ['csv', 'txt']:
		return pd.read_csv(data_file, index_col=index_col, header=header, delimiter=delimiter)
	else:
		print("Unsupported file format")
		return None