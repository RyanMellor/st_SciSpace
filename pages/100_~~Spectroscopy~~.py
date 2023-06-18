import streamlit as st
from helpers import sci_setup
sci_setup.setup_page("Spectroscopy")

st.markdown('''
Visit [SciSpace - Spectroscopy](https://docs.sci-space.co.uk/methods-in-pharmacy/analysis/spectroscopy) for more information on spectroscopic techniquies.
''')

st.image(sci_setup.logo())