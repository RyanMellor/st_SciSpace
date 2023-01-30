import streamlit as st
from PIL import Image
from helpers import setup

PRIMARY_COLOR = "#4589ff"

setup.setup_page("Scispace")

def main():
    st.markdown('''
        ---
        Homepage of SciSpace Streamlit applications.

        Visit [SciSpace](http://sci-space.co.uk) for more information

        ---
    ''')

    st.image(setup.logo())

if __name__ == '__main__':
    main()
