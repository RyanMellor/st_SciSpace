import streamlit as st
from PIL import Image
from helpers import setup

PRIMARY_COLOR = "#4589ff"

setup.setup_page("SciSpace")

def main():
    st.markdown('''
        ---
        Homepage of [SciSpace](http://sci-space.co.uk) Streamlit applications.

        Many apps are still in the testing phase, expect bugs and please report them.

        ---
    ''')

    st.image(setup.logo())

if __name__ == '__main__':
    main()
