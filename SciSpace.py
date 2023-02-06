import streamlit as st
# from st_pages import Page, Section, show_pages
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

    # show_pages(
    #     [
    #         Page("SciSpace.py.py", "Home"),
    #         Section("Spectral"),
    #         Page("apps/101_Deconvolution.py", "Deconvolution"),
    #         Page("apps/102_Raman Signal Processor.py", "Raman Signal Processor"),
    #         Page("apps/103_Quantitative Signal Analyser.py", "Quantitative Signal Analyser")
    #     ]
    # )

if __name__ == '__main__':
    main()
