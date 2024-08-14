import streamlit as st
# from st_pages import Page, Section, show_pages
from helpers import sci_setup
sci_setup.setup_page("SciSpace")

PRIMARY_COLOR = "#4589ff"

def main():
    st.markdown('''
        ---
        Homepage of [SciSpace](http://sci-space.co.uk) Streamlit applications.

        Many apps are still in the testing phase, expect bugs and please report them.

        ---
    ''')

    st.image(sci_setup.logo())

    # page_deconvolution = st.Page("pages/101_Deconvolution.py", title="Deconvolution", )
    # page_raman_signal_processor = st.Page("pages/102_Raman Signal Processing.py", title="Raman Signal Processing", )
    # page_quantitative_signal_analyser = st.Page("pages/103_Quantitative Signal Analysis.py", title="Quantitative Signal Analysis", )

    # pg = st.navigation(
    #     {
    #         "Spectoscopy": [
    #             page_deconvolution,
    #             page_raman_signal_processor,
    #             page_quantitative_signal_analyser
    #         ],
    #     }
    # )
    # pg.run()

if __name__ == '__main__':
    main()
