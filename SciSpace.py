import streamlit as st
# from streamlit_extras.app_logo import add_logo

# st.set_option('deprecation.showfileUploaderEncoding', False)

PRIMARY_COLOR = "#4589ff"

st.set_page_config(
    page_title="SciSpace",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("SciSpace")

logo_url = "http://sci-space.co.uk/scispace.png"

st.markdown(
    f"""
    <style>
        [data-testid="stSidebarNav"] {{
            background-image: url({logo_url});
            background-repeat: no-repeat;
            padding-top: 80px;
            background-position: 20px 20px;
            href: "http://sci-space.co.uk/";
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# st.sidebar.markdown("[![SciSpace](http://sci-space.co.uk/scispace.png)](http://sci-space.co.uk/)")

# --- HIDE STREAMLIT STYLE ---
# hide_st_style = """
#             <style>
#             # MainMenu {visibility: hidden;}
#             footer {visibility: hidden;}
#             header {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_st_style, unsafe_allow_html=True)


def main():

    pass


if __name__ == '__main__':
    main()
