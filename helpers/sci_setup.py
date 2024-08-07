import streamlit as st
from PIL import Image

def logo():
    return Image.open('./scispace.png')

def fav():
    return Image.open('./favicon.ico')

def setup_page(page_title):

    st.set_page_config(
        page_title=page_title,
        page_icon=fav(),
        layout="centered"
    )

    st.title(page_title)

    st.sidebar.image(logo())

    page_setup = """
        <div>
            <a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
                <img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
            </a>
        </div>
        <hr/>
        <style>
            footer {visibility: hidden;}
            [data-testid="stTickBar"] {height:0; visibility:hidden;}
            thead tr th:first-child {display:none}
            tbody th {display:none}
            [data-testid="stFileUploadDropzone"] {display:grid}
            [data-testid="stSidebarNav"] ul {max-height:none}
        </style>
    """
    st.sidebar.markdown(page_setup, unsafe_allow_html=True,)

    # [data-testid="stSidebarNav"] .ul {max-heigh:fit-content}

def plot_layout ():
    return {
	"template": 'plotly_dark',
	"margin": dict(l=20, r=20, t=20, b=20),
	"height":300,
	"legend": {},
    "title": {},
	"xaxis": {},
	"yaxis": {},
	# "uirevision": "foo",
    }