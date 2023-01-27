import streamlit as st
from PIL import Image

PRIMARY_COLOR = "#4589ff"

logo = Image.open('./scispace.png')
fav = Image.open('./favicon.ico')

# ---- Page setup ----
st.set_page_config(
    page_title="SciSpace",
    page_icon=fav,
)

st.title("SciSpace")

st.sidebar.image(logo)

page_setup = """
	<div>
		<a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
			<img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
		</a>
	</div>
	<hr/>
	<style>
		footer {visibility: hidden;}
	</style>
"""
st.sidebar.markdown(page_setup, unsafe_allow_html=True,)

def main():
    st.markdown('''
        ---
        Homepage of SciSpace Streamlit applications.

        Visit [SciSpace](http://sci-space.co.uk) for more information

        ---
    ''')

    st.image(logo)

if __name__ == '__main__':
    main()
