import streamlit as st
from PIL import Image
import requests
from io import BytesIO


PRIMARY_COLOR = "#4589ff"

# ---- Page setup ----
fav_url = r"http://sci-space.co.uk//favicon.ico"
fav_response = requests.get(fav_url)
fav = Image.open(BytesIO(fav_response.content))
st.set_page_config(
    page_title="SciSpace",
    page_icon=fav,
)

st.title("SciSpace")

page_setup = """
	<div>
		<a href="http://sci-space.co.uk/" target="_blank">
			<img src="http://sci-space.co.uk/scispace.png" alt="SciSpace">
		</a>
		<p></p>
		<a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
			<img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
		</a>
	</div>
	<hr/>
	<style>
		footer {visibility: hidden;}
		[data-testid="stTickBar"] {{
			height:0;
			visibility:hidden;
		}}
	</style>
	"""
st.sidebar.markdown(page_setup, unsafe_allow_html=True,)





def main():
    pass

if __name__ == '__main__':
    main()
