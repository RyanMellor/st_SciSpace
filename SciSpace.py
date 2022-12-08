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

scispace_url = "http://sci-space.co.uk/"
logo_url = "http://sci-space.co.uk/scispace.png"


st.sidebar.markdown(
    f"""
    <div>
        <a href="{scispace_url}" target="_blank">
            <img src="{logo_url}" alt="SciSpace">
        </a>
        <p></p>
        <a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)





def main():
    # Branding
	branding = f"""
		<div>
			<a href="http://sci-space.co.uk/" target="_blank">
				<img src="http://sci-space.co.uk/scispace.png" alt="SciSpace">
			</a>
			<p></p>
			<a href="https://www.buymeacoffee.com/ryanmellor" target="_blank">
				<img src="https://cdn.buymeacoffee.com/buttons/default-black.png" alt="Buy Me A Coffee" height="41" width="174">
			</a>
		</div>
		"""
	st.sidebar.markdown(branding, unsafe_allow_html=True,)

	# --- HIDE STREAMLIT STYLE ---
	hide_st_style = """
		<style>
			MainMenu {visibility: hidden;}
			footer {visibility: hidden;}
			header {visibility: hidden;}
		</style>
		"""
	st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
