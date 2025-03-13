import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import math
import os
os.environ["OMP_NUM_THREADS"] = '1'

# Import helpers from your existing codebase
from helpers import sci_setup, sci_data, sci_image
from helpers.sci_style import *
sci_setup.setup_page("Western Blot Analysis")

import warnings
warnings.filterwarnings('ignore')

FILETYPES_IMG = ['bmp', 'gif', 'jpg', 'jpeg', 'png', 'tif', 'tiff']
MARKER_COLOR = "goldenrod"


# ---- Functions ----

@st.cache_data(show_spinner=False)
def merge_channels(red=None, green=None, blue=None):
    """
    Merge individual channel images into an RGB image.
    
    Parameters:
    -----------
    red, green, blue : PIL.Image or numpy.ndarray or None
        Individual channel images. None channels will be filled with zeros.
        
    Returns:
    --------
    numpy.ndarray
        Merged RGB image
    """
    # Determine image dimensions from the first non-None channel
    sample = next(img for img in [red, green, blue] if img is not None)
    if isinstance(sample, Image.Image):
        sample = np.array(sample)
    
    height, width = sample.shape[:2]
    
    # Create empty RGB image
    rgb_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Fill channels
    for idx, channel in enumerate([red, green, blue]):
        if channel is not None:
            if isinstance(channel, Image.Image):
                channel = np.array(channel)
            if len(channel.shape) == 3 and channel.shape[2] >= 3:
                # If input is already RGB, extract just the needed channel
                rgb_img[:, :, idx] = channel[:, :, idx]
            else:
                # Grayscale input
                rgb_img[:, :, idx] = channel
    
    return rgb_img

@st.cache_data(show_spinner=False)
def get_lanes(img, params):
    """
    Calculate lane positions based on manual settings and return lane images.
    
    Parameters:
    -----------
    img : PIL.Image or numpy.ndarray
        The input gel image
    params : dict
        Detection parameters including:
        - lane_count: Number of lanes
        - lane_width: Width of detected lanes in pixels
        - lane_offsets: Manual offsets from evenly spaced positions (optional)
        
    Returns:
    --------
    tuple
        (Output image with lanes marked, lane positions, cropped lane images)
    """
    # Convert to numpy array if it's a PIL image
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
        
    # Create output image for visualization
    img_output = img_np.copy() if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Get parameters
    lane_count = params.get("lane_count", 8)  # Default to 8 lanes if not specified
    lane_width = params.get("lane_width", 34)
    lane_offsets = params.get("lane_offsets", [0] * lane_count)
    
    # Make sure we have the right number of offsets
    if len(lane_offsets) != lane_count:
        lane_offsets = [0] * lane_count
    
    # Calculate positions for evenly spaced lanes
    img_width = img_np.shape[1]
    
    # Set margins at 5% of image width from each edge
    margin = int(img_width * 0.05)
    
    # Calculate base lane positions (evenly spaced)
    if lane_count == 1:
        # Single lane in the center
        base_positions = np.array([img_width // 2])
    else:
        # Multiple evenly spaced lanes
        base_positions = np.linspace(
            margin, 
            img_width - margin, 
            lane_count
        ).astype(int)
    
    # Apply offsets to get final lane positions
    lane_positions = [base_positions[i] + lane_offsets[i] for i in range(lane_count)]
    
    # Create list to store cropped lane images
    lane_images = []
    
    # Draw lane boxes and extract lane images
    for i, lane_pos in enumerate(lane_positions):
        start_x = max(0, int(lane_pos - lane_width/2))
        end_x = min(img_output.shape[1]-1, int(lane_pos + lane_width/2))
        
        # Draw box around lane
        cv2.rectangle(img_output, (start_x, 0), (end_x, img_output.shape[0]-1), THEME_PRIMARY_RGB, 1)
        
        # Add lane number
        cv2.putText(img_output, f"{i+1}", (start_x+5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Extract lane image
        lane_img = img_np[0:img_np.shape[0], start_x:end_x].copy()
        lane_images.append(lane_img)
    
    return img_output, lane_positions, lane_images

@st.cache_data(show_spinner=False)
def prepare_display_image(img, channel="Grayscale", invert=False):
    """
    Prepare image for display with current analysis settings applied.
    
    Parameters:
    -----------
    img : PIL.Image or numpy.ndarray
        The input image
    channel : str
        Channel to display ("Grayscale", "Red", "Green", "Blue")
    invert : bool
        Whether to invert the image
        
    Returns:
    --------
    numpy.ndarray
        Processed image ready for display
    """
    # Convert to numpy array if it's a PIL image
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img.copy()
    
    # Extract the requested channel
    if len(img_np.shape) == 3 and img_np.shape[2] >= 3:
        if channel == "Red":
            # Create RGB image with only red channel
            display_img = np.zeros_like(img_np)
            display_img[:,:,0] = img_np[:,:,0]
        elif channel == "Green":
            # Create RGB image with only green channel
            display_img = np.zeros_like(img_np)
            display_img[:,:,1] = img_np[:,:,1]
        elif channel == "Blue":
            # Create RGB image with only blue channel
            display_img = np.zeros_like(img_np)
            display_img[:,:,2] = img_np[:,:,2]
        elif channel == "Grayscale":
            # Convert to grayscale for display
            display_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
        else:
            # Default to original image
            display_img = img_np
    else:
        # Already grayscale
        display_img = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Invert if requested
    if invert:
        display_img = 255 - display_img
    
    return display_img

@st.cache_data(show_spinner=False)
def detect_bands_all_lanes(img, lane_positions, lane_width, params):
    """
    Detect bands across all lanes in the gel.
    
    Parameters:
    -----------
    img : PIL.Image or numpy.ndarray
        The input gel image
    lane_positions : list
        List of horizontal positions for each lane
    lane_width : int
        Width of each lane in pixels
    params : dict
        Detection parameters including:
        - invert_val: Whether to invert intensity values
        - band_threshold: Detection threshold (fraction of max)
        - band_prominence: Required prominence for peaks
        - band_min_dist: Minimum distance between bands
        
    Returns:
    --------
    dict
        Detection results for all lanes and channels
    """
    results = {}
    
    # Process each lane
    for lane_idx, lane_pos in enumerate(lane_positions):
        lane_results = detect_bands(img, lane_pos, lane_width, params)
        results[lane_idx+1] = lane_results
    
    return results


@st.cache_data(show_spinner=False)
def mark_all_bands_on_image(img, all_band_results, lane_positions, lane_width, channel="Grayscale"):
    """
    Create an image with all detected bands marked.
    
    Parameters:
    -----------
    img : PIL.Image or numpy.ndarray
        The input gel image (already processed for display)
    all_band_results : dict
        Results from detect_bands_all_lanes
    lane_positions : list
        List of horizontal positions for each lane
    lane_width : int
        Width of each lane in pixels
    channel : str
        Channel to use for band visualization
        
    Returns:
    --------
    numpy.ndarray
        Image with bands marked
    """
    # Convert to numpy array if it's a PIL image
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
        
    # Create output image for visualization
    img_output = img_np.copy() if len(img_np.shape) == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    # Draw lane boxes and bands
    for lane_idx, lane_pos in enumerate(lane_positions):
        lane_num = lane_idx + 1
        start_x = max(0, int(lane_pos - lane_width/2))
        end_x = min(img_output.shape[1]-1, int(lane_pos + lane_width/2))
        
        # Draw box around lane
        cv2.rectangle(img_output, (start_x, 0), (end_x, img_output.shape[0]-1), THEME_PRIMARY_RGB, 1)
        
        # Add lane number
        cv2.putText(img_output, f"{lane_num}", (start_x+5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw bands if available
        if lane_num in all_band_results and channel in all_band_results[lane_num]:
            channel_data = all_band_results[lane_num][channel]
            peaks = channel_data["peaks"]
            
            for peak_pos in peaks:
                # Draw circle at peak position
                cv2.circle(
                    img_output,
                    (lane_pos, peak_pos),
                    5,
                    (0, 0, 0),  # black border
                    -1
                )
                cv2.circle(
                    img_output, 
                    (lane_pos, peak_pos), 
                    4, 
                    (218, 165, 32),  # goldenrod color in BGR
                    -1
                )
    
    return img_output

@st.cache_data(show_spinner=False)
def detect_bands(img, lane_pos, lane_width, params):
    """
    Detect bands within a lane without normalization.
    
    Parameters:
    -----------
    img : PIL.Image or numpy.ndarray
        The input gel image
    lane_pos : int
        Horizontal position of the lane center
    lane_width : int
        Width of the lane in pixels
    params : dict
        Detection parameters
        
    Returns:
    --------
    dict
        Detection results for each channel
    """
    # Convert to numpy array if it's a PIL image
    if isinstance(img, Image.Image):
        img_np = np.array(img)
    else:
        img_np = img
    
    # Extract region of interest for the lane
    start_x = max(0, lane_pos - lane_width//2)
    end_x = min(img_np.shape[1], lane_pos + lane_width//2)
    
    # Process different channels
    if len(img_np.shape) == 3:
        # Extract channels
        channels = {}
        if img_np.shape[2] >= 3:  # RGB
            channels = {
                "Red": img_np[:,:,0],
                "Green": img_np[:,:,1],
                "Blue": img_np[:,:,2]
            }
            # Add Alpha if present
            if img_np.shape[2] == 4:
                channels["Alpha"] = img_np[:,:,3]
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        channels["Grayscale"] = img_gray
    else:
        img_gray = img_np
        channels = {"Grayscale": img_gray}
    
    # Analyse each channel
    results = {}
    for channel_name, channel_data in channels.items():
        # Extract lane region
        lane_data = channel_data[:, start_x:end_x]
        
        # Average across lane width
        intensity_profile = np.mean(lane_data, axis=1)
        
        # Apply smoothing
        intensity_smooth = savgol_filter(intensity_profile, 15, 3)
        
        # Invert if needed (dark bands on light background)
        if params["invert_val"]:
            intensity_smooth = np.max(intensity_smooth) - intensity_smooth
            intensity_profile = np.max(intensity_profile) - intensity_profile
        
        # Detect peaks (bands)
        peaks, properties = find_peaks(
            intensity_smooth,
            height=params["band_threshold"] * np.max(intensity_smooth) if np.max(intensity_smooth) > 0 else 0,
            distance=params["band_min_dist"],
            prominence=params["band_prominence"] * np.max(intensity_smooth) if np.max(intensity_smooth) > 0 else 0
        )
        
        results[channel_name] = {
            "intensity_raw": intensity_profile,
            "intensity_smooth": intensity_smooth,
            "peaks": peaks,
            "properties": properties
        }
    
    return results

@st.cache_data(show_spinner=False)
def calculate_molecular_weights(positions, calibration_data):
    """Calculate MWs based on calibration curve"""
    # Extract calibration points
    positions_cal = [point[1] for point in calibration_data]
    weights_cal = [point[0] for point in calibration_data]
    
    # Convert to log scale for MWs
    log_weights = np.log10(weights_cal)
    
    # Fit a polynomial to the calibration data
    if len(positions_cal) > 1:
        z = np.polyfit(positions_cal, log_weights, 1)
        p = np.poly1d(z)
        
        # Calculate MWs for positions
        log_mw = p(positions)
        mw = 10 ** log_mw
        
        return mw, p, z
    
    return None, None, None

@st.cache_data(show_spinner=False)
def open_img(path):
    return Image.open(path)

@st.cache_data(show_spinner=False)
def rotate_lane_image(lane_img):
    """Rotate lane image to horizontal orientation"""
    if isinstance(lane_img, Image.Image):
        lane_img = np.array(lane_img)
    
    # Change to ROTATE_90_COUNTERCLOCKWISE
    rotated = cv2.rotate(lane_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated

# ---- Initialize Session State ----

def initialize_session_state():
    """Initialize all session state variables"""
    if 'ladder_markers' not in st.session_state:
        st.session_state.ladder_markers = []
    
    if 'mw_calibration' not in st.session_state:
        st.session_state.mw_calibration = None
        
    if 'mw_coefficients' not in st.session_state:
        st.session_state.mw_coefficients = None
    
    if 'band_params' not in st.session_state:
        st.session_state.band_params = {
            "invert_val": False,
            "band_threshold": 0.05,
            "band_prominence": 0.05,
            "band_min_dist": 10
        }
        
    if 'lane_offsets' not in st.session_state:
        st.session_state.lane_offsets = []
        
    if 'all_bands_detected' not in st.session_state:
        st.session_state.all_bands_detected = False
        
    if 'band_detection_results' not in st.session_state:
        st.session_state.band_detection_results = None
    
    if 'ladder_lane' not in st.session_state:
        st.session_state.ladder_lane = 1
        
    if 'ladder_channel' not in st.session_state:
        st.session_state.ladder_channel = "Grayscale"
    
    # Add analysis_channel to session state
    if 'analysis_channel' not in st.session_state:
        st.session_state.analysis_channel = "Grayscale"
        

@st.fragment
def image_input_section():
    """Handle image input through file uploads and pass to next section"""
    st.markdown("### Image Input")
    container = st.container(border=True)
    with container:
        input_method = st.radio(
            "Choose input method:",
            options=["Single RGB Image", "Individual Channels"],
            horizontal=True
        )
        
        if input_method == "Single RGB Image":
            img_file = st.file_uploader(
                label='Upload RGB image file', 
                type=FILETYPES_IMG, 
                label_visibility='collapsed', 
                on_change=sci_image.new_canvas_key
            )

            if not img_file:
                st.info("Upload a Western blot gel image")
                return None
            
            img_original = open_img(img_file)
            img_original = img_original.convert("RGB")
            
            # Set initial ROI position
            initial_roi_pos = (0.05, 0.05, 0.9, 0.9)
            
            # Perform cropping using sci_image.crop
            img_cropped = sci_image.crop(img_original, initial_roi_pos)
            
        else:  # Individual Channels
            st.markdown("Upload each channel separately (will be merged into RGB)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                red_file = st.file_uploader(
                    label='Upload Red channel image', 
                    type=FILETYPES_IMG,
                    key="red_channel"
                )
                if red_file:
                    red_img = open_img(red_file).convert("L")
                    st.image(red_img, caption="Red Channel", use_container_width=True)
                else:
                    red_img = None
            
            with col2:
                green_file = st.file_uploader(
                    label='Upload Green channel image', 
                    type=FILETYPES_IMG,
                    key="green_channel"
                )
                if green_file:
                    green_img = open_img(green_file).convert("L")
                    st.image(green_img, caption="Green Channel", use_container_width=True)
                else:
                    green_img = None
            
            with col3:
                blue_file = st.file_uploader(
                    label='Upload Blue channel image', 
                    type=FILETYPES_IMG,
                    key="blue_channel"
                )
                if blue_file:
                    blue_img = open_img(blue_file).convert("L")
                    st.image(blue_img, caption="Blue Channel", use_container_width=True)
                else:
                    blue_img = None
            
            # Merge channels if at least one is uploaded
            if any([red_img, green_img, blue_img]):
                img_cropped = merge_channels(red_img, green_img, blue_img)
                st.image(img_cropped, caption="Merged RGB Image", use_container_width=True)
            else:
                st.info("Upload at least one channel image")
                return None
    
    if img_cropped is not None:
        # Continue to lane detection if we have an image
        lane_detection_section(img_cropped)
    
    return img_cropped

@st.fragment
def lane_detection_section(img_cropped):
    """Handle lane detection UI and processing, then pass to next section"""
    st.markdown("### Define Lanes")
    container = st.container(border=True)
    with container:
        col_content, col_settings = st.columns([8, 4])

        with col_settings:
            st.markdown("#### Settings")
            invert_val = st.checkbox("Invert intensity values", value=st.session_state.band_params["invert_val"])
            lane_count = st.slider("Number of Lanes", min_value=1, max_value=20, value=13)
            lane_width = st.slider("Lane Width", min_value=10, max_value=100, value=25)
            
            # Update band parameters
            st.session_state.band_params["invert_val"] = invert_val
            
            # Initialize or resize lane_offsets if lane_count changes
            if len(st.session_state.lane_offsets) != lane_count:
                st.session_state.lane_offsets = [0] * lane_count
            
            lane_detection_settings = {
                "lane_count": lane_count,
                "lane_width": lane_width,
                "invert_val": invert_val,
                "lane_offsets": st.session_state.lane_offsets
            }
        
        with col_content:
            st.caption("Edit lane offsets to adjust lane positions")

            # Create a dataframe with a column per lane for offset adjustment
            offset_data = {}
            
            # Create one column for each lane
            for i in range(lane_count):
                lane_num = i + 1
                offset_data[lane_num] = [st.session_state.lane_offsets[i]]
            
            # Use Streamlit's dataframe with editing enabled
            edited_offset_df = st.data_editor(
                pd.DataFrame(offset_data),
                hide_index=True,
                use_container_width=True,
                num_rows=1
            )
            
            # Update lane offsets from edited dataframe
            for i in range(lane_count):
                lane_num = i + 1
                if lane_num in edited_offset_df.columns:
                    st.session_state.lane_offsets[i] = edited_offset_df[lane_num].iloc[0]
            
            lane_detection_settings["lane_offsets"] = st.session_state.lane_offsets
            
            # Process the image according to current settings
            if invert_val:
                display_img = prepare_display_image(img_cropped, invert=True)
            else:
                display_img = img_cropped
                
            # Get lane positions and visualization
            img_lanes, lane_positions, lane_images = get_lanes(display_img, lane_detection_settings)
            st.image(img_lanes, caption="Lane Boundaries", use_container_width=True)
    
    # Continue to band detection
    band_detection_section(img_cropped, lane_positions, lane_width, lane_images)
    
    return lane_positions, lane_width, lane_images

@st.fragment
def band_detection_section(img_cropped, lane_positions, lane_width, lane_images):
    """Handle band detection UI and processing, then pass to next section"""
    st.markdown("### Detect Bands")
    container = st.container(border=True)    
    with container:
        # Split into columns
        col_content, col_settings = st.columns([8, 4])
        
        with col_settings:
            st.markdown("#### Settings")
            
            band_threshold = st.slider(
                "Band detection threshold", 
                min_value=0.0, 
                max_value=0.5, 
                value=st.session_state.band_params["band_threshold"],
                step=0.01,
                help="Threshold for band detection (fraction of max intensity)"
            )
            
            band_prominence = st.slider(
                "Band prominence", 
                min_value=0.0, 
                max_value=0.5, 
                value=st.session_state.band_params["band_prominence"],
                step=0.01,
                help="Required prominence for band detection (fraction of max)"
            )
            
            band_min_dist = st.slider(
                "Min band distance", 
                min_value=5, 
                max_value=100, 
                value=st.session_state.band_params["band_min_dist"],
                help="Minimum distance between bands (pixels)"
            )
            
            display_channel = "Grayscale"  # Default
            
            # Get all possible channels
            if len(np.array(img_cropped).shape) == 3 and np.array(img_cropped).shape[2] >= 3:
                channel_options = ["Grayscale", "Red", "Green", "Blue"]
            else:
                channel_options = ["Grayscale"]
                
            display_channel = st.selectbox(
                "Select channel to display", 
                options=channel_options,
                index=0
            )
            
            band_params = {
                "invert_val": st.session_state.band_params["invert_val"],
                "band_threshold": band_threshold,
                "band_prominence": band_prominence,
                "band_min_dist": band_min_dist
            }
            
            # Update session state
            st.session_state.band_params = band_params
            
            with st.spinner("Detecting bands..."):
                all_results = detect_bands_all_lanes(
                    img_cropped, 
                    lane_positions, 
                    lane_width, 
                    band_params
                )
                
                # Store results in session state
                st.session_state.band_detection_results = all_results
                st.session_state.all_bands_detected = True

        with col_content:
            if st.session_state.all_bands_detected and st.session_state.band_detection_results:
                # Prepare the display image with current settings
                display_img = prepare_display_image(
                    img_cropped,
                    channel=display_channel,
                    invert=st.session_state.band_params["invert_val"]
                )
                
                # Create and display image with all bands marked
                img_bands = mark_all_bands_on_image(
                    display_img, 
                    st.session_state.band_detection_results, 
                    lane_positions, 
                    lane_width,
                    channel=display_channel
                )
                
                st.image(img_bands, caption=f"All Detected Bands ({display_channel} channel)", use_container_width=True)
            else:
                st.error("Band detection failed. Please adjust parameters.")
    
    # Continue to ladder calibration with results
    if st.session_state.all_bands_detected:
        ladder_calibration_section(lane_positions, lane_width, lane_images)
    
    return st.session_state.band_detection_results

@st.fragment
def ladder_calibration_section(lane_positions, lane_width, lane_images):
    """Handle ladder calibration UI and processing, then pass to next section"""
    st.markdown("### Ladder Calibration")
    container = st.container(border=True)
    
    with container:
        # Split into columns for settings and calibration curve
        col_content, col_settings = st.columns([8, 4])
        
        with col_settings:
            st.markdown("#### Settings")
            # Select ladder lane
            ladder_lane = st.selectbox(
                "Select ladder lane", 
                options=list(range(1, len(lane_positions)+1)),
                index=list(range(1, len(lane_positions)+1)).index(st.session_state.ladder_lane) 
                    if st.session_state.ladder_lane in range(1, len(lane_positions)+1) else 0,
                key="ladder_lane_select"
            )
            
            # Update session state with ladder lane
            st.session_state.ladder_lane = ladder_lane
            
            # Get results for the ladder lane
            if ladder_lane in st.session_state.band_detection_results:
                ladder_results = st.session_state.band_detection_results[ladder_lane]
                
                # Get channel options for this lane
                channel_options = list(ladder_results.keys())
                
                # Add ladder channel selector
                ladder_channel = st.selectbox(
                    "Select channel for ladder calibration", 
                    options=channel_options,
                    index=channel_options.index(st.session_state.ladder_channel) 
                        if st.session_state.ladder_channel in channel_options else 
                        (channel_options.index("Grayscale") if "Grayscale" in channel_options else 0),
                    key="ladder_channel_select"
                )
                
                # Update session state with ladder channel
                st.session_state.ladder_channel = ladder_channel
                
                # Get band data for selected channel
                if ladder_channel in ladder_results:
                    ladder_data = ladder_results[ladder_channel]
                    detected_bands = ladder_data["peaks"]
                    
                    # If no bands detected, show error
                    if len(detected_bands) == 0:
                        st.error("No bands detected in the ladder lane. Please adjust band detection parameters.")
                    else:
                        # Common MW ladder values
                        common_weights = [250, 150, 100, 75, 50, 37, 25, 20, 15, 10]
                        
                        # Initialize MW inputs if needed
                        if len(st.session_state.ladder_markers) != len(detected_bands):
                            st.session_state.ladder_markers = []
                            for i, band_pos in enumerate(detected_bands):
                                weight = common_weights[i] if i < len(common_weights) else 10
                                st.session_state.ladder_markers.append((weight, band_pos))
                        
                        # Create MW input as editable dataframe
                        mw_data = {"Position (px)": [], "MW (kDa)": []}
                        for i, (weight, band_pos) in enumerate(st.session_state.ladder_markers):
                            mw_data["Position (px)"].append(band_pos)
                            mw_data["MW (kDa)"].append(weight)
                        
                        mw_df = pd.DataFrame(mw_data)
                        
                        # Add a unique key that changes with ladder lane and channel to force re-render
                        editor_key = f"mw_editor_{ladder_lane}_{ladder_channel}"
                        
                        # Use Streamlit's dataframe with editing enabled
                        edited_df = st.data_editor(
                            mw_df, 
                            num_rows="fixed",
                            disabled=["Position (px)"],
                            hide_index=True,
                            use_container_width=True,
                            key=editor_key,
                            column_config={
                                "MW (kDa)": st.column_config.NumberColumn(
                                    "MW (kDa)", 
                                    min_value=1.0, 
                                    max_value=500.0, 
                                    step=0.1,
                                    format="%.1f"
                                )
                            }
                        )
                        
                        # Create a button to explicitly apply changes
                        if st.button("Apply MW Changes", key=f"apply_mw_{ladder_lane}_{ladder_channel}"):
                            # Update ladder markers from edited dataframe
                            st.session_state.ladder_markers = [
                                (row["MW (kDa)"], row["Position (px)"])
                                for _, row in edited_df.iterrows()
                            ]
                            
                            # Force recalculation of calibration
                            if len(st.session_state.ladder_markers) > 1:
                                weights = [m[0] for m in st.session_state.ladder_markers]
                                positions = [m[1] for m in st.session_state.ladder_markers]
                                
                                mw_estimates, calibration_fn, coefficients = calculate_molecular_weights(
                                    positions, 
                                    st.session_state.ladder_markers
                                )
                                
                                # Store in session state
                                st.session_state.mw_calibration = calibration_fn
                                st.session_state.mw_coefficients = coefficients
                                
                                # Force rerun to display the calibration curve
                                st.rerun()
                            else:
                                st.error("At least two calibration points are needed.")
        
        with col_content:
            # Display calibration curve if available
            if st.session_state.mw_calibration is not None:              
                weights = [m[0] for m in st.session_state.ladder_markers]
                positions = [m[1] for m in st.session_state.ladder_markers]
                
                # Calculate calibration
                if st.session_state.mw_coefficients is not None:
                    coefficients = st.session_state.mw_coefficients
                else:
                    _, _, coefficients = calculate_molecular_weights(positions, st.session_state.ladder_markers)
                
                # Plot calibration curve
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=positions, 
                    y=np.log10(weights), 
                    mode='markers+lines', 
                    name='Calibration Points',
                    marker=dict(color=THEME_PRIMARY, size=8)
                ))
                
                # Add fit line
                fit_x = np.linspace(min(positions) - 10, max(positions) + 10, 100)
                fit_y = st.session_state.mw_calibration(fit_x)
                
                fig.add_trace(go.Scatter(
                    x=fit_x, 
                    y=fit_y, 
                    mode='lines', 
                    name='Calibration Fit',
                    line=dict(color=MARKER_COLOR, dash='dash')
                ))
                
                fig.update_layout(
                    xaxis_title="Vertical Position (pixels)",
                    yaxis_title="Log10(MW)",
                    margin=dict(l=20, r=20, t=20, b=20),
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display equation
                equation = f"$log_{{10}}(MW) = {coefficients[0]:.6f} \\times \\text{{position}} + {coefficients[1]:.4f}$"
                st.markdown(equation)
    
    # Continue to analysis only if we have a calibration
    if st.session_state.mw_calibration is not None:
        analysis_results_section(lane_positions, lane_width, lane_images)
                
    return st.session_state.mw_calibration

@st.fragment
def analysis_results_section(lane_positions, lane_width, lane_images):
    """Display analysis results section with tabs"""
    st.markdown("### Analyse Results")
    container = st.container(border=True)    
    with container:
        if st.session_state.mw_calibration is None:
            st.warning("Please complete the ladder calibration first.")
            return
            
        # Tabs for Single Lane and Lane Comparison
        analysis_tab1, analysis_tab2 = st.tabs(["Single Lane Analysis", "Lane Comparison"])
        
        # Single Lane Analysis
        with analysis_tab1:
            single_lane_analysis(lane_positions, lane_width, lane_images)
        
        # Lane Comparison
        with analysis_tab2:
            lane_comparison_analysis(lane_positions, lane_width, lane_images)

@st.fragment
def single_lane_analysis(lane_positions, lane_width, lane_images):
    """Handle single lane analysis section"""
    col_content, col_settings = st.columns([8, 4])
    
    with col_settings:
        # Channel selection for analysis
        channel_options = list(st.session_state.band_detection_results[1].keys())
        analysis_channel = st.selectbox(
            "Select channel for analysis", 
            options=channel_options,
            index=channel_options.index(st.session_state.analysis_channel) if st.session_state.analysis_channel in channel_options else 0,
            key="analysis_channel_single"
        )
        
        # Store the selected channel in session state
        st.session_state.analysis_channel = analysis_channel
        
        # Get ladder lane from session state for defaulting
        ladder_lane = st.session_state.ladder_lane if 'ladder_lane' in st.session_state else 1
        
        selected_lane = st.selectbox(
            "Select lane to analyse", 
            options=list(range(1, len(lane_positions)+1)),
            index=0 if ladder_lane != 1 else 1,
            key="analysis_lane_select"
        )
    
    with col_content:
        # Display lane image and analysis
        if selected_lane in st.session_state.band_detection_results:
            lane_results = st.session_state.band_detection_results[selected_lane]
            
            if analysis_channel in lane_results:
                display_lane_analysis(selected_lane, analysis_channel, lane_results, lane_images)

    # Band data table and other full width content
    if selected_lane in st.session_state.band_detection_results:
        lane_results = st.session_state.band_detection_results[selected_lane]
        
        if analysis_channel in lane_results:
            display_lane_data_table(selected_lane, analysis_channel, lane_results)


def display_lane_analysis(selected_lane, analysis_channel, lane_results, lane_images):
    """Display analysis for a single lane"""
    channel_data = lane_results[analysis_channel]
    intensity_profile = channel_data["intensity_raw"]
    peaks = channel_data["peaks"]
    properties = channel_data["properties"]
    
    # Calculate MWs
    if len(peaks) > 0:
        mw_estimates, _, _ = calculate_molecular_weights(peaks, st.session_state.ladder_markers)
    else:
        mw_estimates = None
    
    # Plot chromatogram with MW labels
    fig = go.Figure()
    
    # Intensity profile
    fig.add_trace(go.Scatter(
        x=list(range(len(intensity_profile))), 
        y=intensity_profile, 
        mode='lines', 
        name='Intensity',
        line=dict(color=THEME_PRIMARY)
    ))
    
    # Add markers for detected peaks with MW labels
    if len(peaks) > 0:
        peak_text = []
        if mw_estimates is not None:
            for i, mw in enumerate(mw_estimates):
                peak_text.append(f"{mw:.1f} kDa")
        else:
            peak_text = [f"Peak {i+1}" for i in range(len(peaks))]
        
        fig.add_trace(go.Scatter(
            x=peaks, 
            y=intensity_profile[peaks], 
            mode='markers+text', 
            name='Detected Bands',
            marker=dict(color=MARKER_COLOR, size=8),
            text=peak_text,
            textposition="top center"
        ))
    
    fig.update_layout(
        title=f"Lane {selected_lane} - {analysis_channel} Channel",
        xaxis_title="Vertical Position (pixels)",
        yaxis_title="Intensity",
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display rotated lane image below the plot to align with bands
    if 0 <= selected_lane - 1 < len(lane_images):
        col_lane_number, col_lane_images = st.columns([2, 10])
        with col_lane_number:
            st.write(f"Lane {selected_lane}")
        with col_lane_images:
            lane_img = lane_images[selected_lane - 1]
            
            # Process the lane image to match the display settings
            processed_lane = prepare_display_image(
                lane_img,
                channel=analysis_channel,
                invert=st.session_state.band_params["invert_val"]
            )
            
            rotated_img = rotate_lane_image(processed_lane)
            st.image(rotated_img, use_container_width=True)

def display_lane_data_table(selected_lane, analysis_channel, lane_results):
    """Display data table for a single lane analysis"""
    channel_data = lane_results[analysis_channel]
    peaks = channel_data["peaks"]
    properties = channel_data["properties"]
    
    # Calculate MWs
    if len(peaks) > 0:
        mw_estimates, _, _ = calculate_molecular_weights(peaks, st.session_state.ladder_markers)
        
        # Band data table
        st.subheader(f"Lane {selected_lane} Band Data")
        
        # Get band properties
        heights = properties["peak_heights"]
        
        # Calculate widths
        left_ips = properties.get("left_ips", [])
        right_ips = properties.get("right_ips", [])
        
        if len(left_ips) > 0 and len(right_ips) > 0:
            widths = right_ips - left_ips
        else:
            # Fallback if width detection failed
            widths = np.ones_like(heights) * 10
        
        # Calculate areas
        areas = heights * widths
        
        # Prepare results
        results = {
            "Band": list(range(1, len(peaks) + 1)),
            "Position (px)": peaks,
            "MW (kDa)": [round(mw, 1) for mw in mw_estimates],
            "Height": heights,
            "Width": widths,
            "Area": areas
        }
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by position (top to bottom)
        results_df = results_df.sort_values("Position (px)")
        
        # Display results
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Height": st.column_config.NumberColumn("Height", format="%.2f"),
                "Width": st.column_config.NumberColumn("Width", format="%.2f"),
                "Area": st.column_config.NumberColumn("Area", format="%.2f"),
            }
        )
        
        # Export results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"lane_{selected_lane}_{analysis_channel}_analysis.csv",
            mime="text/csv",
        )
        
        # Bar chart of band intensities
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=[f"{mw} kDa" for mw in results_df["MW (kDa)"]],
            y=results_df["Area"],
            marker_color=THEME_PRIMARY,
            text=results_df["Area"].round(1),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Band Intensities",
            xaxis_title="MW (kDa)",
            yaxis_title="Band Intensity (Area)",
            margin=dict(l=20, r=20, t=50, b=20),
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)

@st.fragment
def lane_comparison_analysis(lane_positions, lane_width, lane_images):
    """Handle lane comparison analysis section"""
    col_content, col_settings = st.columns([8, 4])
    
    with col_settings:
        # Channel selection for analysis
        channel_options = list(st.session_state.band_detection_results[1].keys())
        analysis_channel = st.selectbox(
            "Select channel for analysis", 
            options=channel_options,
            index=channel_options.index(st.session_state.analysis_channel) if st.session_state.analysis_channel in channel_options else 0,
            key="analysis_channel_comparison"
        )
        
        # Store the selected channel in session state
        st.session_state.analysis_channel = analysis_channel
        
        # Get ladder lane from session state for excluding from defaults
        ladder_lane = st.session_state.ladder_lane if 'ladder_lane' in st.session_state else 1
        
        # Select lanes to compare
        selected_lanes = st.multiselect(
            "Select lanes to compare",
            options=list(range(1, len(lane_positions)+1)),
            default=[i for i in range(1, min(len(lane_positions)+1, 4)) if i != ladder_lane],
            key="comparison_lanes"
        )
        
        # Display options
        display_mode = st.radio(
            "Chromatogram display mode:",
            options=["Overlay", "Stacked"],
            horizontal=True,
            key="display_mode"
        )
        
        # Optional separation factor for stacked display
        if display_mode == "Stacked":
            separation_factor = st.slider(
                "Separation between chromatograms", 
                min_value=0.2, 
                max_value=2.0, 
                value=0.5, 
                step=0.1,
                help="Adjust spacing between stacked chromatograms",
                key="separation_factor"
            )
        else:
            separation_factor = 0.5  # Default value
    
    with col_content:
        if selected_lanes:
            display_lane_comparison(selected_lanes, analysis_channel, display_mode, separation_factor, lane_images)
    
    # Create comparison table for MWs
    if len(selected_lanes) > 1:
        create_lane_comparison_table(selected_lanes, analysis_channel)

def display_lane_comparison(selected_lanes, analysis_channel, display_mode, separation_factor, lane_images):
    """Display comparison of multiple lanes with aligned lane images"""
    # Collect data for selected lanes
    lane_data = []
    
    for lane_idx in selected_lanes:
        if lane_idx in st.session_state.band_detection_results:
            lane_results = st.session_state.band_detection_results[lane_idx]
            
            if analysis_channel in lane_results:
                channel_data = lane_results[analysis_channel]
                intensity_profile = channel_data["intensity_raw"]
                peaks = channel_data["peaks"]
                properties = channel_data["properties"]
                
                # Calculate MWs
                if len(peaks) > 0:
                    mw_estimates, _, _ = calculate_molecular_weights(peaks, st.session_state.ladder_markers)
                else:
                    mw_estimates = None
                
                lane_data.append({
                    "lane_idx": lane_idx,
                    "intensity": intensity_profile,
                    "peaks": peaks,
                    "mw_estimates": mw_estimates,
                    "properties": properties if len(peaks) > 0 else None
                })
    
    # Create chromatogram plot and lane images in the same column
    if len(lane_data) > 0:
        if display_mode == "Overlay":
            create_overlay_plot(lane_data, selected_lanes, lane_images)
        else:  # Stacked mode
            create_stacked_plot(lane_data, separation_factor, selected_lanes, lane_images)


def create_overlay_plot(lane_data, selected_lanes, lane_images):
    """Create overlay plot for lane comparison with lane images below"""
    fig = go.Figure()
    
    for data in lane_data:
        lane_idx = data["lane_idx"]
        intensity_profile = data["intensity"]
        peaks = data["peaks"]
        
        # Add line for this lane
        fig.add_trace(go.Scatter(
            x=list(range(len(intensity_profile))),
            y=intensity_profile,
            mode='lines',
            name=f"Lane {lane_idx}",
            opacity=0.7
        ))
        
        # Add peak markers
        if len(peaks) > 0 and data["mw_estimates"] is not None:
            peak_text = [f"{mw:.1f}" for mw in data["mw_estimates"]]
            
            fig.add_trace(go.Scatter(
                x=peaks,
                y=intensity_profile[peaks],
                mode='markers+text',
                marker=dict(size=8, color=MARKER_COLOR),
                name=f"Peaks Lane {lane_idx}",
                text=peak_text,
                textposition="top center",
                showlegend=False
            ))
    
    fig.update_layout(
        title="Overlay Chromatogram Comparison",
        xaxis_title="Vertical Position (pixels)",
        yaxis_title="Intensity",
        margin=dict(l=20, r=20, t=50, b=20),
        template='plotly_dark',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display lane images aligned with the plot
    display_lane_images_aligned(selected_lanes, lane_images)

def create_stacked_plot(lane_data, separation_factor, selected_lanes, lane_images):
    """Create stacked plot for lane comparison with lane images below"""
    fig = go.Figure()
    
    # Find max intensity for scaling
    all_intensities = [np.max(data["intensity"]) for data in lane_data if len(data["intensity"]) > 0]
    max_intensity = max(all_intensities) if all_intensities else 1
    
    # Add each lane with offset for stacking
    for i, data in enumerate(lane_data):
        lane_idx = data["lane_idx"]
        intensity_profile = data["intensity"]
        peaks = data["peaks"]
        
        # Calculate offset for this lane
        offset = i * max_intensity * separation_factor
        
        # Add to figure with offset
        fig.add_trace(go.Scatter(
            x=list(range(len(intensity_profile))),
            y=intensity_profile + offset,  # Add offset for stacking
            mode='lines',
            name=f"Lane {lane_idx}",
            opacity=0.7
        ))
        
        # Add peak markers
        if len(peaks) > 0 and data["mw_estimates"] is not None:
            peak_text = [f"{mw:.1f}" for mw in data["mw_estimates"]]
            
            fig.add_trace(go.Scatter(
                x=peaks,
                y=intensity_profile[peaks] + offset,  # Add same offset
                mode='markers+text',
                marker=dict(size=8, color=MARKER_COLOR),
                name=f"Peaks Lane {lane_idx}",
                text=peak_text,
                textposition="top center",
                showlegend=False
            ))
        
        # Add lane label on the right side
        mid_y = offset + max_intensity / 2
        fig.add_annotation(
            x=len(intensity_profile) * 1.02,  # Just past the right edge
            y=mid_y,
            text=f"Lane {lane_idx}",
            showarrow=False,
            xanchor="left",
            yanchor="middle"
        )
    
    fig.update_layout(
        title="Stacked Chromatogram Comparison",
        xaxis_title="Vertical Position (pixels)",
        yaxis_title="Intensity",
        margin=dict(l=20, r=20, t=50, b=80),  # Extra bottom margin for labels
        template='plotly_dark',
        showlegend=False,  # Hide legend in stacked mode
        xaxis=dict(
            domain=[0, 0.98]  # Leave room for lane labels
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display lane images aligned with the plot
    display_lane_images_aligned(selected_lanes, lane_images)

def display_lane_images_aligned(selected_lanes, lane_images):
    """Display lane images aligned with the peaks"""
    for lane_idx in selected_lanes:
        if 0 <= lane_idx - 1 < len(lane_images):
            lane_img = lane_images[lane_idx - 1]
            
            # Process the lane image to match the display settings
            processed_lane = prepare_display_image(
                lane_img,
                channel=st.session_state.analysis_channel,  # Use the shared analysis channel
                invert=st.session_state.band_params["invert_val"]
            )
            
            rotated_img = rotate_lane_image(processed_lane)
            
            # Create a small container for each lane with a label
            container = st.container()
            col1, col2 = container.columns([2, 10])
            
            with col1:
                st.write(f"Lane {lane_idx}")
            
            with col2:
                st.image(rotated_img, use_container_width=True)

def display_lane_images(selected_lanes, lane_images):
    """Display lane images for selected lanes"""
    col_lane_number, col_lane_images = st.columns([2, 10])
    with col_lane_number:
        for i, lane_idx in enumerate(reversed(selected_lanes)):
            st.write(f"Lane {lane_idx}")
    with col_lane_images:
        for i, lane_idx in enumerate(reversed(selected_lanes)):
            if 0 <= lane_idx - 1 < len(lane_images):
                lane_img = lane_images[lane_idx - 1]
                rotated_img = rotate_lane_image(lane_img)
                st.image(rotated_img, use_container_width=True)

def create_lane_comparison_table(selected_lanes, analysis_channel):
    """Create comparison table for molecular weights across lanes"""
    # Collect data for selected lanes
    lane_data = []
    
    for lane_idx in selected_lanes:
        if lane_idx in st.session_state.band_detection_results:
            lane_results = st.session_state.band_detection_results[lane_idx]
            
            if analysis_channel in lane_results:
                channel_data = lane_results[analysis_channel]
                peaks = channel_data["peaks"]
                properties = channel_data["properties"]
                
                # Calculate MWs
                if len(peaks) > 0:
                    mw_estimates, _, _ = calculate_molecular_weights(peaks, st.session_state.ladder_markers)
                else:
                    mw_estimates = None
                
                lane_data.append({
                    "lane_idx": lane_idx,
                    "peaks": peaks,
                    "mw_estimates": mw_estimates,
                    "properties": properties if len(peaks) > 0 else None
                })
    
    # Collect all MWs
    all_mw = []
    for data in lane_data:
        if data["mw_estimates"] is not None and len(data["mw_estimates"]) > 0:
            all_mw.extend(data["mw_estimates"])
    
    if len(all_mw) > 0:
        # Group similar MWs (within 10%)
        all_mw = sorted(all_mw)
        grouped_mw = []
        current_group = [all_mw[0]]
        
        for mw in all_mw[1:]:
            if mw <= current_group[-1] * 1.1:  # Within 10%
                current_group.append(mw)
            else:
                grouped_mw.append(np.mean(current_group))
                current_group = [mw]
        
        if current_group:
            grouped_mw.append(np.mean(current_group))
        
        # Create comparison table
        comparison_data = {
            "MW (kDa)": [round(mw, 1) for mw in grouped_mw]
        }
        
        # Find band intensity for each lane at each MW
        for data in lane_data:
            lane_idx = data["lane_idx"]
            intensities = []
            
            if (data["mw_estimates"] is not None and 
                len(data["mw_estimates"]) > 0 and 
                data["properties"] is not None):
                
                # Get peak heights
                peak_heights = data["properties"]["peak_heights"]
                
                for target_mw in grouped_mw:
                    # Find closest matching band
                    matches = []
                    for i, mw in enumerate(data["mw_estimates"]):
                        rel_diff = abs(mw - target_mw) / target_mw
                        if rel_diff <= 0.1:  # Within 10%
                            matches.append((i, rel_diff))
                    
                    if matches:
                        # Get best match
                        best_match = min(matches, key=lambda x: x[1])
                        intensity = peak_heights[best_match[0]]
                        intensities.append(intensity)
                    else:
                        intensities.append(0)
            else:
                intensities = [0] * len(grouped_mw)
            
            comparison_data[f"Lane {lane_idx}"] = intensities
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display table
        st.subheader("Band Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Create heatmap and bar chart
        create_comparison_visualizations(comparison_df, selected_lanes, analysis_channel)

def create_comparison_visualizations(comparison_df, selected_lanes, analysis_channel):
    """Create heatmap and bar chart for lane comparison"""
    heatmap_data = []
    heatmap_labels = []
    
    for lane_idx in selected_lanes:
        col_name = f"Lane {lane_idx}"
        if col_name in comparison_df.columns:
            heatmap_data.append(comparison_df[col_name].values)
            heatmap_labels.append(f"Lane {lane_idx}")
    
    if len(heatmap_data) > 0:
        heatmap_array = np.array(heatmap_data)
        
        # Absolute values heatmap
        fig_abs = go.Figure(data=go.Heatmap(
            z=heatmap_array,
            x=[f"{mw} kDa" for mw in comparison_df["MW (kDa)"]],
            y=heatmap_labels,
            colorscale='Blues',
            text=heatmap_array.round(2),
            texttemplate="%{text}",
            colorbar=dict(title="Intensity")
        ))
        
        fig_abs.update_layout(
            title="Band Intensity Heatmap",
            xaxis_title="MW (kDa)",
            yaxis_title="Lane",
            margin=dict(l=20, r=20, t=50, b=20),
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_abs, use_container_width=True)
        
        # Bar chart comparison
        fig_bar = go.Figure()
        
        for i, lane_idx in enumerate(selected_lanes):
            col_name = f"Lane {lane_idx}"
            if col_name in comparison_df.columns:
                fig_bar.add_trace(go.Bar(
                    name=f"Lane {lane_idx}",
                    x=[f"{mw} kDa" for mw in comparison_df["MW (kDa)"]],
                    y=comparison_df[col_name],
                    text=comparison_df[col_name].round(2),
                    textposition='auto'
                ))
        
        fig_bar.update_layout(
            barmode='group',
            title="Band Intensity Comparison",
            xaxis_title="MW (kDa)",
            yaxis_title="Intensity",
            margin=dict(l=20, r=20, t=50, b=20),
            template='plotly_dark',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Export comparison
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="Download Comparison as CSV",
            data=csv,
            file_name=f"lane_comparison_{analysis_channel}.csv",
            mime="text/csv",
        )
    else:
        st.info("Unable to create visualization with the current data.")

def main():
    """Main function with refactored structure using chained fragments"""
    # Initialize the session state variables
    initialize_session_state()
    
    # Section 1: Image Input
    image_input_section()

    # Subsequent sections are called by the previous section
    # Section 2: Define Lanes
    # Section 3: Detect Bands
    # Section 4: Calibrate Ladder
    # Section 5: Analyze Results

if __name__ == '__main__':
    main()