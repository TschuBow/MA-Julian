import audio_recorder
import audio_cropper
import audio_plotter
import nozzle_controller
import audio_feature_generator
import audio_feature_regressor
import streamlit as st

import own_functions


st.set_page_config(layout="wide")

PAGES = {
    "Recorder": audio_recorder,
    "Cropper": audio_cropper,
    "Feature Generator": audio_feature_generator,
    "Feature Regressor": audio_feature_regressor,
    "Plotter": audio_plotter,
    "Nozzle Controller": nozzle_controller,
}
st.sidebar.image(
    "images/fau.png"
)
st.sidebar.image(
    "images/reps.png"
)
# st.sidebar.title("MASS")
st.sidebar.write(
    '<span style="font-size: 3rem; font-weight: 650;">MASS</span>',
    unsafe_allow_html=True,
)

selection = st.sidebar.radio("Measuring with Air Stream and Sound", list(PAGES.keys()))
page = PAGES[selection]

page.main()


own_functions.footer()