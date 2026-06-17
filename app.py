import os
import tempfile

import streamlit as st
import tensorflow as tf
import rasterio

from LLM_Report import (
    run_environmental_analysis
)

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Land Cover Analysis System",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# Load Models Once
# ==========================================

@st.cache_resource
def load_models():

    rgb_model = tf.keras.models.load_model(
        "best_rgb_model.keras"
    )

    ndvi_model = tf.keras.models.load_model(
        "best_ndvi_model.keras"
    )

    rgb_nir_model = tf.keras.models.load_model(
        "best_rgb_nir_model.keras"
    )

    return (
        rgb_model,
        ndvi_model,
        rgb_nir_model
    )

rgb_model, ndvi_model, rgb_nir_model = load_models()

# ==========================================
# Class Names
# ==========================================

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# ==========================================
# Header
# ==========================================

st.title(
    "🌍 Intelligent Land Cover Analysis System"
)

st.markdown("""
This system combines:

- RGB CNN
- NDVI CNN
- RGB + NIR CNN
- LLM-based Environmental Reporting

Upload a Sentinel-2 TIFF image to begin.
""")

# ==========================================
# Upload TIFF
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Sentinel-2 TIFF Image",
    type=["tif", "tiff"]
)

# ==========================================
# Run Analysis
# ==========================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".tif"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.read()
        )

        image_path = tmp_file.name

    try:

        with rasterio.open(
            image_path
        ) as src:

            band_count = src.count

        if band_count < 8:

            st.error(
                f"Invalid image. Found only {band_count} bands."
            )

            st.stop()

    except Exception as e:

        st.error(
            f"Image Error: {str(e)}"
        )

        st.stop()

    # ======================================
    # Start Analysis
    # ======================================

    if st.button(
        "Run Analysis"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            output = run_environmental_analysis(

                image_path=image_path,

                rgb_model=rgb_model,

                ndvi_model=ndvi_model,

                rgb_nir_model=rgb_nir_model,

                class_names=CLASS_NAMES
            )

        results = output["results"]

        report = output["report"]

        # ==================================
        # Model Predictions
        # ==================================

        st.header(
            "Model Predictions"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(

                label="RGB Model",

                value=results[
                    "rgb_predictions"
                ][0]["class"],

                delta=f"{results['rgb_predictions'][0]['confidence']:.2f}%"
            )

        with col2:

            st.metric(

                label="NDVI Model",

                value=results[
                    "ndvi_predictions"
                ][0]["class"],

                delta=f"{results['ndvi_predictions'][0]['confidence']:.2f}%"
            )

        with col3:

            st.metric(

                label="RGB+NIR Model",

                value=results[
                    "rgbnir_predictions"
                ][0]["class"],

                delta=f"{results['rgbnir_predictions'][0]['confidence']:.2f}%"
            )

        # ==================================
        # NDVI Statistics
        # ==================================

        st.header(
            "NDVI Statistics"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean NDVI",
            f"{results['mean_ndvi']:.3f}"
        )

        c2.metric(
            "Max NDVI",
            f"{results['max_ndvi']:.3f}"
        )

        c3.metric(
            "Min NDVI",
            f"{results['min_ndvi']:.3f}"
        )

        # ==================================
        # Top Predictions
        # ==================================

        st.header(
            "Top Predictions"
        )

        tab1, tab2, tab3 = st.tabs(
            [
                "RGB",
                "NDVI",
                "RGB+NIR"
            ]
        )

        with tab1:

            st.json(
                results[
                    "rgb_predictions"
                ]
            )

        with tab2:

            st.json(
                results[
                    "ndvi_predictions"
                ]
            )

        with tab3:

            st.json(
                results[
                    "rgbnir_predictions"
                ]
            )

        # ==================================
        # Environmental Report
        # ==================================

        st.header(
            "Environmental Report"
        )

        st.markdown(
            report
        )

        # ==================================
        # Download Report
        # ==================================

        st.download_button(

            label="Download Report",

            data=report,

            file_name="environmental_report.txt",

            mime="text/plain"
        )

    try:
        os.remove(image_path)
    except:
        pass