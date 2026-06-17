import os
import numpy as np
import rasterio
from groq import Groq

# Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

# TIFF Validation
def validate_tiff(image_path):
    try:
        with rasterio.open(image_path) as src:
            bands = src.count
        if bands < 8:
            return (
                False,
                f"Invalid image. Found only {bands} bands."
            )
        return (
            True,
            "Valid Sentinel-2 image."
        )
    except Exception as e:
        return (
            False,
            str(e)
        )

# RGB Loader
def load_rgb(image_path):
    with rasterio.open(image_path) as src:
        rgb = src.read(
            [4, 3, 2]
        ).astype(np.float32)
    rgb = np.transpose(
        rgb,
        (1, 2, 0)
    )
    rgb = (
        rgb - rgb.min()
    ) / (
        rgb.max() - rgb.min() + 1e-8
    )

    return rgb

# RGB + NIR Loader
def load_rgb_nir(image_path):

    with rasterio.open(image_path) as src:
        img = src.read(
            [4, 3, 2, 8]
        ).astype(np.float32)

    img = np.transpose(
        img,
        (1, 2, 0)
    )

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-8
    )

    return img

# NDVI Calculation
def compute_ndvi_from_tif(image_path):

    with rasterio.open(image_path) as src:

        red = src.read(4).astype(
            np.float32
        )

        nir = src.read(8).astype(
            np.float32
        )

    ndvi = (
        nir - red
    ) / (
        nir + red + 1e-8
    )

    return ndvi

# Top Predictions

def get_top_predictions(
    model,
    image,
    class_names,
    top_k=3
):
    probs = model.predict(

        np.expand_dims(
            image,
            axis=0
        ),
        verbose=0
    )[0]
    indices = np.argsort(
        probs
    )[::-1][:top_k]
    predictions = []
    for idx in indices:
        predictions.append({
            "class":
            class_names[idx],

            "confidence":
            round(
                float(
                    probs[idx] * 100
                ),
                2
            )
        })
    return predictions

# Image Analysis
def analyze_image(
    image_path,
    rgb_model,
    ndvi_model,
    rgb_nir_model,
    class_names
):
    rgb = load_rgb(
        image_path
    )
    rgb_nir = load_rgb_nir(
        image_path
    )
    ndvi = compute_ndvi_from_tif(
        image_path
    )
    ndvi_input = np.expand_dims(
        ndvi,
        axis=-1
    )
    rgb_predictions = get_top_predictions(
        rgb_model,
        rgb,
        class_names
    )
    ndvi_predictions = get_top_predictions(
        ndvi_model,
        ndvi_input,
        class_names
    )
    rgbnir_predictions = get_top_predictions(
        rgb_nir_model,
        rgb_nir,
        class_names
    )
    return {

        "rgb_predictions":
        rgb_predictions,

        "ndvi_predictions":
        ndvi_predictions,

        "rgbnir_predictions":
        rgbnir_predictions,

        "mean_ndvi":
        float(np.mean(ndvi)),

        "max_ndvi":
        float(np.max(ndvi)),

        "min_ndvi":
        float(np.min(ndvi))
    }

# Prompt Builder
def build_prompt(results):
    return f"""
You are a Remote Sensing and Environmental Monitoring expert.

Analyze the following outputs from three CNN models.

RGB Model Predictions:
{results['rgb_predictions']}

NDVI Model Predictions:
{results['ndvi_predictions']}

RGB+NIR Model Predictions:
{results['rgbnir_predictions']}

NDVI Statistics:

Mean NDVI: {results['mean_ndvi']:.3f}
Max NDVI : {results['max_ndvi']:.3f}
Min NDVI : {results['min_ndvi']:.3f}

Tasks:

1. Compare all model predictions.
2. Identify agreement/disagreement.
3. Determine the most reliable class.
4. Assess vegetation condition.
5. Explain environmental implications.
6. Identify possible risks.
7. Provide actionable recommendations.

Generate a professional report.

Use analytical reasoning instead of repeating predictions.
"""

# Generate Report
def generate_report(results):
    prompt = build_prompt(
        results
    )
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        messages=[
            {
                "role": "system",

                "content":
                """
                You are an expert in:

                - Remote Sensing
                - GIS
                - Earth Observation
                - Environmental Monitoring
                - Land Cover Analysis

                Analyze model outputs critically.

                Explain confidence.

                Provide environmental interpretation.

                Provide recommendations.
                """
            },

            {
                "role": "user",

                "content": prompt
            }

        ]
    )

    return response.choices[
        0
    ].message.content

# Save Report
def save_report(
    report,
    filename="environmental_report.txt"
):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(report)

# Full Pipeline
def run_environmental_analysis(
    image_path,
    rgb_model,
    ndvi_model,
    rgb_nir_model,
    class_names
):
    is_valid, message = validate_tiff(
        image_path
    )
    if not is_valid:

        raise ValueError(
            message
        )

    results = analyze_image(
        image_path,
        rgb_model,
        ndvi_model,
        rgb_nir_model,
        class_names
    )
    report = generate_report(
        results
    )
    return {
        "results": results,
        "report": report
    }