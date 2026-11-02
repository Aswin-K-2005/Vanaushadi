"""Shared constants for the EuroSAT land-use project.

Everything that more than one script needs lives here, so train / evaluate /
app / change detection can never silently disagree about class order, image
size, or where outputs go.
"""

import os

# Alphabetical, which is also the order of the class folders in both the RGB
# and the MS zip. The trained model's output index i == CLASS_NAMES[i].
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
    "SeaLake",
]

# Human-friendly labels for figures and the web UI.
PRETTY_NAMES = {
    "AnnualCrop": "Annual Crop",
    "Forest": "Forest",
    "HerbaceousVegetation": "Herbaceous Veg.",
    "Highway": "Highway",
    "Industrial": "Industrial",
    "Pasture": "Pasture",
    "PermanentCrop": "Permanent Crop",
    "Residential": "Residential",
    "River": "River",
    "SeaLake": "Sea / Lake",
}

# Fixed colours per class, used for land-use maps in change detection.
CLASS_COLORS = {
    "AnnualCrop": (230, 196, 72),
    "Forest": (34, 110, 52),
    "HerbaceousVegetation": (140, 190, 90),
    "Highway": (120, 120, 120),
    "Industrial": (170, 60, 160),
    "Pasture": (190, 225, 130),
    "PermanentCrop": (200, 140, 50),
    "Residential": (220, 70, 60),
    "River": (60, 140, 220),
    "SeaLake": (20, 70, 160),
}

# EuroSAT patches are 64x64 px at 10 m/px = 640 m x 640 m on the ground.
PATCH_PX = 64
METRES_PER_PX = 10
PATCH_AREA_HA = (PATCH_PX * METRES_PER_PX) ** 2 / 10_000  # 40.96 ha

SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test, stratified per class

# Sentinel-2 band order inside the EuroSAT_MS GeoTIFFs (13 bands).
MS_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]

# 3-channel composites we can feed to an ImageNet-pretrained backbone.
#   rgb            : the official EuroSAT_RGB JPEGs (default, what the app uses)
#   ms_rgb         : true colour rebuilt from the MS GeoTIFFs (B4, B3, B2)
#   ms_falsecolor  : NIR, Red, Green (B8, B4, B3) -- vegetation shows up bright red
#   ms_swir        : SWIR2, NIR, Red (B12, B8, B4) -- separates moisture / bare soil
# Comparing ms_rgb against ms_falsecolor / ms_swir with the SAME pipeline is the
# spectral experiment: does the infrared information the RGB release throws
# away fix the confusions the RGB model makes?
COMPOSITES = {
    "rgb": None,
    "ms_rgb": ("B04", "B03", "B02"),
    "ms_falsecolor": ("B08", "B04", "B03"),
    "ms_swir": ("B12", "B08", "B04"),
}

# Official download locations (the Zenodo record is the authors' current mirror).
RGB_URLS = [
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1",
    "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
]
MS_URLS = [
    "https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1",
    "https://madm.dfki.de/files/sentinel/EuroSATallBands.zip",
]

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("EUROSAT_DATA", os.path.join(HERE, "data"))
OUTPUT_ROOT = os.environ.get("EUROSAT_OUTPUT", os.path.join(HERE, "outputs"))


def run_dir(composite, backbone):
    """Each (composite, backbone) experiment gets its own output folder."""
    return os.path.join(OUTPUT_ROOT, f"{composite}_{backbone}")
