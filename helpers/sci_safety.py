
import pandas as pd

PICTOGRAMS = {
    "GHS01": {"name": "Explosive", "image": "assets/GHS01.png"},
    "GHS02": {"name": "Flammable", "image": "assets/GHS02.png"},
    "GHS03": {"name": "Oxidizer", "image": "assets/GHS03.png"},
    "GHS04": {"name": "Compressed Gas", "image": "assets/GHS04.png"},
    "GHS05": {"name": "Corrosive", "image": "assets/GHS05.png"},
    "GHS06": {"name": "Acute Toxic", "image": "assets/GHS06.png"},
    "GHS07": {"name": "Irritant", "image": "assets/GHS07.png"},
    "GHS08": {"name": "Health Hazard", "image": "assets/GHS08.png"},
    "GHS09": {"name": "Environment", "image": "assets/GHS09.png"},
}

HAZARD_STATEMENTS = pd.read_excel("assets\public_data\ghs_h_codes.xlsx")
PRECAUTIONARY_STATEMENTS = pd.read_excel("assets\public_data\ghs_p_codes.xlsx")