
import pandas as pd

PICTOGRAMS = {
    "GHS01": {"name": "Explosive", "image": "./assets/images/GHS01.png"},
    "GHS02": {"name": "Flammable", "image": "./assets/images/GHS02.png"},
    "GHS03": {"name": "Oxidizer", "image": "./assets/images/GHS03.png"},
    "GHS04": {"name": "Compressed Gas", "image": "./assets/images/GHS04.png"},
    "GHS05": {"name": "Corrosive", "image": "./assets/images/GHS05.png"},
    "GHS06": {"name": "Acute Toxic", "image": "./assets/images/GHS06.png"},
    "GHS07": {"name": "Irritant", "image": "./assets/images/GHS07.png"},
    "GHS08": {"name": "Health Hazard", "image": "./assets/images/GHS08.png"},
    "GHS09": {"name": "Environment", "image": "./assets/images/GHS09.png"},
}

HAZARD_STATEMENTS = pd.read_excel("assets/public_data/ghs_h_codes.xlsx")
PRECAUTIONARY_STATEMENTS = pd.read_excel("assets/public_data/ghs_p_codes.xlsx")