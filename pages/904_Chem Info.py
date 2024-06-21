import streamlit as st

import io
import cv2
import sys
import json
import numpy as np
import base64
from datetime import date
from PIL import Image
import warnings
import pandas as pd
import os
import argparse

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, Descriptors
import py3Dmol
from stmol import showmol
import re
import requests

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from helpers import sci_setup, sci_data
from helpers.sci_style import *
sci_setup.setup_page("Chem Info")

@st.cache_data
def cactus_search(query, search_type):
    url = f"https://cactus.nci.nih.gov/chemical/structure/{query}/{search_type}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise ValueError(f"Identifier not found in database. {response.text}")
    
def validate_cas(cas):
    # validate cas number using regex
    cas_regex = r"^\d{2,7}-\d{2}-\d$"
    if not re.match(cas_regex, cas):
        raise ValueError("Invalid CAS Number")
    else:
        return True
    
def makeblock(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, useRandomCoords=True)
    AllChem.MMFFOptimizeMolecule(mol)
    mblock = Chem.MolToMolBlock(mol)
    return mblock

def render_mol(xyz):
    xyzview = py3Dmol.view(width=300, height=300)
    xyzview.addModel(xyz,'mol')
    xyzview.setBackgroundColor('#2b2b2b')
    for elm in ELEMENT_COLORS_HEX:
        xyzview.addStyle({},{'stick':{'colorscheme':f'Jmol', 'color':f'{elm}'}})
    xyzview.zoomTo()
    showmol(xyzview, height=300,width=300)

@st.cache_data
def molprop_calc(mol):
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'MolLogP': round(Descriptors.MolLogP(mol), 2),
        'MolMR': round(Descriptors.MolMR(mol), 2),
        'TPSA': round(Descriptors.TPSA(mol), 2 ),
        'NumHDonors': Descriptors.NumHDonors(mol),
        'NumHAcceptors': Descriptors.NumHAcceptors(mol),
        'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'Charge': Chem.GetFormalCharge(mol),
    }

def main():
    # mol_ids_dict = {
    #     'names': [],
    #     'cas': '',
    #     'smiles': '',
    #     'formula': '',
    #     'inchi': '',
    #     'inchikey': '',	
    #     'pubchem_cid': '',
    # }
    # compound_smiles = st_keyup('SMILES', value='c1cc(C(=O)O)c(OC(=O)C)cc1', label_visibility='collapsed')

    # with st.expander("Cactus - Chemical Identifier Resolver"):
    #     compound_id = st.text_input("Identifier", value='58-08-2', label_visibility='collapsed')
    #     compound_smiles = cactus_search(compound_id, 'smiles')
    #     st.write(f"SMILES:\n\n{compound_smiles}")

    input_type = st.selectbox("Input Type", ["Identifier", "SMILES", "MOL"])
    
    if input_type == "Identifier":
        compound_id = st.text_input("Identifier", value='58-08-2', label_visibility='collapsed')
        try:
            compound_smiles = cactus_search(compound_id, 'smiles')
        except ValueError as e:
            st.error(f"{e}")
            return None
        mol = Chem.MolFromSmiles(compound_smiles)
    
    elif input_type == "SMILES":
        compound_smiles = st.text_input(
            'SMILES', value='Cn1cnc2n(C)c(=O)n(C)c(=O)c12', label_visibility='collapsed')
        mol = Chem.MolFromSmiles(compound_smiles)

    elif input_type == "MOL":
        molfile = st.file_uploader("Upload MOL File", type=['mol'])
        if molfile == None:
            molfile = r".\assets\public_data\Caffeine.mol"
            mol = Chem.rdmolfiles.MolFromMolFile(molfile)
        else:
            mol = Chem.rdmolfiles.MolFromMolBlock(molfile.getvalue().decode("utf-8"))
        compound_smiles = Chem.MolToSmiles(mol)
    
    st.markdown('---')

    col_1, col_2 = st.columns([1,2])
    with col_1:
        st.markdown("#####")
        prop = molprop_calc(mol)
        st.dataframe(prop)
        
    with col_2:
        tab_2d, tab_3d = st.tabs(['2D', '3D'])
        with tab_2d:
            options = Draw.MolDrawOptions()
            options.setAtomPalette(ELEMENT_COLORS_RGB)
            options.clearBackground = False
            im = Draw.MolToImage(mol, size=(300, 300), options=options)
            st.image(im)

            # image = st.empty()
            # i = 0
            # while True:
            #     options.rotate = i
            #     im = Draw.MolToImage(mol, size=(300, 300), options=options)
            #     image.image(im)
            #     i += 0.1
            #     if i > 360:
            #         i = 0

            blk = makeblock(compound_smiles)
        with tab_3d:
            render_mol(blk)
            

if __name__ == '__main__':
    main()