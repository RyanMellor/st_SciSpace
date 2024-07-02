import streamlit as st

import warnings
import os
import re
import requests

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, Descriptors
import py3Dmol
from stmol import showmol
import pubchempy as pcp

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
    xyzview.setStyle({'stick':{'colorscheme':'Jmol'}})
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

@st.cache_data
def pubchem_from_smiles(compound_smiles):
    compound_pc = pcp.get_compounds(compound_smiles, 'smiles')[0]
    return compound_pc

@st.cache_data
def pubchem_from_name(compound_name):
    compound_pc = pcp.get_compounds(compound_name, 'name')[0]
    return compound_pc

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

    input_type = st.selectbox("Input Type", ["Name", "SMILES", "MOL"])
    
    if input_type == "Name":
        compound_name = st.text_input("Name", value='Caffeine', label_visibility='collapsed')
        try:
            compound_pc = pubchem_from_name(compound_name)
            compound_smiles = compound_pc.canonical_smiles
            mol = Chem.MolFromSmiles(compound_smiles)
        except ValueError as e:
            st.error(f"{e}")
            return None
    
    elif input_type == "SMILES":
        compound_smiles = st.text_input(
            'SMILES', value='CN1C=NC2=C1C(=O)N(C(=O)N2C)C', label_visibility='collapsed')
        mol = Chem.MolFromSmiles(compound_smiles)

    # elif input_type == "Identifier":
    #     compound_id = st.text_input("Identifier", value='58-08-2', label_visibility='collapsed')
    #     try:
    #         compound_smiles = cactus_search(compound_id, 'smiles')
    #     except ValueError as e:
    #         st.error(f"{e}")
    #         return None
    #     mol = Chem.MolFromSmiles(compound_smiles)

    elif input_type == "MOL":
        molfile = st.file_uploader("Upload MOL File", type=['mol'])
        if molfile == None:
            molfile = r".\assets\public_data\Caffeine.mol"
            mol = Chem.rdmolfiles.MolFromMolFile(molfile)
        else:
            mol = Chem.rdmolfiles.MolFromMolBlock(molfile.getvalue().decode("utf-8"))
    
    compound_smiles = Chem.MolToSmiles(mol)
    compound_pc = pubchem_from_smiles(compound_smiles)
    
    st.markdown('---')

    cas_re = re.compile(r"^\d{2,7}-\d{2}-\d$")

    st.markdown("### Compound Identifiers")
    identifiers = {
        # 'Common Name': compound_pc.synonyms[0],
        'IUPAC Name': compound_pc.iupac_name,
        'Formula': compound_pc.molecular_formula,
        'Canonical SMILES': compound_pc.canonical_smiles,
        'InChI': compound_pc.inchi,
        'InChIKey': compound_pc.inchikey,
        'PubChem CID': compound_pc.cid,
        'CAS': "",
        'Synonyms': ',\n'.join(compound_pc.synonyms) if compound_pc.synonyms else 'N/A',
    }
    for syn in compound_pc.synonyms:
        if cas_re.match(syn):
            identifiers['CAS'] = syn
            break

    st.dataframe(identifiers, use_container_width=True)

    properties = {
        'Molecular Weight': compound_pc.molecular_weight,
        'XLogP': compound_pc.xlogp,
        'H-Bond Donors': compound_pc.h_bond_donor_count,
        'H-Bond Acceptors': compound_pc.h_bond_acceptor_count,
        'Rotatable Bonds': compound_pc.rotatable_bond_count,
        'Exact Mass': compound_pc.exact_mass,
        'Monoisotopic Mass': compound_pc.monoisotopic_mass,
        'Topological Polar Surface Area': compound_pc.tpsa,
        'Heavy Atom Count': compound_pc.heavy_atom_count,
        'Charge': compound_pc.charge,
        'Complexity': compound_pc.complexity,
    }
    for key, value in properties.items():
        try:
            properties[key] = round(float(value), 2)
        except TypeError:
            pass    

    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown("### Calculated Properties")
        # prop = molprop_calc(mol)
        # st.dataframe(prop)
        st.dataframe(properties, use_container_width=True)
        
    with col_2:
        tab_2d, tab_3d = st.tabs(['2D', '3D'])
        with tab_2d:
            options = Draw.MolDrawOptions()
            options.setAtomPalette(ELEMENT_COLORS_RGB)
            options.clearBackground = False
            im = Draw.MolToImage(mol, size=(300, 300), options=options)
            st.image(im)

            blk = makeblock(compound_smiles)
        with tab_3d:
            render_mol(blk)

    # for key, value in compound_pc.to_dict().items():
    #     if value:
    #         st.write(f"{key}: {value}")
            
if __name__ == '__main__':
    main()