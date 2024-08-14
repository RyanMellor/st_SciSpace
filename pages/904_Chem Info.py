import streamlit as st
from streamlit_ketcher import st_ketcher

import warnings
import os
import re
import requests
import base64
from pathlib import Path
import pandas as pd

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

HAZARD_STATEMENTS = sci_data.file_to_df("./assets/public_data/ghs_h_codes.xlsx")
PRECAUTIONARY_STATEMENTS = sci_data.file_to_df("./assets/public_data/ghs_p_codes.xlsx")
PICTOGRAMS = sci_data.file_to_df("./assets/public_data/ghs_pictograms.xlsx")

re_cas = re.compile(r'\d{2,7}-\d\d-\d')
re_ghs_p_statements = re.compile(r'(P\d{3})')
re_ghs_h_statements = re.compile(r'(H\d{3})')

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
    
def makeblock(smi, optimize=True):
    mol = Chem.MolFromSmiles(smi)
    if optimize:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        AllChem.Compute2DCoords(mol)
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

@st.cache_data
def get_full_json(cid):
    return requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON').json()

@st.cache_data
def parse_full_json(full_record):
    parsed = {
        "identifiers" : {},
		"computed_properties" : {},
		"experimental_properties" : {},
		"safety_and_hazards" : {},
    }
    for section in find_dicts_with_key(full_record, 'TOCHeading'):
        name = section['TOCHeading']
        if name == "First Aid Measures":
            try:
                for risk in section['Information']:
                    parsed["safety_and_hazards"][risk['Name']] = risk['Value']['StringWithMarkup'][0]['String']
            except KeyError:
                pass

    try:
        for record in full_record["Record"]["Section"]:

            if record["TOCHeading"] == "Names and Identifiers":
                for section in record["Section"]:
                    if section["TOCHeading"] == "Computed Descriptors":
                        for subsection in section["Section"]:
                            name = subsection["TOCHeading"]
                            value = subsection["Information"][0]["Value"]["StringWithMarkup"][0]['String']
                            parsed["identifiers"][name] = value

                    elif section["TOCHeading"] == "Molecular Formula":
                        parsed["identifiers"]['Molecular Formula'] = section["Information"][0]["Value"]["StringWithMarkup"][0]["String"]

                    elif section["TOCHeading"] == "Other Identifiers":
                        for subsection in section["Section"]:
                            name = subsection["TOCHeading"]
                            value = subsection["Information"][0]["Value"]["StringWithMarkup"][0]['String']
                            parsed["identifiers"][name] = value

                    elif section["TOCHeading"] == "Synonyms":
                        for subsection in section["Section"]:
                            if subsection["TOCHeading"] == "Depositor-Supplied Synonyms":
                                parsed["identifiers"]['Synonyms'] = []
                                existing_names = [x.lower().replace("-", '') for x in [parsed["identifiers"]['CAS']]]
                                for synonym in subsection["Information"][0]["Value"]["StringWithMarkup"]:
                                    if synonym["String"].lower().replace("-", '') not in existing_names:
                                        parsed["identifiers"]['Synonyms'].append(synonym["String"])
                                        if len(parsed["identifiers"]['Synonyms']) >= 10:
                                            break

            elif record["TOCHeading"] == "Chemical and Physical Properties":
                for section in record["Section"]:

                    if section["TOCHeading"] == "Computed Properties":
                        for subsection in section["Section"]:
                            name = subsection["TOCHeading"]
                            parsed["computed_properties"][name] = {}

                            try:
                                value = subsection["Information"][0]["Value"]["StringWithMarkup"][0]['String']
                                if value == "Yes":
                                    value = True
                                elif value == "No":
                                    value = False
                            except KeyError:
                                value = subsection["Information"][0]["Value"]["Number"][0]
                            parsed["computed_properties"][name]['value'] = value

                            try:
                                unit = subsection["Information"][0]["Value"]["Unit"]
                            except KeyError:
                                unit = None
                            parsed["computed_properties"][name]["unit"] = unit

                    elif section["TOCHeading"] == "Experimental Properties":
                        for subsection in section["Section"]:
                            name = subsection["TOCHeading"]
                            parsed["experimental_properties"][name] = []

                            for source in subsection["Information"]:
                                entry = {}
                                try:
                                    reference = source["Reference"]
                                except KeyError:
                                    reference = None
                                entry["reference"] = reference

                                try:
                                    unit = source["Unit"]
                                except KeyError:
                                    unit = None
                                entry["unit"] = unit

                                try:
                                    value = source["Value"]["StringWithMarkup"][0]['String']
                                    if value == "Yes":
                                        value = True
                                    elif value == "No":
                                        value = False
                                except KeyError:
                                    try:
                                        value = source["Value"]["Number"]
                                        if len(value) == 1:
                                            value = value[0]
                                    except KeyError:
                                        print(source["Value"])
                                        continue
                                entry['value'] = value

                                parsed["experimental_properties"][name].append(entry)

            elif record["TOCHeading"] == "Safety and Hazards":
                for section in record["Section"]:
                    for subsection in section["Section"]:
                        name = subsection["TOCHeading"]
                        if name == "GHS Classification":
                            for classification in subsection["Information"]:
                                if classification['Name'] == 'Pictogram(s)':
                                    if not 'Pictogram(s)' in parsed["safety_and_hazards"].keys():
                                        parsed["safety_and_hazards"]['Pictogram(s)'] = [i['Extra'] for i in classification['Value']['StringWithMarkup'][0]['Markup']]
                                if classification['Name'] == 'Signal':
                                    if not 'Signal' in parsed["safety_and_hazards"].keys():
                                        parsed["safety_and_hazards"]['Signal'] = classification['Value']['StringWithMarkup'][0]['String']
                                if classification['Name'] == 'GHS Hazard Statements':
                                    if not 'GHS Hazard Statements' in parsed["safety_and_hazards"].keys():
                                        string = ''.join(i['String'] for i in classification['Value']['StringWithMarkup'])
                                        parsed["safety_and_hazards"]['GHS Hazard Statements'] = re_ghs_h_statements.findall(string)
                                if classification['Name'] == 'Precautionary Statement Codes':
                                    if not 'Precautionary Statement Codes' in parsed["safety_and_hazards"].keys():
                                        string = classification['Value']['StringWithMarkup'][0]['String']
                                        parsed["safety_and_hazards"]['Precautionary Statement Codes'] = re_ghs_p_statements.findall(string)
    except KeyError:
        pass

    return parsed

def find_dicts_with_key(node, k):
	if isinstance(node, list):
		for i in node:
			for x in find_dicts_with_key(i, k):
				yield x
	elif isinstance(node, dict):
		if k in node:
			yield node
		for v in node.values():
			for x in find_dicts_with_key(v, k):
				yield x

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def img_to_html(img_path):
    img_bytes = img_to_bytes(img_path)
    img_html = f"<img src='data:image/png;base64,{img_bytes}' class='img-fluid' width=50px>"
    return img_html

def main():
    input_type = st.selectbox("Input Type", ["Name", "Draw Structure", "SMILES", "MOL"])
    
    if input_type == "Name":
        compound_name = st.text_input("Name", value='Caffeine', label_visibility='collapsed')
        try:
            compound_pc = pubchem_from_name(compound_name)
            if compound_pc.cid == None:
                st.error(f"Compound '{compound_name}' not found in PubChem")
                return None
            compound_smiles = compound_pc.canonical_smiles
            mol = Chem.MolFromSmiles(compound_smiles)
        except ValueError as e:
            st.error(f"{e}")
            return None
    
    elif input_type == "Draw Structure":
        compound_smiles = st_ketcher(value='CN1C=NC2=C1C(=O)N(C(=O)N2C)C', molecule_format='SMILES', height=650)
        mol = Chem.MolFromSmiles(compound_smiles)
        compound_pc = pubchem_from_smiles(compound_smiles)
        if compound_pc.cid == None:
            st.error(f"Compound with SMILES '{compound_smiles}' not found in PubChem")
            return None
    
    elif input_type == "SMILES":
        compound_smiles = st.text_input(
            'SMILES', value='CN1C=NC2=C1C(=O)N(C(=O)N2C)C', label_visibility='collapsed')
        mol = Chem.MolFromSmiles(compound_smiles)
        compound_smiles = Chem.MolToSmiles(mol)
        compound_pc = pubchem_from_smiles(compound_smiles)
        if compound_pc.cid == None:
            st.error(f"Compound with SMILES '{compound_smiles}' not found in PubChem")
            return None

    elif input_type == "MOL":
        molfile = st.file_uploader("Upload MOL File", type=['mol'])
        if molfile == None:
            molfile = r".\assets\public_data\Caffeine.mol"
            mol = Chem.rdmolfiles.MolFromMolFile(molfile)
        else:
            mol = Chem.rdmolfiles.MolFromMolBlock(molfile.getvalue().decode("utf-8"))
        compound_smiles = Chem.MolToSmiles(mol)
        compound_pc = pubchem_from_smiles(compound_smiles)
        if compound_pc.cid == None:
            st.error(f"Compound with SMILES '{compound_smiles}' not found in PubChem")
            return None
    
    st.markdown('---')

    cas_re = re.compile(r"^\d{2,7}-\d{2}-\d$")

    full_record = get_full_json(compound_pc.cid)
    parsed = parse_full_json(full_record)
    common_name = full_record['Record']['RecordTitle']

    st.markdown("### Compound Identifiers")
    identifiers = {
        'Common Name': common_name,
        'IUPAC Name': compound_pc.iupac_name,
        'Formula': compound_pc.molecular_formula,
        'Canonical SMILES': compound_pc.canonical_smiles,
        'InChI': compound_pc.inchi,
        'InChIKey': compound_pc.inchikey,
        'PubChem CID': str(compound_pc.cid),
        'CAS': parsed['identifiers'].get('CAS', 'N/A'),
        'Synonyms': ',\n'.join(compound_pc.synonyms) if compound_pc.synonyms else 'N/A',
    }
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
        st.dataframe(properties, use_container_width=True, height=40*len(properties))
        
    with st.container():
        with col_2:
            tab_2d, tab_3d = st.tabs(['2D', '3D'])
            with tab_2d:
                options = Draw.MolDrawOptions()
                options.setAtomPalette(ELEMENT_COLORS_RGB)
                options.clearBackground = False
                im = Draw.MolToImage(mol, size=(300, 300), options=options)
                st.image(im)

            with tab_3d:
                optimize_3d = st.checkbox("Optimize 3D Structure", value=True)
                blk = makeblock(compound_smiles, optimize_3d)
                render_mol(blk)

    st.markdown("### Experimental Properties")
    with st.container(height=600):
        for key, value in parsed['experimental_properties'].items():
            st.markdown(f"##### {key}")
            for entry in value:
                st.caption(entry["value"])
                
    st.markdown("### Safety and Hazards")
    with st.container(height=600):

        st.markdown("##### Signal Word and Pictograms")
        signal_word = parsed['safety_and_hazards'].get('Signal', '')
        if signal_word == '':
            st.caption("No signal word found")
        else:
            color = 'red' if signal_word == 'Danger' else 'orange'
            st.markdown(f":{color}[{signal_word}]", unsafe_allow_html=True)
        pictograms = parsed['safety_and_hazards'].get('Pictogram(s)', [])
        if len(pictograms) == 0:
            st.caption("No pictograms found")
        else:
            pictogram_cols = st.columns(len(pictograms))
            for i, pictogram in enumerate(pictograms):
                row = PICTOGRAMS[PICTOGRAMS['description'] == pictogram]
                with pictogram_cols[i]:
                    st.caption(row['code'].values[0])
                    st.image(row['image'].values[0], width=50)
                    st.caption(row['description'].values[0])

        st.markdown("---")
        st.markdown("##### GHS Hazard Statements")
        hazard_statements = parsed['safety_and_hazards'].get('GHS Hazard Statements', [])
        if len(hazard_statements) == 0:
            st.caption("No hazard statements found")
        else:
            hazard_markdown = ""
            for code in parsed['safety_and_hazards'].get('GHS Hazard Statements', []):
                row = HAZARD_STATEMENTS[HAZARD_STATEMENTS['h_code'] == code]
                statement = row['hazard_statements'].values[0]
                hazard_markdown += f"<abbr title='{statement}'>{code}</abbr> | "
            st.markdown(hazard_markdown[:-2], unsafe_allow_html=True)
        with st.popover("H-Codes"):
            st.dataframe(HAZARD_STATEMENTS, hide_index=True)

        # st.markdown("##### Precautionary Statement Codes")
        # precaution_markdown = ""
        # for code in parsed['safety_and_hazards'].get('Precautionary Statement Codes', []):
        #     row = PRECAUTIONARY_STATEMENTS[PRECAUTIONARY_STATEMENTS['p_code'] == code]
        #     statement = row['statement'].values[0]
        #     precaution_markdown += f"<abbr title='{statement}'>{code}</abbr> | "
        # st.markdown(precaution_markdown[:-2], unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("##### First Aid")  
        first_aid_found = False     
        for key, value in parsed['safety_and_hazards'].items():
            if key in ['Inhalation First Aid','Skin First Aid','Eye First Aid','Ingestion First Aid', 'First Aid Measures']:
                first_aid_found = True
                st.markdown(f"{key}")
                st.caption(value)
        if not first_aid_found:
            st.caption("No first aid information found")




if __name__ == '__main__':
    main()