import streamlit as st
import mols2grid

import numpy as np
import sys

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Descriptors3D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprintAsBitVect,
    GetHashedMorganFingerprint,
    GetHashedAtomPairFingerprint,
    GetHashedTopologicalTorsionFingerprint
)
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

import mordred
from mhfp.encoder import MHFPEncoder
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.fp import FPVecTransformer
from descriptastorus.descriptors import rdNormalizedDescriptors, rdDescriptors

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score, precision_recall_curve, average_precision_score
from lightning import pytorch as pl
import chemprop as cp
import deepchem as dc

from catboost import Pool, CatBoostRegressor

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

from helpers.sci_style import *

CALCULATORS = {}

@st.cache_data
def generate_fps(smiles, mol_fp, radius=2, n_bits=1024):
    """Generate fixed molecular fpresentations 
    Inputs:
        smiles: a molecule in SMILES fpresentation 
        mol_fp: fp name - options include morgan_bits, morgan_counts, maccs, atom_pairs, avalon, erg, rdkit2d, physchem

    Reference:
        https://github.com/dengjianyuan/Respite_MPP/blob/9f3df9e2af747091edb6d60bb06b56294ce24dc4/src/dataset.py#L46
    """
    if isinstance(smiles, Chem.rdchem.Mol):
        mol = smiles
    else:
        mol = Chem.MolFromSmiles(smiles)

    # if mol_fp is list of smiles then recursively call the function and return concatenated features
    if isinstance(mol_fp, list):
        feats = np.concatenate([generate_fps(smiles, fp, radius, n_bits) for fp in mol_fp])
        return feats
    
    try:
        if mol_fp == 'morgan_bits':
            features_vec = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'morgan_counts':
            features_vec = GetHashedMorganFingerprint(mol, radius=radius, nBits=n_bits)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'maccs':
            features_vec = MACCSkeys.GenMACCSKeys(mol)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'atom_pairs':
            features_vec = GetHashedAtomPairFingerprint(mol, nBits=n_bits, use2D=True)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'avalon':
            features_vec = GetAvalonCountFP(mol, nBits=n_bits)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'erg':
            feats = GetErGFingerprint(mol)
        elif mol_fp == 'topological_torsion':
            features_vec = GetHashedTopologicalTorsionFingerprint(mol, nBits=n_bits)
            feats = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, feats)
        elif mol_fp == 'rdkit2d_histnorm':
            if not 'rdkit2d_histnorm' in CALCULATORS.keys():
                CALCULATORS['rdkit2d_histnorm'] = rdNormalizedDescriptors.RDKit2DHistogramNormalized()
            feats = CALCULATORS['rdkit2d_histnorm'].process(smiles)[1:]
            feats = np.array(feats, dtype=np.float32)
        elif mol_fp == 'rdkit2d_norm':
            if not 'rdkit2d_norm' in CALCULATORS.keys():
                CALCULATORS['rdkit2d_norm'] = rdNormalizedDescriptors.RDKit2DNormalized()
            feats = CALCULATORS['rdkit2d_norm'].process(smiles)[1:]
            feats = np.array(feats, dtype=np.float32)
        elif mol_fp == 'rdkit2d':
            if not 'rdkit2d' in CALCULATORS.keys():
                chosen_descriptors = [n for n,f in Descriptors.descList]
                CALCULATORS['rdkit2d'] = MolecularDescriptorCalculator(chosen_descriptors)
            feats = CALCULATORS['rdkit2d'].CalcDescriptors(mol)
            # feats = [v for k, v in feats.items()]
            feats = np.array(feats, dtype=np.float32)
        elif mol_fp == 'rdkit3d':
            if not 'rdkit3d' in CALCULATORS.keys():
                CALCULATORS['rdkit3d'] = FPVecTransformer(kind='desc3D', dtype=float)
            feats = CALCULATORS['rdkit3d'](mol)[0]
            feats = np.array(feats, dtype=np.float32)
        elif mol_fp == 'physchem':
            feats = np.array([
                Descriptors.ExactMolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Chem.rdchem.Mol.GetNumAtoms(mol),
                Chem.rdchem.Mol.GetNumHeavyAtoms(mol),
                Chem.Crippen.MolMR(mol),
                Chem.QED.properties(mol).PSA,
                Chem.rdmolops.GetFormalCharge(mol),
                Chem.rdMolDescriptors.CalcNumRings(mol)
            ])
        elif mol_fp == 'gin_supervised_masking':
            if not 'gin_supervised_masking' in CALCULATORS.keys():
                CALCULATORS['gin_supervised_masking'] = PretrainedDGLTransformer(kind='gin_supervised_masking', dtype=float)
            feats = CALCULATORS['gin_supervised_masking'](smiles)[0]
        elif mol_fp == 'gin_supervised_infomax':
            if not 'gin_supervised_infomax' in CALCULATORS.keys():
                CALCULATORS['gin_supervised_infomax'] = PretrainedDGLTransformer(kind='gin_supervised_infomax', dtype=float)
            feats = CALCULATORS['gin_supervised_infomax'](smiles)[0]
        elif mol_fp == 'mhfp':
            if not 'mhfp' in CALCULATORS.keys():
                CALCULATORS['mhfp'] = MHFPEncoder()
            feats = CALCULATORS['mhfp'].encode(smiles)
        elif mol_fp == 'mordred2d':
            if not 'mordred2d' in CALCULATORS.keys():
                CALCULATORS['mordred2d'] = mordred.Calculator(mordred.descriptors, ignore_3D=True)
            feats = CALCULATORS['mordred2d'](mol)
        elif mol_fp == 'mordred3d':
            if not 'mordred3d' in CALCULATORS.keys():
                CALCULATORS['mordred3d'] = mordred.Calculator(mordred.descriptors, ignore_3D=False)
            feats = CALCULATORS['mordred3d'](mol)
        else:
            raise ValueError('Not defined fingerprint!')
    except Exception as e:
        print(f'{e} for {smiles} with {mol_fp}')
        try:
            feats = np.zeros_like(generate_fps('C', mol_fp))
        except:
            feats = np.zeros((1,))

    feats = np.nan_to_num(feats)
    feats = feats.astype(np.float32)
    return feats




@st.cache_data
def generate_splits(df, smiles_column, split_type='random', val_size=0.1, test_size=0.1, random_state=42):
    """
    Splits a DataFrame containing SMILES into training, validation, and testing sets.

    Parameters:
    - df: DataFrame containing the SMILES strings.
    - smiles_column: The name of the column containing the SMILES strings.
    - split_type: Type of split to perform ('random' or 'scaffold').
    - test_size: Fraction of the dataset to be used as test set.
    - val_size: Fraction of the dataset to be used as validation set.
    - random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
    - train_df: DataFrame containing the training set.
    - val_df: DataFrame containing the validation set.
    - test_df: DataFrame containing the testing set.
    """
    if split_type == 'random':
        # Random split
        if val_size == 0:
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            val_df = None
        else:
            rest_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
            # Adjust validation size relative to the remaining dataset
            relative_val_size = val_size / (1 - test_size)
            train_df, val_df = train_test_split(rest_df, test_size=relative_val_size, random_state=random_state)
    elif split_type == 'scaffold':
        # Scaffold split
        scaffolds = {}
        for index, row in df.iterrows():
            mol = Chem.MolFromSmiles(row[smiles_column])
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(index)
        
        # Flatten scaffold indices to list and shuffle for randomness
        scaffold_indices = [idx for indices in scaffolds.values() for idx in indices]
        np.random.shuffle(scaffold_indices)
        
        # Calculate the actual number of samples for each dataset
        n_total = len(df)
        n_test = int(n_total * test_size)
        n_val = int(n_total * val_size)

        # Allocate indices to test, validation, and training sets
        test_indices = scaffold_indices[:n_test]
        val_indices = scaffold_indices[n_test:n_test + n_val]
        train_indices = scaffold_indices[n_test + n_val:]

        train_df = df.iloc[train_indices]
        val_df = df.iloc[val_indices]
        test_df = df.iloc[test_indices]
    else:
        raise ValueError("split_type must be 'random' or 'scaffold'")
    
    return train_df, val_df, test_df


@st.cache_data
def generate_mol_grid_html(df, smiles_col):
    return mols2grid.display(
        df,
        smiles_col=smiles_col,

        n_items_per_page = 60,
        atomColourPalette=ELEMENT_COLORS_RGB,
        background_color="#2b2b2b",
        custom_css = """
        .m2g-cell-actions {
            color: grey;
        }
        .data-mols2grid-id-display {
            color: grey;
        }
        """,
        border="none",
        gap=2,
        selection=False,

        )._repr_html_()