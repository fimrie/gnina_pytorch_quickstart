"""Module to check RMSD between docked and crystal ligand."""
"""Adapted from https://github.com/maabuu/posebusters/blob/13bbbf08e081385d9d4215c626170da2fb5f41c2/posebusters/tools/molecules.py """

import logging
from copy import deepcopy

import numpy as np
from rdkit.Chem.AllChem import AssignBondOrdersFromTemplate
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdMolAlign import CalcRMS, GetBestRMS
from rdkit.Chem.rdmolfiles import MolFromSmarts
from rdkit.Chem.rdmolops import RemoveHs, RemoveStereochemistry

tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
tautomer_enumerator.SetMaxTautomers(100000)
tautomer_enumerator.SetMaxTransforms(100000)
tautomer_enumerator.SetReassignStereo(True)
tautomer_enumerator.SetRemoveBondStereo(True)
tautomer_enumerator.SetRemoveSp3Stereo(True)


def neutralize_atoms(mol: Mol) -> Mol:
    """Add and remove hydrogens to neutralize charges ignoring overall charge."""
    # https://www.rdkit.org/docs/Cookbook.html#neutralizing-charged-molecules
    # stronger than rdkit.Chem.MolStandardize.rdMolStandardize.Uncharger
    try:
        pattern = MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
    except AtomValenceException:
        logger.warning("AtomValenceException raised while neutralizing molecule. Continuing with original molecule.")
        return mol
    return mol

def remove_all_charges_and_hydrogens(mol: Mol) -> Mol:
    """Remove all charges and hydrogens from molecule."""
    try:
        # rdkit keeps hydrogens that define sterochemistry so remove stereo first
        RemoveStereochemistry(mol)
        mol = RemoveHs(mol)
        # remove charges
        mol = neutralize_atoms(mol)
    except AtomValenceException:
        # from rdkit.Chem.MolStandardize.rdMolStandardize import Uncharger
        # mol = Uncharger().uncharge(mol)
        logger.warning("AtomValenceException raised while neutralizing molecule. Continuing with original molecule.")
        return mol
    return mol

def robust_rmsd(  # noqa: PLR0913
    mol_probe: Mol,
    mol_ref: Mol,
    conf_id_probe: int = -1,
    conf_id_ref: int = -1,
    drop_stereo: bool = False,
    heavy_only: bool = True,
    kabsch: bool = False,
    symmetrizeConjugatedTerminalGroups=True,
    **params,
) -> float:
    """RMSD calculation for isomers."""
    mol_probe = deepcopy(mol_probe)
    mol_ref = deepcopy(mol_ref)  # copy mols because rdkit RMSD calculation aligns mols

    if drop_stereo:
        RemoveStereochemistry(mol_probe)
        RemoveStereochemistry(mol_ref)

    if heavy_only:
        mol_probe = RemoveHs(mol_probe, sanitize=False)
        mol_ref = RemoveHs(mol_ref, sanitize=False)

    # combine parameters
    params = dict(symmetrizeConjugatedTerminalGroups=symmetrizeConjugatedTerminalGroups, kabsch=kabsch, **params)

    # calculate RMSD
    rmsd = _call_rdkit_rmsd(mol_probe, mol_ref, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again ignoring charges or tautomers
    rmsd = _rmsd_ignoring_charges_and_tautomers(mol_ref, mol_probe, conf_id_ref, conf_id_probe, params)
    if not np.isnan(rmsd):
        return rmsd

    # try assigning the bond orders of one molecule to the other
    try:
        mol_probe_new_bonds = AssignBondOrdersFromTemplate(mol_ref, mol_probe)
    except ValueError:
        return np.nan
    rmsd = _call_rdkit_rmsd(mol_probe_new_bonds, mol_ref, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    return np.nan


def _rmsd_ignoring_charges_and_tautomers(
    mol_ref: Mol, mol_probe: Mol, conf_id_ref: int, conf_id_probe: int, params: dict
) -> float:
    # try again but remove charges and hydrogens
    mol_ref_uncharged = remove_all_charges_and_hydrogens(mol_ref)
    mol_probe_uncharged = remove_all_charges_and_hydrogens(mol_probe)
    rmsd = _call_rdkit_rmsd(mol_probe_uncharged, mol_ref_uncharged, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but neutralize atoms
    mol_ref_neutralized = neutralize_atoms(mol_ref)
    mol_probe_neutralized = neutralize_atoms(mol_probe)
    rmsd = _call_rdkit_rmsd(mol_probe_neutralized, mol_ref_neutralized, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but on canonical tautomers
    mol_ref_canonical = tautomer_enumerator.Canonicalize(mol_ref)
    mol_probe_canonical = tautomer_enumerator.Canonicalize(mol_probe)
    rmsd = _call_rdkit_rmsd(mol_probe_canonical, mol_ref_canonical, conf_id_probe, conf_id_ref, **params)
    if not np.isnan(rmsd):
        return rmsd

    # try again but after neutralizing atoms
    mol_ref_neutral_canonical = tautomer_enumerator.Canonicalize(neutralize_atoms(mol_ref))
    mol_probe_neutral_canonical = tautomer_enumerator.Canonicalize(neutralize_atoms(mol_probe))
    rmsd = _call_rdkit_rmsd(
        mol_probe_neutral_canonical, mol_ref_neutral_canonical, conf_id_probe, conf_id_ref, **params
    )
    if not np.isnan(rmsd):
        return rmsd

    return rmsd


def _call_rdkit_rmsd(mol_probe: Mol, mol_ref: Mol, conf_id_probe: int, conf_id_ref: int, **params):
    try:
        return _rmsd(mol_probe, mol_ref, conf_id_probe, conf_id_ref, **params)
    except RuntimeError:
        pass
    except ValueError:
        pass

    return np.nan


def _rmsd(mol_probe: Mol, mol_ref: Mol, conf_id_probe: int, conf_id_ref: int, kabsch: bool = False, **params):
    if kabsch is True:
        return GetBestRMS(prbMol=mol_probe, refMol=mol_ref, prbId=conf_id_probe, refId=conf_id_ref, **params)
    return CalcRMS(prbMol=mol_probe, refMol=mol_ref, prbId=conf_id_probe, refId=conf_id_ref, **params)


def intercentroid(
    mol_probe: Mol, mol_ref: Mol, conf_id_probe: int = -1, conf_id_ref: int = -1, heavy_only: bool = True
) -> float:
    """Distance between centroids of two molecules."""

    centroid_probe = get_centroid(mol_probe, heavy_only, conf_id_probe)
    centroid_ref = get_centroid(mol_ref, heavy_only, conf_id_ref)
    return float(np.linalg.norm(centroid_probe - centroid_ref))


def get_centroid(mol: Mol, heavy_only: bool = True, conf_id: int = -1) -> np.ndarray:
    """Get centroid of molecule."""
    pos = mol.GetConformer(conf_id).GetPositions()
    if heavy_only:
        pos = pos[[atom.GetAtomicNum() != 1 for atom in mol.GetAtoms()], :]
    return pos.mean(axis=0)