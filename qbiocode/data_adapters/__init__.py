"""
QBioCode Data Adapters
======================

Subpackage providing data loaders and molecular featurizers that bridge
external benchmark datasets (TDC ADMET) into QBioCode's CSV-based pipeline.

Modules
-------
tdc_admet_loader
    Downloads TDC ADMET benchmark endpoints, applies binarization where
    needed, and writes canonical train/valid/test CSV splits ready for
    QProfiler consumption.

molecular_featurizers
    Converts SMILES strings to numerical feature matrices using ECFP4,
    MACCS key fingerprints, and RDKit 2D physicochemical descriptors.

Usage
-----
>>> from qbiocode.data_adapters import TDCAdmetLoader, MolecularFeaturizer
>>> loader = TDCAdmetLoader(data_dir='data/admet')
>>> loader.prepare_all()
"""

from .tdc_admet_loader import TDCAdmetLoader, ADMET_ENDPOINTS
from .molecular_featurizers import MolecularFeaturizer, FEATURIZERS

__all__ = [
    "TDCAdmetLoader",
    "ADMET_ENDPOINTS",
    "MolecularFeaturizer",
    "FEATURIZERS",
]
