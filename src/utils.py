import numpy as np
import pandas as pd
import time
from rdkit import Chem

from .model_utils import one_hot_index, one_hot_array
from .constants import CHARSET


def load(s2_keys_path, s2_matrix_path, key2inchi_path, sep=',', calc_smiles=False, save=True, limit=None):
    s2_keys = np.load(s2_keys_path)
    s2_matrix = np.load(s2_matrix_path)

    s2_keys = np.array(list(map(lambda x: x.decode('utf-8'), s2_keys)))
    key2inchi = pd.read_csv(key2inchi_path, sep=sep)

    if calc_smiles:
        t1 = time.time()
        key2inchi['smiles'] = key2inchi.inchi.apply(lambda x: Chem.MolToSmiles(Chem.MolFromInchi(x)))
        print("time to transform to smiles:", time.time() - t1)
        if save:
            key2inchi.to_csv(key2inchi_path, index=False)

    dict_map = dict(zip(key2inchi.key, key2inchi.smiles))
    smiles = pd.Series([dict_map[x] for x in s2_keys])
    mask = smiles.str.len() <= 120
    smiles = smiles[mask]
    s2_matrix = s2_matrix[mask]

    return s2_matrix[:limit], smiles[:limit].tolist()


def preprocess(smiles_list):
    print("Loading", len(smiles_list), "smiles.")
    cropped = [s.ljust(120) for s in smiles_list]
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(CHARSET)), one_hot_index(c, CHARSET))) for c in cropped])
    return preprocessed


def preprocess_multiprocess(smiles):
    cropped = smiles.ljust(120)
    preprocessed = list(map(lambda x: one_hot_array(x, len(CHARSET)), one_hot_index(cropped, CHARSET)))
    return preprocessed


def preprocess_generator(smiles_list, conditions, batch_size=64, shuffle=True):
    num_samples = len(smiles_list)
    while True:
        # Every iteration here is an epoch
        if shuffle:
            np.random.shuffle(smiles_list)

        for i in range(0, num_samples, batch_size):
            # create the batches
            smiles_batch = smiles_list[i: min(i + batch_size, num_samples)]
            conditions_batch = conditions[i: min(i + batch_size, num_samples)]

            # left zero-padding for the smiles
            cropped = [s.ljust(120) for s in smiles_batch]

            preprocessed = np.array(
                [list(map(lambda x: one_hot_array(x, len(CHARSET)), one_hot_index(c, CHARSET))) for c in cropped])
            yield ([preprocessed, conditions_batch], preprocessed)