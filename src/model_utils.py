import h5py
import numpy as np
from .constants import CHARSET


def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))


def one_hot_index(vec):
    return list(map(CHARSET.index, vec))


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0,):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec):
    return "".join(map(lambda x: CHARSET[x], vec)).strip()


def load_dataset(filename, split=True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset


def preprocess_smiles(smiles_list):
    cropped = [s.ljust(120) for s in smiles_list]
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(CHARSET)), one_hot_index(c))) for c in cropped])
    return preprocessed


def encode_smiles(smiles_list, signatures, model, batch_size=256):
    # cropped = list(smiles.ljust(120))
    if type(smiles_list) == str:
        smiles_list = np.array([smiles_list])

    preprocessed = preprocess_smiles(smiles_list)

    latent = model.encoder.predict([preprocessed, signatures], batch_size=batch_size)
    return latent


def decode_latent_molecules(latents, signatures, model, batch_size=256):
    if type(latents) == str:
        latents = np.array(latents)

    decoded_mols = np.array(
        [x.argmax(axis=1) for x in model.decoder.predict(
            np.concatenate([latents, signatures], axis=1), batch_size=batch_size
        )]
    )

    smiles = np.array([decode_smiles_from_indexes(d, CHARSET) for d in decoded_mols])
    return smiles

