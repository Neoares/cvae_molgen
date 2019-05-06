import h5py
import numpy as np
from rdkit import Chem


def one_hot_array(i, n):
    return list(map(int, [ix == i for ix in range(n)]))


def one_hot_index(vec, charset):
    return list(map(charset.index, vec))


def from_one_hot_array(vec):
    oh = np.where(vec == 1)
    if oh[0].shape == (0,):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


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


def encode_smiles(smiles_list, model, charset):
    # cropped = list(smiles.ljust(120))
    if type(smiles_list) == str:
        smiles_list = np.array([smiles_list])

    cropped = [s.ljust(120) for s in smiles_list]
    preprocessed = np.array(
        [list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(c, charset))) for c in cropped])
    latent = model.encoder.predict(preprocessed)
    return latent


def decode_latent_molecules(latents, model, charset, latent_dim):
    if type(latents) == str:
        latents = np.array(latents)

    decoded_mols = np.array([x.argmax(axis=1) for x in model.decoder.predict(latents)])
    smiles = np.array([decode_smiles_from_indexes(d, charset) for d in decoded_mols])
    return smiles


def interpolate(source_smiles, dest_smiles, steps, charset, model, latent_dim):
    source_latent = encode_smiles(source_smiles, model, charset)
    dest_latent = encode_smiles(dest_smiles, model, charset)
    step = (dest_latent - source_latent) / float(steps)
    results = []
    for i in range(steps):
        item = source_latent + (step * i)
        decoded = decode_latent_molecules(item, model, charset, latent_dim)
        results.append(decoded)
    return results


def get_unique_mols(mol_list):
    inchi_keys = [Chem.InchiToInchiKey(Chem.MolToInchi(m)) for m in mol_list]
    u, indices = np.unique(inchi_keys, return_index=True)
    unique_mols = [[mol_list[i], inchi_keys[i]] for i in indices]
    return unique_mols

