import h5py
import numpy as np
import os
from tqdm import tqdm

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

    smiles = np.array([decode_smiles_from_indexes(d) for d in decoded_mols])
    return smiles


'''
Reconstructs molecules using the model.

@param smiles_list list of smiles to reconstruct
@param n number of reconstructions for each molecule
@param m number of molecules to reconstruct from the array passed
@param stds list of standard deviations to perform the simulation
'''


def sim_reconstruction(model, smiles_list, signatures, latent_dim=292, su=None, n=100, m=0, stds=[0.05],
                       fix='signatures'):
    if m > 0:
        smiles_list = smiles_list[:m]
    if fix == 'smiles' and not su:
        print("In fix=smiles mode, you may provide a SignatureUtils instance")
        return 0

    if fix in ['smiles', 'smile']:
        stds = ['different', 'neutral', 'similar']

    reconstructed_smiles = {std: [] for std in stds}

    if fix in ['signatures', 'signature']:
        for i, std in enumerate(stds):
            # print("std:", std)
            for j, smiles in tqdm(enumerate(smiles_list)):  # enumerate(smiles_list)
                signature = signatures[j]

                replicated_smiles = np.array([smiles] * n)
                replicated_signature = np.array([signature] * n)
                latents = encode_smiles(replicated_smiles, replicated_signature, model)

                noised_latents = std * np.random.normal(size=(n, latent_dim)) + latents
                recons = decode_latent_molecules(noised_latents, replicated_signature, model)

                reconstructed_smiles[std].append(recons)

    elif fix in ['smiles', 'smile']:
        for idx, smiles in tqdm(enumerate(smiles_list)):
            signature = signatures[idx]

            replicated_smiles = np.array([smiles] * n)
            replicated_signature = np.array([signature] * n)
            latents = encode_smiles(replicated_smiles, replicated_signature, model)

            # Let's try with 0.05 std for all the molecules
            std = 0.05
            noised_latents = std * np.random.normal(size=(n, latent_dim)) + latents

            args = su.get_argsorted(signature)
            chosen_signatures = {'different': su.all_signatures[args[:n]],
                                 'neutral': su.all_signatures[
                                     args[len(args) // 2 - n // 2: len(args) // 2 - n // 2 + n]],
                                 'similar': su.all_signatures[args[-n:]]}

            for name, signs in chosen_signatures.items():
                recons = decode_latent_molecules(noised_latents, signs, model)
                reconstructed_smiles[name].append(recons)

    return reconstructed_smiles
