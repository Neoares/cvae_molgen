import numpy as np
import os
from argparse import ArgumentParser

from src.utils import load, SignatureUtils
from src.model_utils import sim_reconstruction
from src.model import MoleculeCVAE
from src.constants import CHARSET, LATENT_DIM

BASE_PATH = os.path.dirname(os.path.join(os.getcwd(), __file__))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--num_molecules', default=0, type=int, help='Number of molecules to reconstruct.')
    parser.add_argument('-r', '--num_reconstructions', default=100, type=int,
                        help='Number of reconstructions per molecule.')
    parser.add_argument('-fix', '--fix', default='signature', type=str, help='Wether to fix the SMILES or the signature')
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU-optimized architecture or not.')

    args = parser.parse_args()

    trained_model = os.path.join(BASE_PATH, 'models/model_400k_.995.hdf5')

    model = MoleculeCVAE(gpu_mode=args.gpu)
    model.create(CHARSET, latent_rep_size=LATENT_DIM)
    # model.load(CHARSET, trained_model, latent_rep_size=LATENT_DIM)
    # model.autoencoder.summary()

    s2_matrix, smiles = load(
        os.path.join(BASE_PATH, 'data/s2_keys_400k.npy'),
        os.path.join(BASE_PATH, 'data/s2_matrix_400k.npy'),
        os.path.join(BASE_PATH, 'data/key2inch_400k.csv'),
        calc_smiles=False)

    m = args.num_molecules
    n = args.num_reconstructions

    reconstructed_smiles = sim_reconstruction(
        model,
        np.array(smiles[-m:]),
        np.array(s2_matrix[-m:]),
        latent_dim=LATENT_DIM,
        su=SignatureUtils(s2_matrix),
        m=m,
        n=n,
        stds=[0, 0.05, 0.1],
        fix=args.fix
    )

    original_smiles = smiles[-m:]
