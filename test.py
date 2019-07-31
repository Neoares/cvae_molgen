import os
from src.gan import MoleculeGAN
from src.aae import MoleculeAAE
from src.utils import load, preprocess
import matplotlib.pyplot as plt

BASE_PATH = os.path.dirname(os.path.join(os.getcwd(), __file__))

signatures, smiles = load(
    os.path.join(BASE_PATH, 'data/inchikeys_B4.npy'),
    os.path.join(BASE_PATH, 'data/signature_B4_matrix.npy'),
    os.path.join(BASE_PATH, 'data/key2inch_B4.csv'),
    calc_smiles=False,
    limit=1000)


aae = MoleculeAAE(preprocess(smiles))
'''
aae.encoder.summary()
aae.decoder.summary()
aae.autoencoder.summary()
aae.discriminator.summary()
aae.enc_disc.summary()
'''
aae.train(n_epochs=2, batch_size=100)


'''
for i in range(100):
    gan.train(n_epochs=1, batch_size=64)
    # generate samples
    # generate samples
    n_samples = 10
    X, _ = gan.generate_fake_samples(n_samples)
    fig = plt.figure(figsize=(24, 24))
    # plot the generated samples
    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1, xticks=[], yticks=[])
        plt.imshow(gan.decode_img(X[i]))
    # show the figure
    plt.show()
'''
