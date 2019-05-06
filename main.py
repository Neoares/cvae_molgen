import os
import numpy as np

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from src.model import MoleculeCVAE
from src.utils import preprocess_generator, load

BASE_PATH = ''

# number of dimensions to represent the molecules
# as the model was trained with this number, any operation made with the model must share the dimensions.
LATENT_DIM = 292

# trained_model 0.99 validation accuracy
# trained with 80% of ALL chembl molecules, validated on the other 20.
trained_model = os.path.join(BASE_PATH, 'models/model_50k.hdf5')
# charset_file = 'charset.json'
charset = [" ", "(", ".", "0", "2", "4", "6", "8", "@", "B", "F", "H", "L", "N", "P", "R", "T", "V", "X", "Z", "\\",
           "b", "d", "l", "n", "p", "r", "t", "#", "%", ")", "+", "-", "/", "1", "3", "5", "7", "9", "=", "A", "C", "G",
           "I", "K", "M", "O", "S", "[", "]", "a", "c", "e", "g", "i", "o", "s", "u"]

model = MoleculeCVAE(gpu_mode=False)
model.create(charset, latent_rep_size=LATENT_DIM)
# model.load(charset, trained_model, latent_rep_size=latent_dim)
# model.autoencoder.summary()


class LossHistory(Callback):
    def __init__(self):
        self.logs = []

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.logs.append(logs)


s2_matrix, smiles = load(
    os.path.join(BASE_PATH, 'data/s2_keys_400k.npy'),
    os.path.join(BASE_PATH, 'data/s2_matrix_400k.npy'),
    os.path.join(BASE_PATH, 'data/key2inch_400k.csv'),
    calc_smiles=False,
    limit=2_000)

validation_split = .2
split = int(len(smiles) * (1-validation_split))
batch_size = 64


x_train = preprocess_generator(smiles[:split], s2_matrix, charset, batch_size=batch_size)
x_test = preprocess_generator(smiles[split:], s2_matrix, charset, batch_size=batch_size)

history = LossHistory()
checkpoint = ModelCheckpoint(filepath=os.path.join(BASE_PATH, 'models/model_50k.hdf5'))

model.autoencoder.fit_generator(x_train,
                                validation_data=x_test,
                                steps_per_epoch=np.ceil((split+1)/batch_size),
                                validation_steps=np.ceil((len(smiles)-split)/batch_size),
                                epochs=200,
                                callbacks=[history])
