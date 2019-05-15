import os
import numpy as np
import multiprocessing
import time

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from src.model import MoleculeCVAE
from src.utils import preprocess, preprocess_multiprocess, preprocess_generator, load

from argparse import ArgumentParser

from src.constants import CHARSET

BASE_PATH = ''


class LossHistory(Callback):
    def __init__(self):
        self.logs = []

    def on_train_begin(self, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.logs.append(logs)


def train(x, c, config, callbacks=()):
    if config.multi_processing:
        t1 = time.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as p:
            x = p.map(preprocess_multiprocess, x)
        print("time loading with multiprocess:", time.time() - t1)
        t1 = time.time()
        print("converting no numpy array...")
        x = np.array(x, dtype='int8')
        print("time to convert to numpy array:", time.time()-t1)
    else:
        x = preprocess(x)

    model.autoencoder.fit(
        [x, c],
        x,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.2,
        callbacks=callbacks)


def train_generator(x, c, config, callbacks=()):
    validation_split = .2
    split = int(len(x) * (1-validation_split))

    x_train = preprocess_generator(x[:split], c, batch_size=config.batch_size)
    x_test = preprocess_generator(x[split:], c, batch_size=config.batch_size)

    model.autoencoder.fit_generator(x_train,
                                    validation_data=x_test,
                                    steps_per_epoch=np.ceil((split+1) / config.batch_size),
                                    validation_steps=np.ceil((len(smiles) - split) / config.batch_size),
                                    epochs=config.epochs,
                                    callbacks=callbacks)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-s', '--size', default=400_000, type=int, help='Number of samples to use in training.')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs for the model.')
    parser.add_argument('-bs', '--batch-size', default=256, type=int, help='Batch size.')
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU-optimized architecture or not.')
    parser.add_argument('-mp', '--multi-processing', action='store_true',
                        help='Whether to use multiple threads to process all data or not.')
    parser.add_argument('--use-generators', action='store_true',
                        help='Whether to use generators while training or not.')

    args = parser.parse_args()
    print(args)

    # number of dimensions to represent the molecules
    # as the model was trained with this number, any operation made with the model must share the dimensions.
    LATENT_DIM = 292

    trained_model = os.path.join(BASE_PATH, 'models/model_400k_.995.hdf5')

    model = MoleculeCVAE(gpu_mode=args.gpu)
    model.create(CHARSET, latent_rep_size=LATENT_DIM)
    # model.load(CHARSET, trained_model, latent_rep_size=LATENT_DIM)
    # model.autoencoder.summary()

    s2_matrix, smiles = load(
        os.path.join(BASE_PATH, 'data/s2_keys_400k.npy'),
        os.path.join(BASE_PATH, 'data/s2_matrix_400k.npy'),
        os.path.join(BASE_PATH, 'data/key2inch_400k.csv'),
        calc_smiles=False,
        limit=args.size)

    history = LossHistory()
    checkpoint = ModelCheckpoint(filepath=os.path.join(BASE_PATH, 'models/model_test.hdf5'))

    if args.use_generators:
        train_generator(smiles, s2_matrix, args)
    else:
        train(smiles, s2_matrix, args)
