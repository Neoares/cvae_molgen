import os
import numpy as np
import multiprocessing
import time

from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from src.model import MoleculeCVAE
from src.utils import preprocess, preprocess_multiprocess, preprocess_generator, load

from argparse import ArgumentParser

from src.constants import CHARSET, LATENT_DIM

BASE_PATH = os.path.dirname(os.path.join(os.getcwd(), __file__))


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
        callbacks=callbacks,
        verbose=config.verbose)


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
                                    callbacks=callbacks,
                                    verbose=config.verbose)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-s', '--size', default=400_000, type=int, help='Number of samples to use in training.')
    parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs for the model.')
    parser.add_argument('-bs', '--batch-size', default=256, type=int, help='Batch size.')
    parser.add_argument('--gpu', action='store_true', help='Whether to use a GPU-optimized architecture or not.')
    parser.add_argument('--loss', default='binary', type=str,
                        help='Loss function to use. Types: [binary (default), categorical]')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')

    parser.add_argument('-mp', '--multi-processing', action='store_true',
                        help='Whether to use multiple threads to process all data or not.')
    parser.add_argument('--use-generators', action='store_true',
                        help='Whether to use generators while training or not.')
    parser.add_argument('-v', '--verbose', default=2, type=int, help='Verbose for the training part.')
    parser.add_argument('-n', '--name', default='model', type=str, help='Name of the model.')

    args = parser.parse_args()

    trained_model = os.path.join(BASE_PATH, 'models/model_400k_.995.hdf5')

    model = MoleculeCVAE(loss=args.loss, gpu_mode=args.gpu, lr=args.learning_rate)
    model.create(CHARSET, latent_rep_size=LATENT_DIM)

    signatures, smiles = load(
        os.path.join(BASE_PATH, 'data/s2_keys_400k.npy'),
        os.path.join(BASE_PATH, 'data/s2_matrix_400k.npy'),
        os.path.join(BASE_PATH, 'data/key2inch_400k.csv'),
        calc_smiles=False,
        limit=args.size)

    history = LossHistory()
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(BASE_PATH, 'models/{}_{}.hdf5'.format(round(time.time()), args.name)))
    tensorboard = TensorBoard(
        log_dir=os.path.join(BASE_PATH, "logs/{}_{}".format(round(time.time()), args.name)))

    if args.use_generators:
        train_generator(smiles, signatures, args, callbacks=[checkpoint, tensorboard])
    else:
        train(smiles, signatures, args, callbacks=[checkpoint, tensorboard])
