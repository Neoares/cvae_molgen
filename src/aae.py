from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, Dropout, LeakyReLU, Reshape, Input, BatchNormalization, ReLU, RepeatVector, CuDNNGRU, GRU, TimeDistributed
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal, TruncatedNormal
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class MoleculeAAE:
    def __init__(self, smiles, latent_dim=128, gpu_mode=False):
        self.smiles = np.array(smiles)
        self.latent_dim = latent_dim
        self.gpu_mode = gpu_mode
        self.build_models()

    def generate_real_samples(self, n_samples):
        # choose random instances
        # ix = np.random.randint(0, len(files), n_samples)
        ix = np.random.randint(0, len(self.smiles), n_samples)
        # retrieve selected images
        # X = np.array([encode_img(data_gen.random_transform(np.array(Image.open(files[i]).resize(size)))) for i in ix])
        X = self.smiles[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        # y = np.random.uniform(0.9, 1, size=(n_samples, 1))
        return X, y

    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        # x_input = np.random.randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        # x_input = x_input.reshape(n_samples, latent_dim)
        x_input = np.random.randn(n_samples, self.latent_dim)
        return x_input

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, n_samples):
        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = self.generator.predict(x_input)
        # create 'fake' class labels (0)
        y = np.zeros((n_samples, 1))
        # y = np.random.uniform(0, 0.1, size=(n_samples, 1))
        return X, y

    def define_generator(self, inp):
        initializer = TruncatedNormal(mean=0.0, stddev=0.02)
        kernel_size = 5

        # x = inp
        # x = Dense(30 * 232)(inp)
        # x = Reshape((30, 232))(x)
        # x = LeakyReLU(alpha=0.2)(x)

        h = Dense(self.latent_dim, name='latent_input', activation='relu')(inp)
        h = RepeatVector(120, name='repeat_vector')(h)
        if self.gpu_mode:
            h = CuDNNGRU(501, return_sequences=True, name='gru_1')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_2')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_3')(h)
        else:
            h = GRU(501, return_sequences=True, name='gru_1', reset_after=True, recurrent_activation='sigmoid')(h)
            h = GRU(501, return_sequences=True, name='gru_2', reset_after=True, recurrent_activation='sigmoid')(h)
            h = GRU(501, return_sequences=True, name='gru_3', reset_after=True, recurrent_activation='sigmoid')(h)
        return TimeDistributed(Dense(58, activation='softmax'), name='decoded_mean')(h)

        '''
        x = Conv1D(n_filters * 4, kernel_size, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Conv1D(n_filters * 4, kernel_size, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = UpSampling1D(size=2)(x)

        x = Conv1D(n_filters * 2, kernel_size, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Conv1D(n_filters * 2, kernel_size, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = UpSampling1D(size=2)(x)

        x = Conv1D(n_filters, kernel_size, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = Conv1D(n_filters, kernel_size, padding='same', activation='softmax', kernel_initializer=initializer)(x)
        # x = UpSampling1D(size=2)(x)
        '''

        # out = Conv2D(3, kernel_size, strides=1, activation='tanh', padding='same', kernel_initializer=initializer)(x)

        '''
        x = Conv2DTranspose(N_FILTERS, (3,3), strides=(2,2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        # out = Conv2D(3, (5, 5), activation='sigmoid', padding='same')(x)
        '''
        return x

    def build_encoder(self, inp):
        h = Conv1D(9, 9, activation='relu', name='conv_1')(inp)
        h = Conv1D(9, 9, activation='relu', name='conv_2')(h)
        h = Conv1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)
        h = Dense(self.latent_dim, activation='relu', name='dense_2')(h)
        return h

    def build_decoder(self, inp):
        # h = Dense(self.latent_dim, name='latent_input', activation='relu')(inp)
        h = RepeatVector(120, name='repeat_vector')(inp)
        if self.gpu_mode:
            h = CuDNNGRU(501, return_sequences=True, name='gru_1')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_2')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_3')(h)
        else:
            h = GRU(501, return_sequences=True, name='gru_1', reset_after=True, recurrent_activation='sigmoid')(h)
            h = GRU(501, return_sequences=True, name='gru_2', reset_after=True, recurrent_activation='sigmoid')(h)
            h = GRU(501, return_sequences=True, name='gru_3', reset_after=True, recurrent_activation='sigmoid')(h)
        return TimeDistributed(Dense(58, activation='softmax'), name='decoded_mean')(h)

    def define_discriminator(self, inp):
        initializer = TruncatedNormal(mean=0.0, stddev=0.02)
        n_filters = 256

        x = Dense(n_filters, activation='relu', kernel_initializer=initializer)(inp)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Dense(n_filters, activation='relu', kernel_initializer=initializer)(x)
        # x = BatchNormalization(epsilon=0.0005)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Dense(n_filters, activation='relu', kernel_initializer=initializer)(x)
        # x = BatchNormalization(epsilon=0.0005)(x)
        # x = LeakyReLU(alpha=0.2)(x)

        x = Dense(n_filters, activation='relu', kernel_initializer=initializer)(x)
        # x = BatchNormalization(epsilon=0.0005)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        return x

    def build_models(self):
        #opt_G = Adam(lr=2e-4, beta_1=0.5)
        #opt_D = Adam(lr=5e-4, beta_1=0.5)
        # opt_D = SGD(lr=5*1e-3, decay=1e-3)

        encoder_input = Input(shape=(120, 58))
        encoder_output = self.build_encoder(encoder_input)
        encoder = Model(inputs=encoder_input, outputs=encoder_output)
        encoder.summary()
        self.encoder = encoder

        decoder_input = Input(shape=(self.latent_dim,))
        decoder_output = self.build_decoder(decoder_input)
        decoder = Model(inputs=decoder_input, outputs=decoder_output)
        decoder.summary()
        self.decoder = decoder

        # self.autoencoder = Model(inputs=encoder_input, outputs=self.build_decoder(encoder_output))
        self.autoencoder = Sequential()
        self.autoencoder.add(encoder)
        self.autoencoder.add(decoder)
        self.autoencoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.autoencoder.summary()

        disc_inp = Input(shape=(self.latent_dim,))
        disc_out = self.define_discriminator(disc_inp)
        self.discriminator = Model(inputs=disc_inp, outputs=disc_out)
        self.discriminator.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.discriminator.summary()

        self.discriminator.trainable = False

        self.enc_disc = Sequential()
        self.enc_disc.add(encoder)
        self.enc_disc.add(self.discriminator)
        self.enc_disc.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        self.enc_disc.summary()


    # train the generator and discriminator
    def train(self, n_epochs=100, batch_size=256):
        # bat_per_epo = int(len(files) / batch_size)
        bat_per_epo = int(len(self.smiles) / batch_size)
        half_batch = int(batch_size / 2)
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            # print("epoch {}".format(i))
            for j in tqdm(range(bat_per_epo)):
                # get randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(batch_size)
                # update autoencoder model weights
                ae_loss, _ = self.autoencoder.train_on_batch(X_real, X_real)

                batchpred = self.encoder.predict(X_real[:half_batch])
                fakepred = np.random.normal(size=(half_batch, self.latent_dim))
                disc_batch_x = np.concatenate([batchpred, fakepred])
                disc_batch_y = np.concatenate([np.zeros(half_batch), np.ones(half_batch)])
                disc_loss, _ = self.discriminator.train_on_batch(disc_batch_x, disc_batch_y)

                self.enc_disc.train_on_batch(X_real, np.ones(batch_size))
                # summarize loss on this batch
            # evaluate the model performance, sometimes
            if (i + 1) % 1 == 0:
                self.summarize_performance(i)

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, epoch, n_samples=200):
        # prepare real samples
        X_real, y_real = self.generate_real_samples(n_samples)
        # evaluate reconstruction loss using autoencoder
        ae_loss, ae_acc = self.autoencoder.evaluate(X_real, X_real)
        # evaluate discriminator on real examples
        batchpred = self.encoder.predict(X_real)
        val_real, acc_real = self.discriminator.evaluate(batchpred, y_real, verbose=0)
        # prepare fake examples
        fake_pred = np.random.normal(size=(n_samples, self.latent_dim))
        # evaluate discriminator on fake examples
        val_fake, acc_fake = self.discriminator.evaluate(fake_pred, np.zeros(n_samples), verbose=0)
        # summarize discriminator performance
        print('>Epoch {}. AE loss: {}, Loss real: {}, fake: {}. Accuracy real: {}, fake: {}'.format(epoch, ae_loss, val_real, val_fake,
                                                                                       acc_real * 100, acc_fake * 100))
        # save plot
        # self.save_plot(x_fake, epoch)
        # save the generator model tile file
        # filename = 'generator_model_%03d.h5' % (epoch + 1)
        # self.generator.save(filename)

    '''
    # create and save a plot of generated images (reversed grayscale)
    def save_plot(self, examples, epoch, n=10):
        # plot images
        for i in range(n * n):
            # define subplot
            plt.subplot(n, n, 1 + i)
            # turn off axis
            plt.axis('off')
            # plot raw pixel data
            plt.imshow(self.decode_img(examples[i]))
        # save plot to file
        filename = 'generated_plot_e%03d.png' % (epoch + 1)
        plt.savefig(filename)
        plt.close()
    '''
