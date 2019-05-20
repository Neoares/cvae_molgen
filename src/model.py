from keras import backend as K
from keras import objectives
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Lambda, concatenate, Permute, CuDNNGRU, GRU, Convolution1D, TimeDistributed, Flatten, RepeatVector


BASE_PATH = '../'


class MoleculeCVAE:
    def __init__(self, loss='binary', gpu_mode=True, lr=1e-4):
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.gpu_mode = gpu_mode
        self.loss = loss
        self.lr = lr

    def create(self,
               charset,
               max_length=120,
               latent_rep_size=292,
               cond_size=128,
               weights_file=None):
        charset_length = len(charset)

        x = Input(shape=(max_length, charset_length))
        cond = Input(shape=(cond_size,))
        cond_rep = RepeatVector(charset_length)(cond)
        cond_trans = Permute((2, 1))(cond_rep)
        inputs = concatenate([x, cond_trans], axis=-2)

        _, z = self._buildEncoder(inputs, latent_rep_size, max_length)
        self.encoder = Model([x, cond], z)

        #z_cond = concatenate([z, cond], axis=1)

        encoded_input = Input(shape=(latent_rep_size + cond_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        # x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(inputs, latent_rep_size, max_length)
        self.autoencoder = Model(
            [x, cond],
            self._buildDecoder(
                concatenate([z1, cond]),
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)

        optimizer = optimizers.Adam(lr=self.lr)

        self.autoencoder.compile(optimizer=optimizer,
                                 loss=vae_loss,
                                 metrics=['accuracy'])
        self.autoencoder.summary()

    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std=0.01):
        h = Convolution1D(9, 9, activation='relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation='relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation='relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        def binary_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        def categorical_loss(x, x_decoded_mean):
            xent_loss = K.mean(objectives.categorical_crossentropy(x, x_decoded_mean))
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        loss = categorical_loss if self.loss == 'categorical' else binary_loss

        return (loss,
                Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        if self.gpu_mode:
            h = CuDNNGRU(501, return_sequences=True, name='gru_1')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_2')(h)
            h = CuDNNGRU(501, return_sequences=True, name='gru_3')(h)
        else:
            h = GRU(501, return_sequences=True, name='gru_1')(h)
            h = GRU(501, return_sequences=True, name='gru_2')(h)
            h = GRU(501, return_sequences=True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)

    def load(self, charset, weights_file, latent_rep_size=292):
        self.create(charset, weights_file=weights_file, latent_rep_size=latent_rep_size)