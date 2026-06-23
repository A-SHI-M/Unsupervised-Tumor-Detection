import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Dropout, LeakyReLU,
    Conv2D, Conv2DTranspose, BatchNormalization,
    Concatenate, Add, GlobalAveragePooling2D, multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16


def attention_block(x, filters: int):
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // 16, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return multiply([x, se])


def build_encoder(img_shape: tuple, latent_dim: int) -> Model:
    inputs = Input(shape=img_shape, name="encoder_input")

    x = Conv2D(32, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 64)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 128)

    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 256)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(0.2)(x)
    z = Dense(latent_dim, name="z")(x)

    return Model(inputs, z, name="encoder")


def build_generator(latent_dim: int, img_shape: tuple) -> Model:
    inputs = Input(shape=(latent_dim,), name="generator_input")

    x = Dense(8 * 8 * 256)(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((8, 8, 256))(x)

    x = Conv2DTranspose(256, (4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 256)

    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 128)
    skip1 = x
    res = Conv2D(128, (3, 3), padding='same')(x)
    res = BatchNormalization()(res)
    res = LeakyReLU(0.2)(res)
    res = Conv2D(128, (3, 3), padding='same')(res)
    res = BatchNormalization()(res)
    res = LeakyReLU(0.2)(res)
    x = Add()([res, skip1])

    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = attention_block(x, 64)
    skip2 = x
    res = Conv2D(64, (3, 3), padding='same')(x)
    res = BatchNormalization()(res)
    res = LeakyReLU(0.2)(res)
    res = Conv2D(64, (3, 3), padding='same')(res)
    res = BatchNormalization()(res)
    res = LeakyReLU(0.2)(res)
    x = Add()([res, skip2])

    x = Conv2DTranspose(32, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    outputs = Conv2D(img_shape[-1], (3, 3), activation='sigmoid', padding='same', name="generated")(x)

    return Model(inputs, outputs, name="generator")


def build_discriminator(img_shape: tuple, latent_dim: int) -> Model:
    img_input = Input(shape=img_shape, name="disc_img_input")
    z_input = Input(shape=(latent_dim,), name="disc_z_input")

    x = Conv2D(32, (4, 4), strides=2, padding='same')(img_input)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Flatten()(x)

    z = Dense(512)(z_input)
    z = LeakyReLU(0.2)(z)

    combined = Concatenate()([x, z])
    combined = Dense(512)(combined)
    combined = LeakyReLU(0.2)(combined)
    combined = Dropout(0.5)(combined)
    validity = Dense(1, activation='sigmoid', name="validity")(combined)

    return Model([img_input, z_input], validity, name="discriminator")


def build_bigan(
    generator: Model,
    discriminator: Model,
    encoder: Model,
    latent_dim: int,
    img_shape: tuple,
    lr: float,
) -> Model:
    discriminator.trainable = False

    z = Input(shape=(latent_dim,), name="bigan_z")
    img = Input(shape=img_shape, name="bigan_img")

    fake_img = generator(z)
    real_z = encoder(img)

    fake_validity = discriminator([fake_img, z])
    real_validity = discriminator([img, real_z])

    model = Model([z, img], [fake_validity, real_validity], name="bigan")
    model.compile(
        optimizer=Adam(learning_rate=lr, beta_1=0.5),
        loss=['binary_crossentropy', 'binary_crossentropy'],
    )
    return model


def build_perceptual_model(img_shape: tuple) -> Model:
    h, w, c = img_shape
    vgg_input_shape = (h, w, 3)
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=vgg_input_shape)
    vgg.trainable = False

    feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3']
    outputs = [vgg.get_layer(name).output for name in feature_layers]
    return Model(vgg.input, outputs, name="perceptual_model")


def build_reconstruction_model(
    encoder: Model,
    generator: Model,
    img_shape: tuple,
    lr: float,
) -> Model:
    inputs = Input(shape=img_shape, name="recon_input")
    z = encoder(inputs)
    reconstructed = generator(z)
    model = Model(inputs, reconstructed, name="reconstruction_model")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
