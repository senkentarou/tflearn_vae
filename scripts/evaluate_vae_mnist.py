# coding: utf-8

import numpy as np
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist
from skimage import io

import sys
import os

SCRIPT_NAME = "vae_mnist"
HOME_DIR = ".."  # os.path.expanduser("~")
MODEL_DIR = "%s/out_models/%s" % (HOME_DIR, SCRIPT_NAME)
MNIST_DIR = "%s/datasets/mnist" % HOME_DIR
RESULT_DIR = "%s/results" % HOME_DIR

LOG_DIR = "%s/logs" % MODEL_DIR
TENSORBOARD_DIR = "%s/models" % MODEL_DIR
CHECKPOINT_DIR = "%s/models/model" % MODEL_DIR

VAE_SCOPE = "VAE"
E_HIDDEN_SIZE = 256
E_OUTPUT_SIZE = 10
D_HIDDEN_SIZE = 256
D_OUTPUT_SIZE = 784

X_SIZE = 784
LATENT_SIZE = 10

SAMPLE_NUM = 1000
BATCH_SIZE = 32

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
if not os.path.exists(TENSORBOARD_DIR):
    os.makedirs(TENSORBOARD_DIR)
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(MNIST_DIR):
    os.makedirs(MNIST_DIR)
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

def build_encoder(inputs, scope="E", reuse=False):
    
    """
     Encoder 構築
    """
    
    e_h = tflearn.fully_connected(inputs,
                                  E_HIDDEN_SIZE,
                                  activation="relu",
                                  reuse=reuse,
                                  scope="%s/e_hidden" % scope)
    
    mu = tflearn.fully_connected(e_h,
                                 E_OUTPUT_SIZE,
                                 activation="linear",
                                 reuse=reuse,
                                 scope="%s/mu" % scope)

    log_var = tflearn.fully_connected(e_h,
                                      E_OUTPUT_SIZE,
                                      activation="linear",
                                      reuse=reuse,
                                      scope="%s/log_var" % scope)

    return mu, log_var


def build_decoder(input_z, scope="D", reuse=False):

    """
     Decoder 構築
    """

    d_i = tflearn.fully_connected(input_z,
                                  LATENT_SIZE,
                                  activation="linear",
                                  reuse=reuse,
                                  scope="%s/z" % scope)

    d_h = tflearn.fully_connected(d_i,
                                  D_HIDDEN_SIZE,
                                  activation="relu",
                                  reuse=reuse,
                                  scope="%s/d_hidden" % scope)

    d_o = tflearn.fully_connected(d_h,
                                  D_OUTPUT_SIZE,
                                  activation="linear",
                                  reuse=reuse,
                                  scope="%s/d_output" % scope)

    return d_o


def sample_z(mu, log_var):

    """
     潜在変数zを抽出
     計算式: z = mu + std @ epsilon (@: 要素ごとの積)
    """

    epsilon = tf.random_normal(tf.shape(log_var), dtype=tf.float32)
    std = tf.exp(tf.multiply(0.5, log_var))
    z = tf.add(mu, tf.multiply(std, epsilon))
    
    return z


def build_vae_trainer():

    target = None

    # Place Holder
    input_x = tflearn.input_data(shape=(None, X_SIZE), name="input_x")

    # Encoder
    mu, log_var = build_encoder(input_x, scope=VAE_SCOPE, reuse=False)

    # Sampling z
    z = sample_z(mu, log_var)

    # Decoder
    output_x = build_decoder(z, scope=VAE_SCOPE, reuse=False)
    target = output_x

    # Loss
    dkl = -0.5 * tf.reduce_sum(1 + log_var - tf.square(mu) - tf.exp(log_var), reduction_indices=1)
    mse = tf.reduce_sum(tf.squared_difference(input_x, output_x), reduction_indices=1)
    loss = tf.reduce_mean(tf.add(dkl, mse))

    # Optimizer
    opt = tflearn.Adam(learning_rate=0.001).get_tensor()

    # Trainable Variables
    train_vars = get_trainable_variables(VAE_SCOPE)

    # TrainOp
    train_op = tflearn.TrainOp(loss=loss,
                               optimizer=opt,
                               batch_size=BATCH_SIZE,
                               trainable_vars=train_vars,
                               name="VAE")

    # Trainer
    vae_trainer = tflearn.Trainer(train_op,
                                  tensorboard_dir=TENSORBOARD_DIR,
                                  max_checkpoints=1)

    return vae_trainer, target


def evaluate_vae(x):

    # Train
    with tf.Graph().as_default():

        vae_trainer, target = build_vae_trainer()

        input_x = get_input_tensor_by_name("input_x")
        feed_dict = {input_x: x}

        # Load
        vae_trainer.restore(CHECKPOINT_DIR)

        # Evaluate
        evaluator = build_evaluator(vae_trainer, target)
        pred_x = evaluator.predict(feed_dict)

        # Output
        sample_mnist_image(x, pred_x)


def build_evaluator(trainer, target):

    evaluator = tflearn.Evaluator(target,
                                  session=trainer.session)

    return evaluator


def sample_mnist_image(x, pred_x):

        img = np.ndarray(shape=(28*10, 28*10))
        for i in range(50):
            row = i // 10 * 2
            col = i % 10
            img[28*row : 28*(row+1), 28*col : 28*(col+1)] = np.reshape(x[i], (28, 28))
            img[28*(row+1) : 28*(row+2), 28*col : 28*(col+1)] = np.reshape(pred_x[i], (28, 28))
        img[img > 1] = 1
        img[img < 0] = 0
        img *= 255
        img = img.astype(np.uint8)
        io.imsave("%s/result.png" % RESULT_DIR, img)


def get_trainable_variables(scope):
    return [v for v in tflearn.get_all_trainable_variable()
            if scope + '/' in v.name]


def get_input_tensor_by_name(name):
    return tf.get_collection(tf.GraphKeys.INPUTS, scope=name)[0]


if __name__ == "__main__":

    print("%s: start" % SCRIPT_NAME)

    # 目的: MNIST文字生成
    X, Y, testX, testY = mnist.load_data(data_dir=MNIST_DIR)
    
    # reshape
    sample_X = X[:SAMPLE_NUM]

    # Train
    evaluate_vae(sample_X)

    print("%s: done" % SCRIPT_NAME)
