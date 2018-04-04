#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

	Code to train model, LSTM and two branch LSTM into feedforward.
	I tried to clean it up from the original script so that it is least confusing.
	
	Author: Tuka Alhanai, CSAIL MIT April 4th 2018

"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.utils import class_weight
import statsmodels.api as sm
from sklearn.isotonic import IsotonicRegression as IR
import csv
from scipy import signal
from scipy.stats import kurtosis, skew, spearmanr
import pickle
from sklearn import preprocessing
import pprint

from keras.models import Sequential, model_from_json, Model
from keras.layers import LSTM, Dense, Flatten, Dropout, Bidirectional, concatenate, Input
from keras.utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, LearningRateScheduler, \
    EarlyStopping, ReduceLROnPlateau, Callback
import keras.backend as K


# Let the user know if they are not using a GPU
if K.tensorflow_backend._get_available_gpus() == []:
    print('YOU ARE NOT USING A GPU ON THIS DEVICE! EXITING!')
    # sys.exit()

# ============================================================================================
# Train LSTM Model
# ============================================================================================
def trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams):
	"""
		Method to train LSTM model.

		X_{train,dev}: should be [Nexamples, Ntimesteps, Nfeatures]
		Y_{train,dev}: is a vector of binary outcomes
		R_{train,dev}: is the subject ID, useful for later when calculating performance at the subject level
		hyperparams:   is a dict

	"""

    # seed generator
    np.random.seed(1337)

    # grab hyperparamters
    exp         = hyperparams['exp']
    batch_size  = hyperparams['batchsize']
    epochs      = hyperparams['epochs']
    lr          = hyperparams['lr']
    hsize       = hyperparams['hsize']
    nlayers     = hyperparams['nlayers']
    loss        = hyperparams['loss']
    dirpath     = hyperparams['dirpath']
    momentum    = hyperparams['momentum']
    decay       = hyperparams['decay']
    dropout     = hyperparams['dropout']
    dropout_rec = hyperparams['dropout_rec']
    merge_mode  = hyperparams['merge_mode']
    layertype   = hyperparams['layertype']
    balClass    = hyperparams['balClass']
    act_output  = hyperparams['act_output']


    # grab input dimension and number of timesteps (i.e. number of interview sequences)
    dim = X_train.shape[2]
    timesteps = X_train.shape[1]

    # balance training classess
    if balClass:
        cweight = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    else:
        cweight = np.array([1, 1])

    # expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()


    if layertype == 'lstm':
        if nlayers == 1:
            model.add(LSTM(hsize, return_sequences=False, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(LSTM(hsize, return_sequences=True,   input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(LSTM(hsize, return_sequences=True,  input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(LSTM(hsize, return_sequences=True, input_shape=(timesteps, dim), recurrent_dropout=dropout_rec, dropout=dropout))
            model.add(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,))
            model.add(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec))
            model.add(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec))
            # model.add(Dense(dsize, activation=act_output))

    elif layertype == 'bi-lstm':
        if nlayers == 1:
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec,
                           dropout=dropout), input_shape=(timesteps, dim), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 2:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec), merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 3:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False,recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

        if nlayers == 4:
            model.add(Bidirectional(LSTM(hsize, return_sequences=True, recurrent_dropout=dropout_rec,
                           dropout=dropout),input_shape=(timesteps, dim), merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=True,  recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            model.add(Bidirectional(LSTM(hsize, return_sequences=False, recurrent_dropout=dropout_rec),merge_mode=merge_mode))
            # model.add(Flatten())
            # model.add(Dense(dsize, activation=act_output))

    # add the final output node
    # this check is useful if training a multiclass model
    if act_output == 'sigmoid':
        dsize = 1
        model.add(Dense(dsize, activation=act_output))

    elif act_output == 'softmax':
        dsize = 27
        model.add(Dense(dsize, activation=act_output))
        Y_train = to_categorical(R_train, num_classes=27)
        Y_dev = to_categorical(R_dev, num_classes=27)

    elif act_output == 'relu':
        dsize = 1
        def myrelu(x):
            return (K.relu(x, alpha=0.0, max_value=27))
        model.add(Dense(dsize, activation=myrelu))
        Y_train = R_train
        Y_dev = R_dev


    # print info on network
    print(model.summary())
    print('--- network has layers:', nlayers, ' hsize:',hsize, ' bsize:', batch_size,
          ' lr:', lr, ' epochs:', epochs, ' loss:', loss, ' act_o:', act_output)

    # define optimizer
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    # compile model
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy','mae','mse'])

    # defining callbacks - creating directory to dump files
    dirpath = dirpath + str(exp)
    os.system('mkdir ' + dirpath)

    # serialize model to JSON
    model_json = model.to_json()
    with open(dirpath + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # checkpoints
    # filepaths to checkpoints
    filepath_best       = dirpath + "/weights-best.hdf5"
    filepath_epochs     = dirpath + "/weights-{epoch:02d}-{loss:.2f}.hdf5"

    # log best model
    checkpoint_best     = ModelCheckpoint(filepath_best,   monitor='loss', verbose=0, save_best_only=True, mode='auto')

    # log improved model
    checkpoint_epochs   = ModelCheckpoint(filepath_epochs, monitor='loss', verbose=0, save_best_only=True, mode='auto')
    
    # log results to csv file
    csv_logger          = CSVLogger(dirpath + '/training.log')
    
    # loss_history        = LossHistory()
    # lrate               = LearningRateScheduler()
    
    # update decay as a function of epoch and lr
    lr_decay            = lr_decay_callback(lr, decay)

    # early stopping criterion
    early_stop          = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
    # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)

    # log files to plot via tensorboard
    tensorboard         = TensorBoard(log_dir=dirpath + '/logs', histogram_freq=0, write_graph=True, write_images=False)
    
    #calculate custom performance metric
    perf                = Metrics()

    # these are the callbacks we care for
    callbacks_list      = [checkpoint_best, checkpoint_epochs, early_stop, lr_decay, tensorboard, csv_logger]

    # train model
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_dev, Y_dev),
              class_weight=cweight,
              callbacks=callbacks_list)

    # load best model and evaluate
    model.load_weights(filepath=filepath_best)
    
    # gotta compile it
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # return predictions of best model
    pred        = model.predict(X_dev,   batch_size=None, verbose=0, steps=None)
    pred_train  = model.predict(X_train, batch_size=None, verbose=0, steps=None)

    return pred, pred_train


# defines step decay
# =================================================
def lr_decay_callback(lr_init, lr_decay):
    def step_decay(epoch):
        return lr_init * (lr_decay ** (epoch + 1))
    return LearningRateScheduler(step_decay)


# prints additional metrics to log file
# =================================================
class Metrics(Callback):

	# log performance on every epoch
    def on_epoch_end(self, epoch, logs):

    	# checking if more than one validation data is being used
        try:
            pred = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
            targ = self.validation_data[2]
        except:
            pred = np.asarray(self.model.predict(self.validation_data[0]))
            targ = self.validation_data[1]

        # calculate f1 score
        logs['val_f1'] = metrics.f1_score(targ, np.round(pred), pos_label=1)

        return

# ============================================================================================
# train DNN model that combines audio and doc LSTM branches.
# ============================================================================================
def trainHierarchy(X_train_fuse, Y_train, X_dev_fuse, Y_dev, R_train, R_dev, hyperparams):

		"""
		Method to train LSTM model.

		X_{train,dev}_fuse: should be [Nexamples, Nfeatures]
		Y_{train,dev}: 		is a vector of binary outcomes
		R_{train,dev}: 		is the subject ID, useful for later when calculating performance at the subject level
		hyperparams:   		is a dict

	"""

	# init random seed
	np.random.seed(1337)

	# number of features
    dim = X_train_fuse.shape[1]

    # hyperparameters
    loss = hyperparams['loss']
    lr = hyperparams['lr']
    momentum = hyperparams['momentum']
    batch_size = hyperparams['batchsize']
    dsize = hyperparams['dsize']
    epochs = hyperparams['epochs']
    decay = hyperparams['decay']
    act = hyperparams['act']
    nlayers = hyperparams['nlayers']
    dropout = hyperparams['dropout']
    exppath = hyperparams['exppath']
    act_output = hyperparams['act_output']

    # define input
    input = Input(shape=(dim,))

    # define number of DNN layers
    if nlayers == 1:
        final = Dense(dsize, activation=act)(input)
        final = Dropout(dropout)(final)

    if nlayers == 2:
        final = Dense(dsize, activation=act)(input)
        final = Dropout(dropout)(final)
        final = Dense(dsize, activation=act)(final)
        final = Dropout(dropout)(final)


    if nlayers == 3:
        final = Dense(dsize, activation=act)(input)
        final = Dropout(dropout)(final)
        final = Dense(dsize, activation=act)(final)
        final = Dropout(dropout)(final)
        final = Dense(dsize, activation=act)(final)
        final = Dropout(dropout)(final)

    if nlayers == 4:
        final = Dense(dsize, activation=act)(input)
        final = Dropout(dropout)(final)
        final = Dense(dsize, activation=act)(final)
        final = Dropout(dropout)(final)
        final = Dense(dsize, activation=act)(final)
        final = Dropout(dropout)(final)

    # add final output node
	final = Dense(1, activation='sigmoid')(final)

	# define model
    model = Model(inputs=input, outputs=final)

    # print summary
    print(model.summary())
    print('--- network has layers:', nlayers, 'dsize:', dsize, 'bsize:', batch_size, 'lr:', lr, 'epochs:',
          epochs)


    # defining files to save
    # dirpath = dirpath + str(exp)
    os.system('mkdir ' + exppath)

    # serialize model to JSON
    model_json = model.to_json()
    with open(exppath + "/model.json", "w") as json_file:
        json_file.write(model_json)

    # define optimizer
    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    # compile model
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # filepaths to checkpoints
    filepath_best = exppath + "/weights-best.hdf5"
    filepath_epochs = exppath + "/weights-{epoch:02d}-{loss:.2f}.hdf5"

    # save best model
    checkpoint_best = ModelCheckpoint(filepath_best, monitor='loss', verbose=0, save_best_only=True, mode='auto')
    
    # save improved model
    checkpoint_epochs = ModelCheckpoint(filepath_epochs, monitor='loss', verbose=0, save_best_only=True, mode='auto')
    
    # log performance to csv file
    csv_logger = CSVLogger(exppath + '/training.log')
    # loss_history        = LossHistory()
    # lrate               = LearningRateScheduler()

    # update decay as function of epoch and lr
    lr_decay = lr_decay_callback(lr, decay)

    # define early stopping criterion
    early_stop = EarlyStopping(monitor='loss', min_delta=1e-04, patience=25, verbose=0, mode='auto')
    # reduce_lr         = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=5, min_lr=0.0001)
    
    # log data to view via tensorboard
    tensorboard = TensorBoard(log_dir=exppath + '/logs', histogram_freq=0, write_graph=True, write_images=False)
    
    # define metrics
    perf = Metrics()

    # callbacks we are interested in
    callbacks_list = [checkpoint_best, checkpoint_epochs, early_stop, lr_decay, perf, tensorboard, csv_logger]

    # train model
    model.fit(X_train_fuse, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_dev_fuse, Y_dev),
              callbacks=callbacks_list)

    # load best model and evaluate
    model.load_weights(filepath=filepath_best)
    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])

    # return predictions of best model
    pred_train = model.predict(X_train_fuse, batch_size=None, verbose=0, steps=None)
    pred = model.predict(X_dev_fuse, batch_size=None, verbose=0, steps=None)

    return pred, pred_train



# ============================================================================================
# Combinding Features
# ============================================================================================
def combineFeats():
	"""
		Combining audio and doc features.
	"""


	# PROCESSING AUDIO
    # ===============================
    hyperparams = {'exp': 20, 'timesteps': 30, 'stride': 1, 'lr': 9.9999999999999995e-07, 'nlayers': 3, 'hsize': 128, 'batchsize': 128, 'epochs': 300, 'momentum': 0.80000000000000004, 'decay': 0.98999999999999999, 'dropout': 0.20000000000000001, 'dropout_rec': 0.20000000000000001, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'mul', 'dirpath': 'data/LSTM_10-audio/', 'exppath': 'data/LSTM_10-audio/20/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']
    # text = 'data/Step10/alltext.txt'

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadAudio()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers-1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train = np.squeeze(_layer2([0] + [X_train]))
    acts_dev = np.squeeze(_layer2([0] + [X_dev]))
    print('--- got activations')

    # PROCESSING DOCS
    # ===============================
    hyperparams = {'exp': 330, 'timesteps': 7, 'stride': 3, 'lr': 0.10000000000000001, 'nlayers': 2, 'hsize': 4, 'batchsize': 64, 'epochs': 300, 'momentum': 0.84999999999999998, 'decay': 1.0, 'dropout': 0.10000000000000001, 'dropout_rec': 0.80000000000000004, 'loss': 'binary_crossentropy', 'dim': 100, 'min_count': 3, 'window': 3, 'wepochs': 25, 'layertype': 'bi-lstm', 'merge_mode': 'concat', 'dirpath': 'data/LSTM_10/', 'exppath': 'data/LSTM_10/330/', 'text': 'data/Step10/alltext.txt', 'balClass': False}
    exppath = hyperparams['exppath']

    # load model
    with open(exppath + "/model.json", "r") as json_file:
        model_json = json_file.read()
    try:
        model = model_from_json(model_json)
    except:
        model = model_from_json(model_json, custom_objects={'myrelu':myrelu})

    lr = hyperparams['lr']
    loss = hyperparams['loss']
    momentum = hyperparams['momentum']
    nlayers = hyperparams['nlayers']

    # load best model and evaluate
    filepath_best = exppath + "/weights-best.hdf5"
    model.load_weights(filepath=filepath_best)
    print('--- load weights')

    sgd = optimizers.SGD(lr=lr, momentum=momentum, decay=0, nesterov=True)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=['accuracy'])
    print('--- compile model')

    # load data
    X_train_doc, Y_train, X_dev_doc, Y_dev, R_train_doc, R_dev_doc = loadDoc()
    print('--- load data')

    # getting activations from final layer
    layer = model.layers[nlayers - 1]
    inputs = [K.learning_phase()] + model.inputs
    _layer2 = K.function(inputs, [layer.output])
    acts_train_doc = np.squeeze(_layer2([0] + [X_train_doc]))
    acts_dev_doc   = np.squeeze(_layer2([0] + [X_dev_doc]))
    print('--- got activations')

    # FUSE EMBEDDINGS
    # ============================
    acts_train_doc_pad = []
    for idx, subj in enumerate(np.unique(S_train)):
        index = np.where(S_train == subj)[0]
        j = 0
        indexpad = np.where(S_train_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_train_doc_pad.append(acts_train_doc[indexpad[j],:])

    acts_dev_doc_pad = []
    for idx, subj in enumerate(np.unique(S_dev)):
        index = np.where(S_dev == subj)[0]
        j = 0
        indexpad = np.where(S_dev_doc == subj)[0]
        for i,_ in enumerate(index):
            # print(i)
            if i%4 == 0 and i > 0 and j < indexpad.shape[0]-1:
                j = j+1
            acts_dev_doc_pad.append(acts_dev_doc[indexpad[j],:])

        # CMVN
        # scaler = preprocessing.StandardScaler().fit(np.asarray(acts_train_doc_pad))
        # acts_train_doc_pad = scaler.transform(np.asarray(acts_train_doc_pad))
        # acts_dev_doc_pad = scaler.transform(np.asarray(acts_dev_doc_pad))

        X_train_fuse = np.hstack((np.asarray(acts_train_doc_pad),acts_train))
        X_dev_fuse = np.hstack((np.asarray(acts_dev_doc_pad),acts_dev))

        # optional
        np.save('data/fuse/X_train.npy', X_train_fuse)
        np.save('data/fuse/features/X_dev.npy', X_dev_fuse)
        np.save('data/fuse/features/Y_train.npy', Y_train)
        np.save('data/fuse/features/Y_dev.npy', Y_dev)
        np.save('data/fuse/features/S_train.npy', S_train)
        np.save('data/fuse/features/S_dev.npy', S_dev)
        np.save('data/fuse/features/R_train.npy', R_train)
        np.save('data/fuse/features/R_dev.npy', R_dev)


# ============================================================================================
# Data Loading
# ============================================================================================
# you will need to point to your data directory
def loadAudio():

	X_train, Y_train = np.load('data/audio/X_train.npy'), np.load('data/audio/Y_train.npy')
	X_dev, Y_dev = np.load('data/audio/X_dev.npy'), np.load('data/audio/Y_dev.npy')
	R_train, R_dev = np.load('data/audio/R_dev.npy'), np.load('data/audio/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev


def loadDoc():

	X_train, Y_train = np.load('data/doc/X_train.npy'), np.load('data/doc/Y_train.npy')
	X_dev, Y_dev = np.load('data/doc/X_dev.npy'), np.load('data/doc/Y_dev.npy')
	R_train, R_dev = np.load('data/doc/R_dev.npy'), np.load('data/doc/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev


def loadFuse():

	X_train, Y_train = np.load('data/fuse/X_train.npy'), np.load('data/fuse/Y_train.npy')
	X_dev, Y_dev = np.load('data/fuse/X_dev.npy'), np.load('data/fuse/Y_dev.npy')
	R_train, R_dev = np.load('data/fuse/R_dev.npy'), np.load('data/fuse/R_dev.npy')

	return X_train, Y_train, X_dev, Y_dev, R_train, R_dev


# ============================================================================================
# main script
# ============================================================================================
if __name__ == "__main__":


	# 1. load the data for audio
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadAudio()

	# 2. train lstm model
	pred_audio, pred_train_audio = trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams)

	# 1b. load the doc data
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadDoc()

	# 2b. train lstm model for doc data
	pred_audio, pred_train_audio = trainLSTM(X_train, Y_train, X_dev, Y_dev, R_train, R_dev, hyperparams)

	# 3. concatenate last layer features for each audio and doc branch.
	combineFeats()
	X_train, Y_train, X_dev, Y_dev, R_train, R_dev = loadFuse()

	# 4. train feedforward.
	# hyperparams can be different (e.g. learning rate, decay, momentum, etc.)
	pred, pred_train = trainHierarchy(X_train_fuse, Y_train, X_dev_fuse, Y_dev, hyperparams)

	# 5. evaluate performance
	f1 = metrics.f1_score(Y_dev, np.round(pred), pos_label=1)


# eof