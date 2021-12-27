#import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow import keras

# import toolbox libraries
import sys
sys.path.insert(0, '..')
from deephyp import autoencoder
from deephyp import data
from common_fun import *
import pickle

PYTHON_FILE_PATH = "spectral_reconstruction"
LOGDIR = os.path.join(PYTHON_FILE_PATH, "logs")
SPECTRAL_DATA_PATH = "SOCS.mat"
OBSERVER_DATA_PATH = "CIEobs.mat"
ILLA_PATH = "IllA.mat"
ILLD65_PATH = "IllD65.mat"
N_SAMPLES = 0
N_TRAIN = 0
N_VAL = 0
N_TEST = 0
TRAIN_LOG_DIR = "logs"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.15
TEST_SPLIT = 1 - (TRAIN_SPLIT +  VAL_SPLIT)
LATENT_DIM = 5

"""def load_ref(spectral_data_path = "SOCS.mat"):
    gamut = scipy.io.loadmat(spectral_data_path)
    gamut = np.transpose(gamut["SOCS"])

    # adding 1: to not include the nanometers
    ref_data = gamut[np.where(np.sum(gamut,axis=1)>1)[0][1:],:]
    ref_lam = range(400,710,10)

    return ref_data, ref_lam"""

def split_data(hypData, train_split = 0.8, val_split = 0.15):
    global N_SAMPLES, N_TRAIN, N_VAL, N_TEST
    N_SAMPLES = hypData.numSamples
    N_TRAIN = int(round(N_SAMPLES*train_split))
    N_VAL = int(round(N_SAMPLES*val_split))
    N_TEST = N_SAMPLES - N_TRAIN - N_VAL
    if hypData.spectralRecovered is not None:
        return hypData.spectra[:N_TRAIN, :], hypData.spectra[N_TRAIN:N_TRAIN+N_VAL, :], hypData.spectra[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST, :], hypData.spectralRecovered[:N_TRAIN, :], hypData.spectralRecovered[N_TRAIN:N_TRAIN+N_VAL, :], hypData.spectralRecovered[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST, :]
    else:
        return hypData.spectraPrep[:N_TRAIN, :], hypData.spectraPrep[N_TRAIN:N_TRAIN+N_VAL, :], hypData.spectraPrep[N_TRAIN+N_VAL:N_TRAIN+N_VAL+N_TEST, :]

def construct_mlp_network(encoderSize=[50, 30, 2], activationFunc = 'relu', activationFuncFinal = 'linear', train_op_name = 'csa', lossFunc = 'CSA', learning_rate = 1e-3, method = 'Adam'):
    # setup a fully-connected autoencoder neural network with 3 encoder layers
    net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=encoderSize, activationFunc=activationFunc,
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False, activationFuncFinal = activationFuncFinal)
    
    # setup a training operation for the network
    net.add_train_op( name=train_op_name, lossFunc=lossFunc, learning_rate=learning_rate, decay_steps=None, decay_rate=None,
                      method=method, wd_lambda=0.0 )
    return net

def create_model_dir(new_dir_name):
    model_dir = os.path.join(PYTHON_FILE_PATH, 'models',new_dir_name)
    if os.path.exists(model_dir):
        # if directory already exists, delete it
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    return model_dir

# Min max scaling of the reflectance data over the whole dataset
def minmax_data(ref_data, common_lam):
    ref_data_scaled = ref_data - np.transpose(np.tile(np.min(ref_data, axis=1)-(1e-3),(len(common_lam),1)))
    ref_data_scaled = ref_data_scaled / np.transpose(np.tile(np.max(ref_data, axis=1), (len(common_lam), 1)))
    return ref_data_scaled

# Load recovered reflectances
def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def get_true_residual_data(res, ref):
    return 2*(res - max(ref))

def normalize_residual(res, ref):
    max_values = np.expand_dims(np.array(np.transpose(np.amax(ref, 1))), axis=1)
    return (res + max_values)/2

if __name__ == '__main__':

    ref_data, ref_lam = load_ref(SPECTRAL_DATA_PATH)
    obs, obs_lam  = load_obs(OBSERVER_DATA_PATH)
    ill_A, ill_A_lam = load_ill(ILLA_PATH)
    ill_D65, ill_D65_lam = load_ill(ILLD65_PATH)

    # interesct wavelengths so that all data has the same range
    #ref_A, ill_A, obs_A, xyz_A, M_A = ref2xyz(ref_data, ref_lam, ill_A, ill_A_lam, obs, obs_lam)
    #ref_D65, ill_D65, obs_D65, xyz_D65, M_D65 = ref2xyz(ref_data, ref_lam, ill_D65, ill_D65_lam, obs, obs_lam)

    common_lam = intersection(ref_lam, intersection(obs_lam, ill_D65_lam))
    ref_data_scaled = minmax_data(ref_data, common_lam)

    rec_ref_D65_data = np.vstack(load_data(os.path.join("spectral_reconstruction", "stored_recovered", "rec_sd_D65.dat")))

    res_D65_data = ref_data_scaled - rec_ref_D65_data
    #res_D65_data_scaled = normalize_residual(res_D65_data, rec_ref_D65_data)
    res_D65_data_scaled = res_D65_data

    hypData = data.HypImg( res_D65_data_scaled, spectralRecovered=rec_ref_D65_data)
    
    # pre-process data to make the model easier to train
    #hypData.pre_process( 'minmax' )

    train_set, val_set, test_set, train_set_rec, val_set_rec, test_set_rec = split_data(hypData)

    dataTrain = data.Iterator(dataSamples=train_set, targets=train_set_rec, batchSize=1000)
    dataVal = data.Iterator(dataSamples=val_set, targets=val_set_rec)

    # shuffle training data
    dataTrain.shuffle()

    # setup tensorboard
    #train_summary_writer = tf.summary.FileWriter(TRAIN_LOG_DIR)
    #create_file_writer(TRAIN_LOG_DIR)

    net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50, 30, LATENT_DIM], activationFunc='relu',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False, isRecovered=True )

    # setup multiple training operations for the network (with different loss functions)
    net.add_train_op(name='sse', lossFunc='SSE', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     method='Adam', wd_lambda=0.0)

    net.add_train_op( name='csa', lossFunc='CSA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )

    net.add_train_op(name='sa', lossFunc='SA', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                     method='Adam', wd_lambda=0.0)

    
    # setup a fully-connected autoencoder neural network with 3 encoder layers
    """net = autoencoder.mlp_1D_network( inputSize=hypData.numBands, encoderSize=[50,30, LATENT_DIM], activationFunc='sigmoid',
                                      weightInitOpt='truncated_normal', tiedWeights=None, skipConnect=False,
                                      activationFuncFinal='sigmoid')

    # setup a training operation for the network
    net.add_train_op( name='sid', lossFunc='SID', learning_rate=1e-3, decay_steps=None, decay_rate=None,
                      method='Adam', wd_lambda=0.0 )"""
    
    methods = ['csa']
    #methods = ['sse','csa','sa']
    for method in methods:

        model_dir = create_model_dir("test_ae_mlp_recovered_unnormalized_{}_{}".format(LATENT_DIM, method))
        """if method == 'sid':
            activationFunc = 'sigmoid'
            activationFuncFinal = 'sigmoid'
        else:
            activationFunc = 'relu'
            activationFuncFinal = 'linear'"""
        
        #net = construct_mlp_network(encoderSize=[50, 30, 2], activationFunc = activationFunc, activationFuncFinal = activationFuncFinal, train_op_name = method, lossFunc = method.upper(), learning_rate = 1e-3, method = 'Adam')

        # train a model for each training op
        dataTrain.reset_batch()
        net.train(dataTrain=dataTrain, dataVal=dataVal, train_op_name=method, n_epochs=2000, save_addr=model_dir,
                  visualiseRateTrain=10, visualiseRateVal=10)



    