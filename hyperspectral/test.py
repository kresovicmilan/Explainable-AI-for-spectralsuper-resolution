import scipy.io
import matplotlib.pyplot as plt
import os
import numpy as np

import sys
sys.path.insert(0, '..')
from deephyp import autoencoder
from deephyp import data
from train import *

if __name__ == '__main__':
    ref_data, ref_lam = load_ref(SPECTRAL_DATA_PATH)
    obs, obs_lam  = load_obs(OBSERVER_DATA_PATH)
    ill_A, ill_A_lam = load_ill(ILLA_PATH)
    ill_D65, ill_D65_lam = load_ill(ILLD65_PATH)

    common_lam = intersection(ref_lam, intersection(obs_lam, ill_D65_lam))
    ref_data_scaled = minmax_data(ref_data, common_lam)

    rec_ref_D65_data = np.vstack(load_data(os.path.join("spectral_reconstruction", "stored_recovered", "rec_sd_D65.dat")))

    res_D65_data = ref_data_scaled - rec_ref_D65_data
    res_D65_data_scaled = normalize_residual(res_D65_data, rec_ref_D65_data)

    hypData = data.HypImg( res_D65_data_scaled, spectralRecovered=rec_ref_D65_data)

    # pre-process data to make the model easier to train
    #hypData.pre_process( 'minmax' )

    # setup a network from a config file
    #net = autoencoder.mlp_1D_network( configFile=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_{}_sse'.format(LATENT_DIM),'config.json') )
    #net_sid = autoencoder.mlp_1D_network( configFile=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_{}_sid'.format(LATENT_DIM),'config.json') )
    #net_sse = autoencoder.mlp_1D_network( configFile=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_adv_sse','config.json') )
    net = autoencoder.mlp_1D_network( configFile=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_recovered_{}_csa'.format(LATENT_DIM),'config.json') )
    #net_sa = autoencoder.mlp_1D_network( configFile=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_adv_sa','config.json') )

    # assign previously trained parameters to the network, and name each model
    """net.add_model( addr=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_{}_sse'.format(LATENT_DIM),'best_epoch'), modelName='sse_best' )
    net.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_{}_csa'.format(LATENT_DIM), 'best_epoch'), modelName='csa_best')
    net.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_{}_sa'.format(LATENT_DIM), 'best_epoch'), modelName='sa_best')"""
    #net_sid.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_{}_sid'.format(LATENT_DIM), 'best_epoch'), modelName='sid_best')
    #net_sse.add_model( addr=os.path.join(PYTHON_FILE_PATH, 'models','test_ae_mlp_adv_sse','epoch_500'), modelName='sse_500' )
    #net_csa.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_adv_csa', 'epoch_500'), modelName='csa_500')
    #net_sa.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_adv_sa', 'epoch_500'), modelName='sa_500')
    net.add_model(addr=os.path.join(PYTHON_FILE_PATH, 'models', 'test_ae_mlp_recovered_{}_csa'.format(LATENT_DIM), 'best_epoch'), modelName='csa_best')


    # feed forward hyperspectral dataset through each encoder model (get latent encoding)
    """dataZ_sse = net.encoder( modelName='sse_best', dataSamples=hypData.spectraPrep )
    dataZ_csa = net.encoder( modelName='csa_best', dataSamples=hypData.spectraPrep )
    dataZ_sa = net.encoder( modelName='sa_best', dataSamples=hypData.spectraPrep )"""
    #dataZ_sid = net_sid.encoder( modelName='sid_best', dataSamples=hypData.spectraPrep )
    #dataZ_sse = net_sse.encoder( modelName='sse_500', dataSamples=hypData.spectraPrep )
    #dataZ_csa = net_csa.encoder(modelName='csa_500', dataSamples=hypData.spectraPrep)
    #dataZ_sa = net_sa.encoder(modelName='sa_100', dataSamples=hypData.spectraPrep)
    #dataZ_sid = net_sid.encoder(modelName='sid_500', dataSamples=hypData.spectraPrep)
    dataZ_csa = net.encoder( modelName='csa_best', dataSamples=hypData.spectra )

    # feed forward latent encoding through each decoder model (get reconstruction)
    """dataY_sse = net.decoder(modelName='sse_best', dataZ=dataZ_sse)
    dataY_csa = net.decoder(modelName='csa_best', dataZ=dataZ_csa)
    dataY_sa = net.decoder(modelName='sa_best', dataZ=dataZ_sa)"""
    #dataY_sid = net_sid.decoder(modelName='sid_best', dataZ=dataZ_sid)
    #dataY_sse = net_sse.decoder(modelName='sse_500', dataZ=dataZ_sse)
    #dataY_csa = net_csa.decoder(modelName='csa_500', dataZ=dataZ_csa)
    #dataY_sa = net_sa.decoder(modelName='sa_100', dataZ=dataZ_sa)
    #dataY_sid = net_sid.decoder(modelName='sid_500', dataZ=dataZ_sid)
    dataY_csa = net.decoder(modelName='csa_best', dataZ=dataZ_csa)


    #method = ['sid']
    #method = ['sse','csa','sa']
    method = ['csa']
    #nets = [net_sse, net_csa, net_sid]
    """dataZ_collection = [dataZ_sid]
    dataY_collection = [dataY_sid]"""
    """dataZ_collection = [dataZ_sse, dataZ_csa, dataZ_sa]
    dataY_collection = [dataY_sse, dataY_csa, dataY_sa]"""
    dataZ_collection = [dataZ_csa]
    dataY_collection = [dataY_csa]

    N_SAMPLES = hypData.numSamples
    N_TRAIN = int(round(N_SAMPLES*TRAIN_SPLIT))
    N_VAL = int(round(N_SAMPLES*VAL_SPLIT))
    N_TEST = N_SAMPLES - N_TRAIN - N_VAL
    test_sample_idx = 5
    sample_idx = N_TRAIN + N_VAL + test_sample_idx

    for j,dataZ in enumerate(dataZ_collection):

        # save a scatter plot image of 2 of 3 latent dimensions
        #idx = np.argsort(-np.std(dataZ, axis=0))
        fig, ax = plt.subplots()
        #for i,gt_class in enumerate(['asphalt', 'meadow', 'gravel','tree','painted metal','bare soil','bitumen','brick','shadow']):
        ax.plot(dataZ[sample_idx, :])
        ax.legend()
        plt.title('latent representation: %s'%(method[j]))
        plt.xlabel('dimensions')
        plt.ylabel('latent feature value')
        plt.savefig(os.path.join(PYTHON_FILE_PATH, 'results', 'results_recovered_{}'.format(LATENT_DIM), 'test_mlp_scatter_%s.png'%(method[j])))

        # save plot comparing pre-processed 'meadow' spectra input with decoder reconstruction
        fig = plt.figure()
        ax = plt.subplot(111)
        scale = np.max(hypData.spectra[sample_idx, :])/np.max(dataY_collection[j][sample_idx, :])
        ax.plot(range(hypData.numBands),hypData.spectra[sample_idx, :],label='input')
        ax.plot(range(hypData.numBands),res_D65_data[sample_idx, :],label='unnormalized input')
        ax.plot(range(hypData.numBands),dataY_collection[j][sample_idx, :]*scale,label='reconstruction')
        plt.xlabel('band')
        plt.ylabel('value')
        ax.legend()
        plt.savefig(os.path.join(PYTHON_FILE_PATH, 'results', 'results_recovered_{}'.format(LATENT_DIM), 'test_mlp_InputVsReconstruct_%s.png'%(method[j])))