import numpy as np
import scipy.io
from colour import (MSDS_CMFS, SDS_ILLUMINANTS, SpectralDistribution, SpectralShape, sd_to_XYZ, XYZ_to_sd, STANDARD_OBSERVERS_CMFS, CCS_ILLUMINANTS, XYZ_to_Lab, Lab_to_XYZ, sd_to_XYZ_integration)
from common_fun import *
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

SPECTRAL_DATA_PATH = "SOCS.mat"
OBSERVER_DATA_PATH = "CIEobs.mat"
ILLA_PATH = "IllA.mat"
ILLD65_PATH = "IllD65.mat"

def common_ref(ref, ref_lam, ill_lam, obs_lam, cmfs, illuminant, name):
    common_lam = intersection(ref_lam, intersection(obs_lam, ill_lam))
    ref = ref[:,np.where(np.not_equal(ismember(ref_lam,common_lam), None))[0]]
    xyz_new_list = []
    for idx, r in enumerate(ref):
        if idx % 10000 == 0:
            print(idx)
        sd = SpectralDistribution(r, common_lam)
        xyz_new_list.append(sd_to_XYZ(sd, cmfs, illuminant))
    
    save_data(xyz_new_list, name)

def save_data(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)

def load_data(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def xyz_to_lab(xyz, illuminant):
    return XYZ_to_Lab(xyz, illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant])

def lab_to_xyz(lab, illuminant):
    return Lab_to_XYZ(lab, illuminant=CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant])

def get_intersection_of_recovered_ref(recovered_ref, common_lam):
    return [ref for ref, lam in zip(recovered_ref.values, recovered_ref.domain) for cmm_lam in common_lam if int(lam) == cmm_lam]

def true_residual(res, ref):
    return 2*(res - max(ref))

def minmax_data(ref_data, common_lam):
    ref_data_scaled = ref_data - np.transpose(np.tile(np.min(ref_data, axis=1)-(1e-3),(len(common_lam),1)))
    ref_data_scaled = ref_data_scaled / np.transpose(np.tile(np.max(ref_data, axis=1), (len(common_lam), 1)))
    return ref_data_scaled
    
if __name__ == '__main__':
    ref_data, ref_lam = load_ref(SPECTRAL_DATA_PATH)
    obs, obs_lam  = load_obs(OBSERVER_DATA_PATH)
    ill_A, ill_A_lam = load_ill(ILLA_PATH)
    ill_D65, ill_D65_lam = load_ill(ILLD65_PATH)

    cmfs = (MSDS_CMFS['CIE 1931 2 Degree Standard Observer'].copy().align(SpectralShape(360, 780, 10)))
    illuminant_D65 = SDS_ILLUMINANTS['D65'].copy().align(cmfs.shape)
    common_lam = intersection(ref_lam, intersection(obs_lam, ill_D65_lam))

    ref_data_inter = ref_data[:,np.where(np.not_equal(ismember(ref_lam,common_lam), None))[0]]
    ref_data_scaled = minmax_data(ref_data_inter, common_lam)
    scaled_recovered_refs = []
    for true_ref in ref_data_scaled:
        sd = SpectralDistribution(true_ref, common_lam)
        xyz = sd_to_XYZ(sd, cmfs=cmfs, illuminant=illuminant_D65)

        recovered_ref = XYZ_to_sd(xyz, method='Mallett 2019', cmfs=cmfs)
        inter_recovered_ref = get_intersection_of_recovered_ref(recovered_ref, common_lam)

        scaled_recovered_ref = minmax_scale(inter_recovered_ref, feature_range=(0, max(true_ref)))
        scaled_recovered_refs.append(scaled_recovered_ref)

    save_data(scaled_recovered_refs, os.path.join("spectral_reconstruction", "stored_recovered", "rec_sd_D65.dat"))


    
