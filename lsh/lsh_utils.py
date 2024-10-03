import pickle
from loguru import logger
import os
import numpy as np
from scipy import interpolate, integrate

def fetch_gaussian_hyperplanes(num_planes, latent_dim):
    """
        latent_dim: dimension of the latent space where random normal hyperplanes are generated
        num_planes: no of hyperplanes to generate
        Load/generate specified no of gaussian hyperplanes
    """
    logger.info('Fetching gaussian hplanes ...')
    fp = f'./data/gauss_hplanes_numPlanes{num_planes}_latentDim{latent_dim}.pkl'
    if os.path.exists(fp):
        hplanes = pickle.load(open(fp,'rb'))
    else:
        hplanes = np.random.normal(size=(latent_dim, num_planes))
        pickle.dump(hplanes, open(fp,"wb"))
    logger.info("Loading random hyperplanes from %s", fp)
    return hplanes

def fetch_omega_samples(conf):
    """
        fetch samples of  w given T, a, b
    """
    num_samples =  conf.dataset.embed_dim * conf.hashing.m_use #* 4
    logger.info('Fetching samples ...')
    samples_fp = "./allPklDumps/samples_num" + str(num_samples) + "_a"+ str(conf.hashing.a) +\
                        "_b" + str(conf.hashing.b) + "_T" + str(conf.hashing.T) + ".pkl"
    if os.path.exists(samples_fp):
        all_d = pickle.load(open(samples_fp,"rb"))
        logger.info(f"Fetching samples from  {samples_fp}")
    else:
        logger.info(f"Samples not found, so generating samples and dumping to {samples_fp}")
        all_d = generate_samples(num_samples, conf.hashing.T, conf.hashing.a, conf.hashing.b)
        pickle.dump(all_d, open(samples_fp, "wb"))
    logger.info('Samples fetched')
    return np.float32(all_d['samples']), np.float32(all_d['pdf'])


def get_pdf_normalization(T, a1, b1):
    N = integrate.quad(pdf, a1, b1, points=0, limit=10000, args=T)[0]
    return N

def pdf(w, T):

    R_G = 2 * (np.sin(w*T/2))**2 / w**2 + T * np.sin(w*T) / w
    I_G = np.sin(w*T) / w**2 - T * np.cos(w*T) / w
    return  (np.abs(R_G) + np.abs(I_G))

def generate_samples(num_samples, T, a1, b1):
    x1 = np.linspace(a1,b1,100000)
    y1 = pdf(x1,T)
    cdf_y = np.cumsum(y1)
    cdf_y = cdf_y/cdf_y.max()
    inverse_cdf = interpolate.interp1d(cdf_y,x1)
    N = get_pdf_normalization(T, a1, b1)
    # num_samples = 1000
    uniform_samples = np.random.uniform(1e-5,1,num_samples)
    cdf_samples = inverse_cdf(uniform_samples)
    pdfs = np.array([pdf(w, T) / N for w in cdf_samples])
    samples_with_pdf = list(zip(cdf_samples, pdfs))
    all_data = {
            'samples': cdf_samples,
            'pdf': pdfs,
            'samples_with_pdf': samples_with_pdf,
            'a': a1,
            'b': b1,
            'T': T,
            'num': num_samples
        }
    return all_data


