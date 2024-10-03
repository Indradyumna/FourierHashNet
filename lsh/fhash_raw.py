from lsh.lsh_base import BaseLSH
from loguru import logger
from lsh.lsh_utils import fetch_gaussian_hyperplanes, fetch_omega_samples
import numpy as np
from utils.utils import *
from utils.training_utils import EarlyStoppingModule

class Fhash_Raw(BaseLSH):
    """
        Untrained Fourier Map
        Untrained Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(Fhash_Raw, self).__init__(conf)
        # 4 * num_omega_samples * emebd_dim  <-- effective random hyperplanes dim
        self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.hcode_dim, conf.hashing.m_use * conf.dataset.embed_dim * 4)

        self.DEBUG = conf.DEBUG
        self.m_use = conf.hashing.m_use
        self.T = conf.hashing.T
        self.some_preprocessing_for_speedup(conf)


    def some_preprocessing_for_speedup(self,conf):
        self.ws, self.pdfs = fetch_omega_samples(conf)
        assert(len(self.ws) == len(self.pdfs)) , f"Size mismatch{len(self.pdfs) }, {len(self.ws)}"

        self.sqrt_pdfs = np.sqrt(self.pdfs)
        self.R_G = 2 * (np.sin(self.ws * self.T/2))**2 / self.ws**2 + \
                self.T * np.sin(self.ws * self.T) / self.ws
        self.I_G = np.sin(self.ws * self.T) / self.ws**2 - \
                self.T * np.cos(self.ws * self.T) / self.ws
        self.sign_RG = np.sign(self.R_G)
        self.sign_IG = np.sign(self.I_G)
        self.sqrt_abs_RG = np.sqrt(np.abs(self.R_G))
        self.sqrt_abs_IG = np.sqrt(np.abs(self.I_G))

        self.temp1 = (self.sign_RG * self.sqrt_abs_RG) /self.sqrt_pdfs
        self.temp2 = (self.sign_IG * self.sqrt_abs_IG) / self.sqrt_pdfs
        self.temp3 = - self.temp2
        self.concat_temp = np.hstack([self.temp1, self.temp1, self.temp3, self.temp2])


    def generate_fmap(self, m_use, embeds, isQuery=False): 
        """
            Given some value of T, limit, a,b
            Fetch/generate prob samples
            compute map and return 
        """
        embeds_rep = np.repeat(embeds,m_use,axis=-1)
        #print(embeds_rep.dtype) 
        thetas = embeds_rep*self.ws
    
        cos_theta_by_sqrt_pdf = np.cos(thetas) / self.sqrt_pdfs
        sin_theta_by_sqrt_pdf = np.sin(thetas) / self.sqrt_pdfs
        if isQuery:
            fmap1 = self.sign_RG * self.sqrt_abs_RG * cos_theta_by_sqrt_pdf 
            fmap2 = self.sign_RG * self.sqrt_abs_RG * sin_theta_by_sqrt_pdf
            fmap3 = - self.sign_IG * self.sqrt_abs_IG * sin_theta_by_sqrt_pdf
            fmap4 = self.sign_IG * self.sqrt_abs_IG * cos_theta_by_sqrt_pdf
        else:
            fmap1 =  self.sqrt_abs_RG * cos_theta_by_sqrt_pdf
            fmap2 =  self.sqrt_abs_RG * sin_theta_by_sqrt_pdf
            fmap3 =  self.sqrt_abs_IG * cos_theta_by_sqrt_pdf
            fmap4 =  self.sqrt_abs_IG * sin_theta_by_sqrt_pdf
        
        fmaps = np.hstack([fmap1,fmap2,fmap3,fmap4])#.numpy()
      
        return fmaps


    def fetch_RH_hashcodes(self, embeds, isQuery, qid=None):
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0],batch_sz):
            batches.append(embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]

        hcode_list = []
        for batch_item in batches :
            fmaps =  self.generate_fmap(self.m_use, batch_item, isQuery)
            
            if self.DEBUG:
                assert np.all(np.isclose(np.linalg.norm(fmaps[0]),\
                                            np.linalg.norm(fmaps, axis=1)))
        

            projections = batch_item@self.gauss_hplanes_cos
            hcode_list.append(np.sign(projections))

        hashcodes = np.vstack(hcode_list)

        return hashcodes