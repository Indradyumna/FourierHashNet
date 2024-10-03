from lsh.lsh_base import BaseLSH
from loguru import logger
from lsh.lsh_utils import fetch_gaussian_hyperplanes
import numpy as np



class RH(BaseLSH):
    """
       Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(RH, self).__init__(conf)
        self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(conf.hashing.hcode_dim, conf.dataset.embed_dim)


    def fetch_RH_hashcodes(self, embeds, isQuery, qid=None):
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0],batch_sz):
            batches.append(embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]

        hcode_list = []
        for batch_item in batches :
            projections = batch_item@self.gauss_hplanes_cos
            hcode_list.append(np.sign(projections))
            
        hashcodes = np.vstack(hcode_list)
        
        return hashcodes
