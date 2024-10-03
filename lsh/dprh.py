from lsh.lsh_base import BaseLSH
from loguru import logger
from lsh.lsh_utils import fetch_gaussian_hyperplanes
import numpy as np



class DPRH(BaseLSH):
    """
       Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(DPRH, self).__init__(conf)
        self.gauss_hplanes_dot = fetch_gaussian_hyperplanes(conf.hashing.hcode_dim, conf.dataset.embed_dim+1)


    def fetch_RH_hashcodes(self, embeds, isQuery, qid=None):
        max_norm = np.max(np.linalg.norm(embeds, axis=1))
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0],batch_sz):
            batches.append(embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]

        hcode_list = []
        for batch_item in batches :
            batch_item_scaled = batch_item/max_norm
            append = np.expand_dims(np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))),axis=-1)
            batch_item_augmented = np.hstack((batch_item_scaled,append))
            projections = batch_item_augmented@self.gauss_hplanes_dot
            hcode_list.append(np.sign(projections))
            
        hashcodes = np.vstack(hcode_list)
        
        return hashcodes
