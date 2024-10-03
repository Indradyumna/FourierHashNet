from lsh.fhash_raw import Fhash_Raw
from loguru import logger
from lsh.lsh_utils import fetch_gaussian_hyperplanes
import numpy as np
from utils.utils import *
from utils.training_utils import EarlyStoppingModule
import os

class Fhash(Fhash_Raw):
    """
        Trained Fourier Map
        Untrained Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(Fhash, self).__init__(conf)
        #Trained Fmap dim used below -- because this is RH not RH_trained
        self.gauss_hplanes_cos = fetch_gaussian_hyperplanes(self.hcode_dim, conf.fmap_training.tr_fmap_dim)
        self.fmap_model_weights_np = self.check_pretrained_fmaps(conf)


    def check_pretrained_fmaps(self, conf):
        # NOTE: Below 3 lines should be same as __main__ function in train_fmaps.py
        temp_IN_ARCH = "L" +  "".join([f"RL_{dim}_" for dim in conf.fmap_training.hidden_layers])
        hashing_config_name_removal_set = {'device', 'embed_dim', 'subset_size', 'classPath'}
        hashing_conf_str = ",".join("{}{}".format(*i) for i in conf.hashing.items() if (i[0] not in hashing_config_name_removal_set))
        fmap_training_config_name_removal_set = {'model_name', 'classPath', 'device', 'hidden_layers'}
        fmap_training_conf_str = ",".join("{}{}".format(*i) for i in conf.fmap_training.items() if (i[0] not in fmap_training_config_name_removal_set))
        curr_task = conf.dataset.name + "," + hashing_conf_str + "," + fmap_training_conf_str + ","+ temp_IN_ARCH

        #loading bestvalmodel
        es = EarlyStoppingModule(conf.base_dir, curr_task, patience=conf.training.patience, logger=logger)
        
        checkpoint = es.load_best_model(device='cpu')
        
        # NOTE: using conf.hashing.device instead of conf.fmap_training.device deliberately
        model = get_class(f"{conf.fmap_training.classPath}.{conf.fmap_training.model_name}")(conf).to(conf.hashing.device)
        # model = AsymFmapTrainer(conf).to(conf.hashing.device)
        # model = FmapTrainer(conf).to(conf.hashing.device)s

        model.load_state_dict(checkpoint['model_state_dict'])    
        model_weights_np = {}
        # "AsymFmapCos":
        model_weights_np['np_w_q'] = model.init_net[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_q'] = model.init_net[0].bias.cpu().detach().numpy()
        model_weights_np['np_w_c'] = model.init_cnet[0].weight.cpu().detach().numpy()
        model_weights_np['np_b_c'] = model.init_cnet[0].bias.cpu().detach().numpy()
        # "FmapCos": 
        # model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        # model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()

        #checking existence of dumped trained fmaps
        pathname =  conf.base_dir  + "allPklDumps/fmapPickles/"+ curr_task +"_fmap_mat.pkl"

        assert( os.path.exists(pathname )), print(pathname)

        return model_weights_np


    def fetch_trained_fmaps(self, m_use, embeds, isQuery=False):
        fmaps =  self.generate_fmap(m_use, embeds, isQuery)

        #"AsymFmapCos":
        if isQuery:
            fmaps = fmaps@self.fmap_model_weights_np['np_w_q'].T+self.fmap_model_weights_np['np_b_q']
            fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)
            #assert (np.allclose(fmaps,self.tr_fmap_data['query'][self.av.SPLIT].cpu().numpy()[qid],atol=1e-06))
        else:
            fmaps = fmaps@self.fmap_model_weights_np['np_w_c'].T+self.fmap_model_weights_np['np_b_c']
            fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)
        # "FmapCos"
        # fmaps = fmaps@self.fmap_model_weights_np['np_w'].T+self.fmap_model_weights_np['np_b']
        # fmaps = fmaps/np.linalg.norm(fmaps,axis=-1,keepdims=True)
        
        if self.DEBUG:
            assert np.all(np.isclose(np.linalg.norm(fmaps[0]),\
                                        np.linalg.norm(fmaps, axis=1)))

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
            fmaps =  self.fetch_trained_fmaps(self.m_use, batch_item, isQuery)
            projections = fmaps@self.gauss_hplanes_cos
            hcode_list.append(np.sign(projections))

        hashcodes = np.vstack(hcode_list)

        return hashcodes