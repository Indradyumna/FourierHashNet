from lsh.lsh_base import BaseLSH
from loguru import logger
import numpy as np
from utils.utils import *
from utils.training_utils import EarlyStoppingModule
import os

class RH_Trained(BaseLSH):
    """
       Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(RH_Trained, self).__init__(conf)
        self.hcode_model_weights_np = self.fetch_pretrained_hashcode_model_weights(conf)

    def fetch_pretrained_hashcode_model_weights(self, conf):       
 
        # NOTE: Below  lines should be same as in __main__  function of train_hashcode.py
        temp_IN_ARCH = "L" +  "".join([f"RL_{dim}_" for dim in conf.hashcode_training.hidden_layers])
        hashing_config_name_removal_set = {'device', 'embed_dim', 'subset_size', 'classPath'}
        hashing_conf_str = ",".join("{}{}".format(*i) for i in conf.hashing.items() if (i[0] not in hashing_config_name_removal_set))
        hashcode_training_config_name_removal_set = {'model_name', 'classPath', 'device', 'hidden_layers'}
        hashcode_training_conf_str = ",".join("{}{}".format(*i) for i in conf.hashcode_training.items() if (i[0] not in hashcode_training_config_name_removal_set))
        curr_task = conf.dataset.name + "," + hashing_conf_str + "," + hashcode_training_conf_str + ","+ temp_IN_ARCH

        #loading bestvalmodel 
        es = EarlyStoppingModule(conf.base_dir, curr_task, patience=conf.training.patience, logger=logger)

        checkpoint = es.load_best_model(device='cpu')
        # NOTE: using conf.hashing.device instead of conf.hashcode_training.device deliberately
        model = get_class(f"{conf.hashcode_training.classPath}.{conf.hashcode_training.model_name}")(conf).to(conf.hashing.device) 

        model.load_state_dict(checkpoint['model_state_dict'])    
        model_weights_np = {}

        model_weights_np['num_layers'] = int((len(model.init_net)+1)/2)
        for idx in range(model_weights_np['num_layers']):
            model_weights_np[idx] = {}
            model_weights_np[idx]['np_w'] = model.init_net[2*idx].weight.cpu().detach().numpy()
            model_weights_np[idx]['np_b'] = model.init_net[2*idx].bias.cpu().detach().numpy()
        #model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        #model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
        # else:
        #     raise NotImplementedError()

        # if av.LOSS_TYPE == "flora_hingeemb"  or av.LOSS_TYPE == "flora_hingeemb2" : 
        #     model_weights_np['np_w_q'] = model.init_qnet[0].weight.cpu().detach().numpy()
        #     model_weights_np['np_b_q'] = model.init_qnet[0].bias.cpu().detach().numpy()
        #     model_weights_np['np_w_c'] = model.init_cnet[0].weight.cpu().detach().numpy()
        #     model_weights_np['np_b_c'] = model.init_cnet[0].bias.cpu().detach().numpy()
        #model_weights_np['np_w'] = model.init_net[0].weight.cpu().detach().numpy()
        #model_weights_np['np_b'] = model.init_net[0].bias.cpu().detach().numpy()
    

        #checking existence of dumped trained hashcodes
        pathname = conf.base_dir + "allPklDumps/hashcodePickles/"+curr_task+"_hashcode_mat.pkl"

        assert os.path.exists(pathname), print(f"{pathname} does not exist")

        return model_weights_np


    def fetch_RH_hashcodes(self, embeds, isQuery, qid=None):
        batch_sz  = 50000
        #Writing split manually to ensure correctness
        batches = []
        for i in range(0, embeds.shape[0],batch_sz):
            batches.append(embeds[i:i+batch_sz])
        assert sum([item.shape[0] for item in batches]) == embeds.shape[0]

        hcode_list = []
        for batch_item in batches :
            # if self.hcode_model_weights_np['num_layers']==1:
            #     projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
            #     projections = np.tanh(self.av.TANH_TEMP * projections)
            # else:
            projections = batch_item@self.hcode_model_weights_np[0]['np_w'].T+self.hcode_model_weights_np[0]['np_b']
            for idx in range(1,self.hcode_model_weights_np['num_layers']):
                projections[projections<0]=0
                projections = projections@self.hcode_model_weights_np[idx]['np_w'].T+self.hcode_model_weights_np[idx]['np_b']
            projections = np.tanh(self.conf.hashcode_training.TANH_TEMP * projections)
            hcode_list.append(np.sign(projections))
            
        hashcodes = np.vstack(hcode_list)
        
        return hashcodes

