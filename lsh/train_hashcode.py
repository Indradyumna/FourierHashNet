import torch
torch.backends.cuda.matmul.allow_tf32 = False
import torch.nn as nn
from lsh.scoring import pairwise_ranking_loss_similarity
import time
from loguru import logger
from src.embeddings_loader import fetch_corpus_embeddings, fetch_query_embeddings, fetch_ground_truths
from lsh.fhash import Fhash
import numpy as np
import tqdm
from lsh.scoring import dot_sim, hinge_sim, pairwise_cosine_sim 
import os
from utils.training_utils import EarlyStoppingModule
import random
import pickle
from sklearn.utils import shuffle
from omegaconf import OmegaConf



class HashCodeTrainer(nn.Module):
    """
    """
    def __init__(self, conf):
        super(HashCodeTrainer, self).__init__()
        self.LOSS_TYPE = conf.hashcode_training.LOSS_TYPE
        self.FENCE_LAMBDA = conf.hashcode_training.FENCE
        self.DECORR_LAMBDA = conf.hashcode_training.DECORR
        self.C1_LAMBDA = conf.hashcode_training.C1
        self.QA_MARGIN = conf.hashcode_training.QA_MARGIN
        self.hashing_name = conf.hashing.name

        self.init_net = []

        if self.hashing_name == "Fhash_Trained":
            self.inner_hs = [conf.fmap_training.tr_fmap_dim] + conf.hashcode_training.hidden_layers + [conf.hashing.hcode_dim]
            # self.inner_hs = [conf.hashing.m_use * conf.dataset.embed_dim * 4] + conf.hashcode_training.hidden_layers + [conf.hashing.hcode_dim]
        elif self.hashing_name == "RH_Trained":
            self.inner_hs = [conf.dataset.embed_dim] + conf.hashcode_training.hidden_layers + [conf.hashing.hcode_dim]
        elif self.hashing_name == "DPRH_Trained":
            self.inner_hs = [conf.dataset.embed_dim+1] + conf.hashcode_training.hidden_layers + [conf.hashing.hcode_dim]
        else:
            assert False, print(f"You should not be here with {self.hashing_name}. This is for trained hashcode variants.")

        for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
                lin = torch.nn.Linear(h0, h1)
                self.init_net.append(lin)
                self.init_net.append(torch.nn.ReLU())
        self.init_net.pop() # pop the last relu 
        self.init_net = torch.nn.Sequential(*self.init_net)
        self.tanh  = nn.Tanh()
        self.TANH_TEMP = conf.hashcode_training.TANH_TEMP     

    def forward(self, fmaps):
        """
            :param  Fmaps
            :return  Hcodes
        """
        code = self.init_net(fmaps)
        return self.tanh(self.TANH_TEMP * code)

    def computeLoss(self, cfmaps, qfmaps, targets):
        if self.LOSS_TYPE == "query_agnostic":
            """
                Note that here all query and corpus embeddings are sent in one chunk
                naming cfmaps us slightly misleading -- it contains both query and corpus representationas
                Thhis was used in PermGNN
            """
            all_hcodes = self.forward(cfmaps)
            bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])
            decorrelation_loss = torch.abs(torch.mean((all_hcodes.T@all_hcodes).fill_diagonal_(0)))
            fence_sitting_loss =  torch.norm(all_hcodes.abs()-1, p=1)/ (all_hcodes.shape[0]*all_hcodes.shape[1])
            loss = self.FENCE_LAMBDA * fence_sitting_loss +\
                self.DECORR_LAMBDA * decorrelation_loss+\
                (1-self.FENCE_LAMBDA-self.DECORR_LAMBDA) * bit_balance_loss
            return loss, bit_balance_loss,decorrelation_loss, fence_sitting_loss
        elif self.LOSS_TYPE == "query_aware":
            q_hcodes = self.forward(qfmaps)
            c_hcodes = self.forward(cfmaps)
            all_hcodes = torch.cat([q_hcodes,c_hcodes])
            
            fence_sitting_loss =  torch.norm(all_hcodes.abs()-1, p=1)/ (all_hcodes.shape[0]*all_hcodes.shape[1])
            bit_balance_loss = torch.sum(torch.abs(torch.sum(all_hcodes,dim=0)))/(all_hcodes.shape[0]*all_hcodes.shape[1])
            
            preds = (q_hcodes*c_hcodes).sum(-1)
            predPos = preds[targets>0.5]
            predNeg = preds[targets<0.5]

            #ranking_loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), 1)
            ranking_loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.QA_MARGIN)

            loss = self.FENCE_LAMBDA * fence_sitting_loss +\
                self.C1_LAMBDA * ranking_loss+\
                (1-self.FENCE_LAMBDA-self.C1_LAMBDA) * bit_balance_loss
            return loss, bit_balance_loss,ranking_loss, fence_sitting_loss

        else:
            assert False, print(f"Unknown loss type {self.LOSS_TYPE}")
            
            
class HashcodeDataLoader(object):
    def __init__(self, conf):
        self.device = conf.hashcode_training.device
        self.LOSS_TYPE = conf.hashcode_training.LOSS_TYPE
        self.BATCH_SIZE = 1024 #hardcoded for now
        self.scoring_function = {"cos": pairwise_cosine_sim,
                                'dot': dot_sim,
                                'sighinge': hinge_sim,
                                "hinge": hinge_sim}[conf.hashing.FUNC]
        self.hashing_name = conf.hashing.name
        
        corpus_embeds_fetch_start = time.time()
        self.corpus_embeds = fetch_corpus_embeddings(conf)
        if isinstance(self.corpus_embeds, np.ndarray):
            self.corpus_embeds = torch.from_numpy(self.corpus_embeds).to(self.device)
        corpus_embeds_fetch_time = time.time() - corpus_embeds_fetch_start
        logger.info(f"Corpus embeds shape: {self.corpus_embeds.shape}, time={corpus_embeds_fetch_time}")
        
        if self.hashing_name == "DPRH_Trained":
            self.corpus_embeds_aug = torch.zeros((self.corpus_embeds.shape[0], self.corpus_embeds.shape[1]+1),dtype = self.corpus_embeds.dtype, device=self.device)
            max_norm = np.max(np.linalg.norm(self.corpus_embeds.cpu().numpy(), axis=1))
            batch_sz = 40000
            for i in range(0, self.corpus_embeds.shape[0],batch_sz):
                batch_item =  self.corpus_embeds[i:i+batch_sz].cpu().numpy()
                batch_item_scaled = batch_item/max_norm
                app = np.expand_dims(np.sqrt(1-np.square(np.linalg.norm(batch_item_scaled, axis=1))),axis=-1)
                self.corpus_embeds_aug[i:i+batch_sz] = torch.from_numpy(np.hstack((batch_item_scaled,app)))

        if self.hashing_name == "Fhash_Trained":
            self.lsh = Fhash(conf)

            corpusfmaps_start_time = time.time()

            self.batch_sz = 40000
            fmaps_init = time.time()
            self.corpus_fmaps = torch.zeros((self.corpus_embeds.shape[0], conf.fmap_training.tr_fmap_dim), device=self.device)
            logger.info(f"init corpus fmaps, shape={self.corpus_fmaps.shape}, time={time.time()-fmaps_init}")
            for i in tqdm.tqdm(range(0, self.corpus_embeds.shape[0],self.batch_sz)):
                    self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.fetch_trained_fmaps(conf.hashing.m_use, self.corpus_embeds[i:i+self.batch_sz].cpu().numpy(), isQuery=False)).type(torch.float).to(self.device)
            
            corpusfmaps_time = time.time() - corpusfmaps_start_time
            logger.info(f"Corpus fmaps shape: {self.corpus_fmaps.shape}, time={corpusfmaps_time}")
        
        self.query_embeds  = {}
        if self.hashing_name == "DPRH_Trained":
            self.query_embeds_aug  = {}
        if self.hashing_name == "Fhash_Trained":
            self.query_fmaps  = {}
        self.ground_truth = {}
        # self.list_pos = {}
        # self.list_neg = {} 
        self.list_total_arranged_per_query = {}
        # self.labels_total_arranged_per_query = {}
        
        for mode in ["train", "val"]:
        
            self.query_embeds[mode] = fetch_query_embeddings(conf, mode)
            if isinstance(self.query_embeds[mode], np.ndarray):
                self.query_embeds[mode] = torch.from_numpy(self.query_embeds[mode]).to(self.device)
            if self.hashing_name == "DPRH_Trained":
                #Query augmentation only needs a zero appended to the end of normalized query vec
                self.query_embeds_aug[mode] = torch.zeros((self.query_embeds[mode].shape[0], self.query_embeds[mode].shape[1]+1),dtype = self.query_embeds[mode].dtype, device=self.device)
                self.query_embeds_aug[mode][:,:self.query_embeds[mode].shape[1]] = torch.from_numpy(self.query_embeds[mode].cpu().numpy()/(np.linalg.norm(self.query_embeds[mode].cpu().numpy(),axis=1)[:,None]+1e-8)).to(self.device)

            if self.hashing_name == "Fhash_Trained":
                self.query_fmaps[mode] = torch.from_numpy(self.lsh.fetch_trained_fmaps(conf.hashing.m_use, self.query_embeds[mode].cpu().numpy(), isQuery=True)).type(torch.float).to(self.device)

            if self.LOSS_TYPE == "query_aware": 
                #NOTE: IMP: Needs to be factored in for Ghash  (TODO)
                num_pos = int(self.corpus_embeds.shape[0]/(2**conf.hashcode_training.QA_subset_size))     
                sc = self.scoring_function((conf, self.query_embeds[mode].cpu(),self.corpus_embeds.cpu()))
                if not isinstance(sc, np.ndarray):
                    sc = sc.numpy()
                gt = {}
                for qidx in range(len(self.query_embeds[mode])):
                    pos_cids = np.argsort(sc[qidx])[::-1][:num_pos].tolist()
                    gt[qidx] = pos_cids
                
                self.ground_truth[mode] = gt
            else:
                self.ground_truth[mode] = fetch_ground_truths(conf, mode)
            # self.list_pos[mode] = []
            # self.list_neg[mode] = []
            self.list_total_arranged_per_query[mode] = []
            # self.labels_total_arranged_per_query[mode] = []
            for q in range(self.query_embeds[mode].shape[0]):
                for c in range(self.corpus_embeds.shape[0]): 
                    if c in self.ground_truth[mode][q]:
                        # self.list_pos[mode].append(((q,c),1.0))
                        self.list_total_arranged_per_query[mode].append(((q,c),1.0))
                        # self.labels_total_arranged_per_query[mode].append(1.0)
                    else:
                        # self.list_neg[mode].append(((q,c),0.0))  
                        self.list_total_arranged_per_query[mode].append(((q,c),0.0))
                        # self.labels_total_arranged_per_query[mode].append(0.0)
            
        logger.info('Query embeds fetched and fmaps generated.')
        logger.info('Ground truth fetched.')
        self.preprocess_create_per_query_batches()


    def create_fmap_batches(self,mode):
        if self.hashing_name == "DPRH_Trained":
            all_fmaps = torch.cat([self.query_embeds_aug[mode], self.corpus_embeds_aug])
        if self.hashing_name == "RH_Trained":
            all_fmaps = torch.cat([self.query_embeds[mode], self.corpus_embeds])
        if self.hashing_name == "Fhash_Trained":
            all_fmaps = torch.cat([self.query_fmaps[mode], self.corpus_fmaps])
        if mode == "train":
            all_fmaps = all_fmaps[torch.randperm(all_fmaps.shape[0])]
        
        self.batches = list(all_fmaps.split(self.BATCH_SIZE))
        self.num_batches = len(self.batches)
        return self.num_batches

    def fetch_fmap_batched_data_by_id(self,i):
        assert(i < self.num_batches)  
        return self.batches[i]

    def preprocess_create_per_query_batches(self):
        split_len  = self.corpus_embeds.shape[0]
        print("In preprocess_create_per_query_batches")
        self.per_query_batches={} 
        for mode in ["train", "val"]:
            self.per_query_batches[mode]={}
            whole_list = self.list_total_arranged_per_query[mode]
            batches = [whole_list[i:i + split_len] for i in range(0, len(whole_list), split_len)]
            alists = []
            blists = []
            scores = []

            for btch in batches:
                btch_np = np.array(btch, dtype=object)
                scores.append(torch.tensor(btch_np[:,1].tolist()).cuda())
                temp = np.array(btch_np[:,0].tolist())
                alists.append(temp[:,0].tolist())
                blists.append(temp[:,1].tolist())
                
            self.per_query_batches[mode]['alists'] = alists
            self.per_query_batches[mode]['blists'] = blists
            self.per_query_batches[mode]['scores'] = scores



    def fetch_batched_data_by_id_optimized(self,i):
        """             
        """
        assert(i < self.num_batches)  
        alist = self.alists[i]
        blist = self.blists[i]
        score = self.scores[i]
        if self.hashing_name == "DPRH_Trained":
            query_tensors = self.query_embeds_aug[self.mode][alist]
            corpus_tensors = self.corpus_embeds_aug[blist]
        if self.hashing_name == "RH_Trained":
            query_tensors = self.query_embeds[self.mode][alist]
            corpus_tensors = self.corpus_embeds[blist]
        if self.hashing_name == "Fhash_Trained":
            query_tensors = self.query_fmaps[self.mode][alist]
            corpus_tensors = self.corpus_fmaps[blist]

        target = score
        return corpus_tensors, query_tensors, target 

    def create_per_query_batches(self,mode):
        """
          create batches as is and return number of batches created
        """
        self.alists = self.per_query_batches[mode]['alists']
        self.blists = self.per_query_batches[mode]['blists']
        self.scores = self.per_query_batches[mode]['scores']

        if mode=="train":
            self.alists,self.blists,self.scores = shuffle(self.alists,self.blists,self.scores)

        self.num_batches = len(self.alists)  
        self.mode = mode

        return self.num_batches

def evaluate_validation_query_aware(model, sampler, mode):
  model.eval()

  total_loss = 0 
  total_bit_balance_loss = 0 
  total_ranking_loss = 0
  total_fence_sitting_loss = 0
  n_batches = sampler.create_per_query_batches(mode)
  for i in tqdm.tqdm(range(n_batches)):
    batch_corpus_tensors, batch_query_tensors, batch_target = sampler.fetch_batched_data_by_id_optimized(i)
    #batch_tensors = sampler.fetch_fmap_batched_data_by_id(i)
    loss,bit_balance_loss,ranking_loss, fence_sitting_loss = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target)
    total_loss = total_loss+loss.item()
    total_bit_balance_loss += bit_balance_loss.item() 
    total_ranking_loss += ranking_loss.item()
    total_fence_sitting_loss += fence_sitting_loss.item()

  return total_loss, total_bit_balance_loss, total_ranking_loss, total_fence_sitting_loss 


def evaluate_validation_query_agnostic(model, sampler, mode):
  model.eval()

  total_loss = 0 
  total_bit_balance_loss = 0 
  total_decorrelation_loss = 0
  total_fence_sitting_loss = 0
  n_batches = sampler.create_fmap_batches(mode)
  for i in tqdm.tqdm(range(n_batches)):
    batch_tensors = sampler.fetch_fmap_batched_data_by_id(i)
    loss,bit_balance_loss,decorrelation_loss, fence_sitting_loss = model.computeLoss(batch_tensors, None, None)
    total_loss = total_loss+loss.item()
    total_bit_balance_loss += bit_balance_loss.item() 
    total_decorrelation_loss += decorrelation_loss.item()
    total_fence_sitting_loss += fence_sitting_loss.item()

  return total_loss, total_bit_balance_loss, total_decorrelation_loss, total_fence_sitting_loss 


def run_hashcode_gen(conf, curr_task):
        train_data = HashcodeDataLoader(conf)
        model = HashCodeTrainer(conf).to(conf.hashcode_training.device)
        
        cnt = 0
        for param in model.parameters():
            cnt=cnt+torch.numel(param)
        logger.info(f"no. of params in model: {cnt}")
        
        es = EarlyStoppingModule(conf.base_dir, curr_task, patience=conf.training.patience, logger=logger)

        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=conf.training.learning_rate,
                                    weight_decay=conf.training.weight_decay)
      
        best_neg_val_loss = 0  # weird naming but more straightforward to understand
        run = 0
        while conf.training.run_till_early_stopping and run < conf.training.num_epochs:
            start_time = time.time()
            if conf.hashcode_training.LOSS_TYPE=="query_agnostic":
                n_batches = train_data.create_fmap_batches(mode="train")
            elif conf.hashcode_training.LOSS_TYPE == "query_aware":
                n_batches = train_data.create_per_query_batches(mode="train")
            else:
                assert False, print(f"Unknown hashcode training loss type {conf.hashcode_training.LOSS_TYPE}")

            epoch_loss =0
            epoch_bit_balance_loss = 0 
            epoch_decorrelation_loss = 0
            epoch_fence_sitting_loss = 0
            epoch_ranking_loss = 0

             
            for i in tqdm.tqdm(range(n_batches)):
                optimizer.zero_grad()
                if conf.hashcode_training.LOSS_TYPE=="query_agnostic":
                    batch_tensors = train_data.fetch_fmap_batched_data_by_id(i)

                    loss,bit_balance_loss,decorrelation_loss, fence_sitting_loss  = model.computeLoss(batch_tensors, None,None)
                    epoch_bit_balance_loss += bit_balance_loss.item() 
                    epoch_decorrelation_loss += decorrelation_loss.item()
                    epoch_fence_sitting_loss += fence_sitting_loss.item()
                if conf.hashcode_training.LOSS_TYPE == "query_aware":
                    batch_corpus_tensors, batch_query_tensors, batch_target = train_data.fetch_batched_data_by_id_optimized(i)

                    loss,bit_balance_loss, fence_sitting_loss, ranking_loss  = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target)
                    epoch_bit_balance_loss += bit_balance_loss.item() 
                    epoch_fence_sitting_loss += fence_sitting_loss.item()
                    epoch_ranking_loss += ranking_loss.item()

                loss.backward()
                optimizer.step()
                epoch_loss = epoch_loss + loss.item()   

            if conf.hashcode_training.LOSS_TYPE=="query_agnostic":
                logger.info(f"Epoch: {run} loss: {epoch_loss} bit_balance_loss: {epoch_bit_balance_loss} decorrelation_loss: {epoch_decorrelation_loss} fence_sitting_loss: {epoch_fence_sitting_loss} time: {time.time()-start_time}")
            if conf.hashcode_training.LOSS_TYPE == "query_aware":
                logger.info(f"Epoch: {run} loss: {epoch_loss} bit_balance_loss: {epoch_bit_balance_loss} ranking_loss: {epoch_ranking_loss} fence_sitting_loss: {epoch_fence_sitting_loss} time: {time.time()-start_time}")

            start_time = time.time()
            if conf.hashcode_training.LOSS_TYPE=="query_aware":
                val_loss,total_bit_balance_loss,total_ranking_loss, total_fence_sitting_loss = evaluate_validation_query_aware(model,train_data, mode="val")
                logger.info(f"Epoch: {run} VAL loss: {val_loss} bit_balance_loss: {total_bit_balance_loss} ranking_loss: {total_ranking_loss} fence_sitting_loss: {total_fence_sitting_loss} time: {time.time()-start_time}")
            if conf.hashcode_training.LOSS_TYPE=="query_agnostic":
                val_loss,total_bit_balance_loss,total_decorrelation_loss, total_fence_sitting_loss = evaluate_validation_query_agnostic(model,train_data, mode="val")
                logger.info(f"Epoch: {run} VAL loss: {val_loss} bit_balance_loss: {total_bit_balance_loss} decorrelation_loss: {total_decorrelation_loss} fence_sitting_loss: {total_fence_sitting_loss} time: {time.time()-start_time}")
        
            neg_val_loss = -val_loss

            state_dict = {
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "epoch": run,
                "best_neg_val_loss": best_neg_val_loss,
                "neg_val_loss": neg_val_loss,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state(),
                'np_rng_state': np.random.get_state(),
                'random_state': random.getstate(),
                'patience': es.patience,
                'best_scores': es.best_scores,
                'num_bad_epochs': es.num_bad_epochs,
                'should_stop_now': es.should_stop_now,
            }

            state_dict =  es.check([neg_val_loss], state_dict)
            best_neg_val_loss = state_dict["best_neg_val_loss"]

            if es.should_stop_now:
                break
            run+=1
       
        #generate and dump hashcode  pickles
        #IMP: Load best validation model here
        checkpoint = es.load_best_model()
        model.load_state_dict(checkpoint['model_state_dict'])      

        all_hashcodes = {}
        corpus_hashcodes = torch.zeros((train_data.corpus_embeds.shape[0], conf.hashing.hcode_dim), device=conf.hashcode_training.device)
        bsz = 40000
        for i in tqdm.tqdm(range(0, train_data.corpus_embeds.shape[0],bsz)):
            if conf.hashing.name == "DPRH_Trained":
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_embeds_aug[i:i+bsz,:]).data
            if conf.hashing.name == "RH_Trained":
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_embeds[i:i+bsz,:]).data
            if conf.hashing.name == "Fhash_Trained":
                corpus_hashcodes[i:i+bsz,:] = model.forward(train_data.corpus_fmaps[i:i+bsz,:]).data

        query_hashcodes = {}
        for mode in ["train", "val"]:
            if conf.hashing.name == "DPRH_Trained":
                query_hashcodes[mode] =  model.forward(train_data.query_embeds_aug[mode]).data
            if conf.hashing.name == "RH_Trained":
                query_hashcodes[mode] =  model.forward(train_data.query_embeds[mode]).data
            if conf.hashing.name == "Fhash_Trained":
                query_hashcodes[mode] =  model.forward(train_data.query_fmaps[mode]).data

        all_hashcodes['query'] = query_hashcodes
        all_hashcodes['corpus'] = corpus_hashcodes
        logger.info(f"Dumping trained hashcode pickle at {pickle_fp}")
        with open(pickle_fp, 'wb') as f:
            pickle.dump(all_hashcodes, f)
            
            
if __name__ == "__main__":

    main_conf = OmegaConf.load("configs/config.yaml")
    cli_conf = OmegaConf.from_cli()
    data_conf = OmegaConf.load(f"configs/data_configs/{cli_conf.dataset.rel_mode}/{cli_conf.dataset.name}.yaml")
    # model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    hash_conf = OmegaConf.load(f"configs/hash_configs/{cli_conf.hashing.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, hash_conf, cli_conf)

    # NOTE: Below  lines should be same as first three in "fetch_pretrained_hashcode_model_weights" function in Fhash_Trained.py/RH_Trained.py
    temp_IN_ARCH = "L" +  "".join([f"RL_{dim}_" for dim in conf.hashcode_training.hidden_layers])
    hashing_config_name_removal_set = {'device', 'embed_dim', 'subset_size', 'classPath'}
    hashing_conf_str = ",".join("{}{}".format(*i) for i in conf.hashing.items() if (i[0] not in hashing_config_name_removal_set))
    hashcode_training_config_name_removal_set = {'model_name', 'classPath', 'device', 'hidden_layers'}
    hashcode_training_conf_str = ",".join("{}{}".format(*i) for i in conf.hashcode_training.items() if (i[0] not in hashcode_training_config_name_removal_set))
    curr_task = conf.dataset.name + "," + hashing_conf_str + "," + hashcode_training_conf_str + ","+ temp_IN_ARCH

    if conf.hashing.name == "Fhash_Trained":
        fmap_IN_ARCH = "L" +  "".join([f"RL_{dim}_" for dim in conf.fmap_training.hidden_layers])
        fmap_training_config_name_removal_set = {'model_name', 'classPath', 'device', 'hidden_layers'}
        fmap_training_conf_str = ",".join("{}{}".format(*i) for i in conf.fmap_training.items() if (i[0] not in fmap_training_config_name_removal_set))
        #Earlier curr_task gets augmented in this case
        curr_task = curr_task + "," + fmap_training_conf_str + "," + fmap_IN_ARCH



    logger.info(f"Task name: {curr_task}")
    logger.add(f"{conf.log.dir}/{curr_task}.log")
    logger.info(OmegaConf.to_yaml(conf))

    # Set random seeds
    seed = 4
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False



    pickle_fp = conf.base_dir + "allPklDumps/hashcodePickles/"+ curr_task + "_hashcode_mat.pkl"
    if not os.path.exists(pickle_fp):
        run_hashcode_gen(conf, curr_task)
        
    

# (CUDA_VISIBLE_DEVICES=0 python -m lsh.train_hashcode dataset.name="msweb294" dataset.rel_mode="fhash" hashing.m_use=10  dataset.embed_dim=294   hashing.hcode_dim=64 hashcode_training.hidden_layers=[]   hashing.FUNC="sighinge" hashing.name="RH_Trained" hashing.num_hash_tables=10 training.patience=50 hashcode_training.LOSS_TYPE="query_agnostic" hashcode_training.QA_subset_size=8  hashcode_training.FENCE=0.1 hashcode_training.DECORR=0.1  hashcode_training.QA_MARGIN=1.0) & 

# msweb294,nameRH_Trained,FUNCsighinge,hcode_dim64,num_hash_tables10,m_use10,LOSS_TYPEquery_agnostic,QA_subset_size8,QA_MARGIN1.0,FENCE0.1,DECORR0.8,C10,TANH_TEMP1.0,L
    
# msweb294,nameRH_Trained,FUNCsighinge,hcode_dim64,num_hash_tables10,LOSS_TYPEquery_aware,QA_subset_size8,QA_MARGIN1.0,FENCE0.1,DECORR0,C10.2,TANH_TEMP1.0,L