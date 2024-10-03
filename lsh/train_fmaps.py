import torch
import torch.nn as nn
from lsh.fhash_raw import Fhash_Raw
import time
from loguru import logger
from src.embeddings_loader import fetch_corpus_embeddings, fetch_query_embeddings, fetch_ground_truths
import tqdm
import math
import random
import numpy as np
from sklearn.metrics import average_precision_score
import os
from utils.training_utils import EarlyStoppingModule
import pickle
from omegaconf import OmegaConf
from lsh.scoring import pairwise_ranking_loss_similarity, pairwise_ranking_loss_similarity_per_query





class AsymFmapTrainer(nn.Module):
    """
      Fetch fmaps for q, c 
      feed into NN(LRL)
      Compute loss on final FMAP 
    """
    def __init__(self, conf):
        super(AsymFmapTrainer, self).__init__()
        self.margin = conf.fmap_training.margin
      
        self.init_net = []
        self.inner_hs = [conf.hashing.m_use * conf.dataset.embed_dim * 4] + conf.fmap_training.hidden_layers + [conf.fmap_training.tr_fmap_dim]
        
        for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
            lin = torch.nn.Linear(h0, h1)
            self.init_net.append(lin)
            self.init_net.append(torch.nn.ReLU())
        self.init_net.pop() # pop the last relu/tanh

        self.init_net = torch.nn.Sequential(*self.init_net)
      
        self.init_cnet = []
        for h0, h1 in zip(self.inner_hs, self.inner_hs[1:]):
            lin = torch.nn.Linear(h0, h1)
            self.init_cnet.append(lin)
            self.init_cnet.append(torch.nn.ReLU())
        self.init_cnet.pop() # pop the last relu/tanh
        
        self.init_cnet = torch.nn.Sequential(*self.init_cnet)


        self.tanh  = nn.Tanh()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.bce_loss_with_prob = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.tr_fmap_loss = conf.fmap_training.tr_fmap_loss
        
    def forward(self, fmaps,isQ=True):
        """
            :param  Fmaps
            :return  Hcodes
        """
        #TODO: check that fmaps are in correct device
        if isQ:
            code = self.init_net(fmaps)
        else:
            code = self.init_cnet(fmaps)
        return code/torch.norm(code,dim=-1,keepdim=True)


    def computeLoss(self, cfmaps, qfmaps, targets, batch_query_ids):
        #TODO: make sure qfmaps and cfmaps are in correct device
        """
            :param   cfmaps  : corpus fourier maps
            :param   qfmaps  : query fourier maps
            :param   targets : ground truth scores 0/1
            :return  loss   : Hinge ranking loss
        """
        q_maps = self.forward(qfmaps,isQ=True)
        c_maps = self.forward(cfmaps,isQ=False)
        preds = (q_maps*c_maps).sum(-1)
        
        # We have tried a variety of loss functions
        # TODO: remove some redundant ones
        if self.tr_fmap_loss == "BCE":
            preds = (preds+1)/2
            loss = self.bce_loss(preds,targets)   
        elif self.tr_fmap_loss == "BCE2":
            targets[targets==0]=-1
            loss = self.bce_loss(preds,targets)   
        elif self.tr_fmap_loss == "BCE3":
            preds = (preds+1)/2
            loss = self.bce_loss_with_prob(preds,targets)   
        elif self.tr_fmap_loss == "MSE":
            targets[targets==0]=-1
            loss = self.mse_loss(preds,targets)   
        elif self.tr_fmap_loss == "PQR":
            predPos = preds[targets>0.5]
            predNeg = preds[targets<0.5]
            qidPos = batch_query_ids[targets>0.5]
            qidNeg = batch_query_ids[targets<0.5]
            loss  = pairwise_ranking_loss_similarity_per_query(predPos.unsqueeze(1),predNeg.unsqueeze(1),qidPos.unsqueeze(1),qidNeg.unsqueeze(1), self.margin)
        else:
            predPos = preds[targets>0.5]
            predNeg = preds[targets<0.5]
            loss = pairwise_ranking_loss_similarity(predPos.unsqueeze(1),predNeg.unsqueeze(1), self.margin)

        return loss
    
    
    
class FmapDataLoader(object):
    def __init__(self, conf): 
        self.device = conf.fmap_training.device
        self.P2N = conf.fmap_training.P2N
        self.BATCH_SIZE = 1024 #hardcoded for now
        self.lsh = Fhash_Raw(conf)
        corpus_embeds_fetch_start = time.time()
        self.corpus_embeds = fetch_corpus_embeddings(conf)
        corpus_embeds_fetch_time = time.time() - corpus_embeds_fetch_start
        logger.info(f"Corpus embeds shape: {self.corpus_embeds.shape}, time={corpus_embeds_fetch_time}")
        
        corpusfmaps_start_time = time.time()

        self.batch_sz = 40000
        fmaps_init = time.time()
        self.corpus_fmaps = torch.zeros((self.corpus_embeds.shape[0], conf.hashing.m_use * self.corpus_embeds.shape[1] * 4), device=self.device)
        logger.info(f"init corpus fmaps, shape={self.corpus_fmaps.shape}, time={time.time()-fmaps_init}")
        for i in tqdm.tqdm(range(0, self.corpus_embeds.shape[0],self.batch_sz)):
                self.corpus_fmaps[i:i+self.batch_sz,:] = torch.from_numpy(self.lsh.generate_fmap(conf.hashing.m_use, self.corpus_embeds[i:i+self.batch_sz], isQuery=False)).type(torch.float).to(self.device)

        
        corpusfmaps_time = time.time() - corpusfmaps_start_time
        logger.info(f"Corpus fmaps shape: {self.corpus_fmaps.shape}, time={corpusfmaps_time}")
        
        self.query_embeds  = {}
        self.query_fmaps  = {}
        self.ground_truth = {}
        self.list_pos = {}
        self.list_neg = {} 
        self.list_total_arranged_per_query = {}
        self.labels_total_arranged_per_query = {}
        self.eval_batches = {}
        #for mode in ["train", "test", "val"]:
        for mode in ["train", "val"]:
        
            self.query_embeds[mode] = fetch_query_embeddings(conf, mode) 
            
            self.query_fmaps[mode] = torch.from_numpy(self.lsh.generate_fmap(conf.hashing.m_use, self.query_embeds[mode], isQuery=True)).type(torch.float).to(self.device)

            self.ground_truth[mode] = fetch_ground_truths(conf, mode) 
            self.list_pos[mode] = []
            self.list_neg[mode] = []
            self.list_total_arranged_per_query[mode] = []
            self.labels_total_arranged_per_query[mode] = []
            for q in range(self.query_embeds[mode].shape[0]) :
                for c in range(self.corpus_embeds.shape[0]): 
                    if c in self.ground_truth[mode][q]:
                        self.list_pos[mode].append(((q,c),1.0))
                        self.list_total_arranged_per_query[mode].append(((q,c),1.0))
                        self.labels_total_arranged_per_query[mode].append(1.0)
                    else:
                        self.list_neg[mode].append(((q,c),0.0))  
                        self.list_total_arranged_per_query[mode].append(((q,c),0.0))
                        self.labels_total_arranged_per_query[mode].append(0.0)
            self.eval_batches[mode] = {} 
            
        logger.info('Query embeds fetched and fmaps generated.')
        logger.info('Ground truth fetched.')
        self.preprocess_create_batches()

    def create_fmap_batches(self,mode):
        all_fmaps = torch.cat([self.query_fmaps[mode], self.corpus_fmaps])
        if mode == "train":
            all_fmaps = all_fmaps[torch.randperm(all_fmaps.shape[0])]
        
        self.batches = list(all_fmaps.split(self.BATCH_SIZE))
        self.num_batches = len(self.batches)
        return self.num_batches

    def fetch_fmap_batched_data_by_id(self,i):
        assert(i < self.num_batches)  
        return self.batches[i]

    def create_batches(self,list_all,VAL_BATCH_SIZE=10000):
        """
          create batches as is and return number of batches created
        """

        self.batches = []
        self.alists = []
        self.blists = []
        self.scores = []

        # pair_all, score_all = zip(* list_all)
        # as_all, bs_all = zip(* pair_all)
        list_all_np = np.array(list_all, dtype=object)
        score_all = list_all_np[:,1].tolist()
        temp = np.array(list_all_np[:,0].tolist())
        as_all = temp[:,0].tolist()
        bs_all = temp[:,1].tolist()

        for i in range(0, len(list_all), VAL_BATCH_SIZE):
          self.batches.append(list_all[i:i+VAL_BATCH_SIZE])
          self.alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
          self.blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
          self.scores.append(list(score_all[i:i+VAL_BATCH_SIZE]))

     
        self.num_batches = len(self.batches)  

        return self.num_batches


    def create_batches_with_p2n(self,mode):
      """
        Creates shuffled batches while maintaining given ratio
      """
      lpos = self.list_pos[mode]
      lneg = self.list_neg[mode]
      
      random.shuffle(lpos)
      random.shuffle(lneg)

      # lpos_pair, lposs = zip(*lpos)
      # lposa, lposb = zip(*lpos_pair)

      lpos_np = np.array(lpos, dtype=object)
      lposs = lpos_np[:,1].tolist()
      lpos_pair = np.array(lpos_np[:,0].tolist())
      lposa = lpos_pair[:,0].tolist()
      lposb = lpos_pair[:,1].tolist()

      # lneg_pair, lnegs = zip(*lneg)
      # lnega, lnegb = zip(*lneg_pair)

      lneg_np = np.array(lneg, dtype=object)
      lnegs = lneg_np[:,1].tolist()
      lneg_pair = np.array(lneg_np[:,0].tolist())
      lnega = lneg_pair[:,0].tolist()
      lnegb = lneg_pair[:,1].tolist()

      p2n_ratio = self.P2N
      batches_pos, batches_neg = [],[]
      as_pos, as_neg, bs_pos, bs_neg, ss_pos, ss_neg = [], [], [], [], [], []
      
      logger.info(f"self.BATCH_SIZE = {self.BATCH_SIZE}")
      if self.BATCH_SIZE > 0:
        npos = math.ceil((p2n_ratio/(1+p2n_ratio))*self.BATCH_SIZE)
        nneg = self.BATCH_SIZE-npos
        self.num_batches = int(math.ceil(max(len(lneg) / nneg, len(lpos) / npos)))
        pos_rep = int(math.ceil((npos * self.num_batches / len(lpos))))
        neg_rep = int(math.ceil((nneg * self.num_batches / len(lneg))))
        logger.info(f"Replicating lpos {pos_rep} times, lneg {neg_rep} times")
        lpos = lpos * pos_rep
        lposa = lposa * pos_rep
        lposb = lposb * pos_rep
        lposs = lposs * pos_rep

        lneg = lneg * neg_rep
        lnega = lnega * neg_rep
        lnegb = lnegb * neg_rep
        lnegs = lnegs * neg_rep

        logger.info(f"self.num_batches = {self.num_batches}")

        for i in tqdm.tqdm(range(self.num_batches)):
          try:
            batches_pos.append(lpos[i * npos:(i+1) * npos])
            as_pos.append(lposa[i * npos:(i+1) * npos])
            bs_pos.append(lposb[i * npos:(i+1) * npos])
            ss_pos.append(lposs[i * npos:(i+1) * npos])

            assert len(batches_pos[-1]) > 0
          except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info(batches_pos[-1], len(lpos), (i+1)*npos)

        for i in tqdm.tqdm(range(self.num_batches)):
          try:
            batches_neg.append(lneg[i * nneg:(i+1) * nneg])
            as_neg.append(lnega[i * nneg:(i+1) * nneg])
            bs_neg.append(lnegb[i * nneg:(i+1) * nneg])
            ss_neg.append(lnegs[i * nneg:(i+1) * nneg])
            assert len(batches_neg[-1]) > 0
          except Exception as e:
            logger.exception(e, exc_info=True)
            logger.info(batches_neg[-1], len(lneg), (i+1)*nneg)
      else:
        self.num_batches = 1
        batches_pos.append(lpos)
        batches_neg.append(lneg)
       
      self.batches = [a+b for (a,b) in zip(batches_pos[:self.num_batches],batches_neg[:self.num_batches])]
      self.alists = [list(a+b) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
      self.blists = [list(a+b) for (a,b) in zip(bs_pos[:self.num_batches],bs_neg[:self.num_batches])]
      self.scores = [list(a+b) for (a,b) in zip(ss_pos[:self.num_batches],ss_neg[:self.num_batches])]
      self.alists_tensorized = [torch.tensor(list(a+b)) for (a,b) in zip(as_pos[:self.num_batches],as_neg[:self.num_batches])]
      self.mode = mode

      return self.num_batches

    def preprocess_create_batches(self,VAL_BATCH_SIZE=10000):
        #for mode in ["train", "test", "val"]:
        for mode in ["train", "val"]:
            list_all_ap = self.list_pos[mode] + self.list_neg[mode]
            list_all_map = self.list_total_arranged_per_query[mode]
            label_map = self.labels_total_arranged_per_query[mode]
            self.eval_batches[mode] = {}
            for metric in ["ap", "map"]:
                self.eval_batches[mode][metric] = {}
                list_all = list_all_ap if metric=="ap" else list_all_map
                batches = []
                alists = []
                blists = []
                scores = []
                alists_tensorized = []

                # pair_all, score_all = zip(* list_all)
                # as_all, bs_all = zip(* pair_all)

                list_all_np = np.array(list_all, dtype=object)
                score_all = list_all_np[:,1].tolist()
                temp = np.array(list_all_np[:,0].tolist())
                as_all = temp[:,0].tolist()
                bs_all = temp[:,1].tolist()
                
                for i in range(0, len(list_all), VAL_BATCH_SIZE):
                  batches.append(list_all[i:i+VAL_BATCH_SIZE])
                  alists.append(list(as_all[i:i+VAL_BATCH_SIZE]))
                  blists.append(list(bs_all[i:i+VAL_BATCH_SIZE]))
                  scores.append(list(score_all[i:i+VAL_BATCH_SIZE]))
                  alists_tensorized.append(torch.tensor(list(as_all[i:i+VAL_BATCH_SIZE])))

                self.eval_batches[mode][metric]['batches'] = batches
                self.eval_batches[mode][metric]['alists'] = alists
                self.eval_batches[mode][metric]['blists'] = blists
                self.eval_batches[mode][metric]['scores'] = scores
                self.eval_batches[mode][metric]['alists_tensorized'] = alists_tensorized

    def create_batches(self,metric,mode):
      """
        create batches as is and return number of batches created
      """
      self.batches = self.eval_batches[mode][metric]['batches']
      self.alists = self.eval_batches[mode][metric]['alists']
      self.blists = self.eval_batches[mode][metric]['blists']
      self.scores = self.eval_batches[mode][metric]['scores']
      self.alists_tensorized = self.eval_batches[mode][metric]['alists_tensorized']
        
      self.num_batches = len(self.batches)  
      self.mode = mode

      return self.num_batches


    def fetch_batched_data_by_id_optimized(self,i):
        """             
        """
        assert(i < self.num_batches)  
        alist = self.alists[i]
        blist = self.blists[i]
        score = self.scores[i]
        query_tensors = self.query_fmaps[self.mode][alist]
        #query_set_sizes = self.query_set_sizes[alist]

        corpus_tensors = self.corpus_fmaps[blist]
        #corpus_set_sizes = self.corpus_set_sizes[blist]
        target = torch.tensor(score, device=self.device)
        return corpus_tensors, query_tensors, target, self.alists_tensorized[i] 

def evaluate_embeddings_similarity(model, sampler, mode):
    model.eval()
    npos = len(sampler.list_pos[mode])
    nneg = len(sampler.list_neg[mode])

    pred = []
    sign_pred = []
    tan_pred = []

    tanh_temp = 1.0 #we don't mess around with this here

    n_batches = sampler.create_batches("ap",mode)
    for i in tqdm.tqdm(range(n_batches)):
        #ignoring target values and qids here since not needed for AP ranking score 
        batch_corpus_tensors,  batch_query_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)
    
        corpus_hashcodes = model.forward(batch_corpus_tensors).data
        query_hashcodes  = model.forward(batch_query_tensors).data        
        sign_corpus_hashcodes = torch.sign(corpus_hashcodes) 
        sign_query_hashcodes = torch.sign(query_hashcodes)
        tan_corpus_hashcodes = torch.nn.Tanh()(tanh_temp*corpus_hashcodes) 
        tan_query_hashcodes  = torch.nn.Tanh()(tanh_temp*query_hashcodes)

        #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        prediction = (query_hashcodes * corpus_hashcodes).sum(-1)
        sign_prediction = (sign_query_hashcodes * sign_corpus_hashcodes).sum(-1)
        tan_prediction = (tan_query_hashcodes * tan_corpus_hashcodes).sum(-1)
        
        pred.append( prediction.data )
        sign_pred.append( sign_prediction.data )
        tan_pred.append( tan_prediction.data )

    all_pred = torch.cat(pred,dim=0) 
    all_sign_pred = torch.cat(sign_pred,dim=0) 
    all_tan_pred = torch.cat(tan_pred,dim=0) 
    labels = torch.cat((torch.ones(npos),torch.zeros(nneg)))
    ap_score = average_precision_score(labels.cpu(), all_pred.cpu())   
    sign_ap_score = average_precision_score(labels.cpu(), all_sign_pred.cpu())   
    tan_ap_score = average_precision_score(labels.cpu(), all_tan_pred.cpu())   
    
    # MAP computation
    all_ap = []
    all_sign_ap = []
    all_tan_ap = []
    pred = []
    sign_pred = []
    tan_pred = []
    n_batches = sampler.create_batches("map",mode)
    for i in tqdm.tqdm(range(n_batches)):
        #ignoring target values and qids here since not needed for AP ranking score 
        batch_corpus_tensors, batch_query_tensors, _, _ = sampler.fetch_batched_data_by_id_optimized(i)

        corpus_hashcodes = model.forward(batch_corpus_tensors).data
        query_hashcodes  = model.forward(batch_query_tensors).data        
        sign_corpus_hashcodes = torch.sign(corpus_hashcodes) 
        sign_query_hashcodes = torch.sign(query_hashcodes)
        tan_corpus_hashcodes = torch.nn.Tanh()(tanh_temp*corpus_hashcodes) 
        tan_query_hashcodes  = torch.nn.Tanh()(tanh_temp*query_hashcodes)

        
        #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        prediction = (query_hashcodes * corpus_hashcodes).sum(-1)
        sign_prediction = (sign_query_hashcodes * sign_corpus_hashcodes).sum(-1)
        tan_prediction = (tan_query_hashcodes * tan_corpus_hashcodes).sum(-1)

        pred.append( prediction.data )
        sign_pred.append( sign_prediction.data )
        tan_pred.append( tan_prediction.data )

    all_pred = torch.cat(pred,dim=0)
    all_sign_pred = torch.cat(sign_pred,dim=0) 
    all_tan_pred = torch.cat(tan_pred,dim=0) 
    labels = sampler.labels_total_arranged_per_query[mode]
    corpus_size = sampler.corpus_embeds.shape[0]
    
    for q_id in tqdm.tqdm(range(sampler.query_embeds[mode].shape[0])):
        q_pred = all_pred[q_id * corpus_size : (q_id+1) * corpus_size]
        q_sign_pred = all_sign_pred[q_id * corpus_size : (q_id+1) * corpus_size]
        q_tan_pred = all_tan_pred[q_id * corpus_size : (q_id+1) * corpus_size]
        q_labels = labels[q_id * corpus_size : (q_id+1) * corpus_size]
        ap = average_precision_score(q_labels, q_pred.cpu())
        sign_ap = average_precision_score(q_labels, q_sign_pred.cpu())
        tan_ap = average_precision_score(q_labels, q_tan_pred.cpu())
        all_ap.append(ap)
        all_sign_ap.append(sign_ap)
        all_tan_ap.append(tan_ap)
    return ap_score, all_ap, np.mean(all_ap),\
            sign_ap_score, all_sign_ap, np.mean(all_sign_ap) ,\
            tan_ap_score, all_tan_ap, np.mean(all_tan_ap) 



def run_fmap_gen(conf, curr_task):
    train_data = FmapDataLoader(conf)
    model = AsymFmapTrainer(conf).to(conf.fmap_training.device)

    cnt = 0
    for param in model.parameters():
        cnt=cnt+torch.numel(param)
    logger.info(f"no. of params in model: {cnt}")
    
    es = EarlyStoppingModule(conf.base_dir, curr_task, patience=conf.training.patience, delta=0.0001, logger=logger)
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=conf.training.learning_rate,
                                weight_decay=conf.training.weight_decay)
    

    best_val_map = 0
    run = 0

    while conf.training.run_till_early_stopping and run < conf.training.num_epochs:
        n_batches = train_data.create_batches_with_p2n(mode="train")
        epoch_loss =0

        start_time = time.time()
            
        for i in tqdm.tqdm(range(n_batches)):
            optimizer.zero_grad()
            batch_corpus_tensors, batch_query_tensors, batch_target, batch_query_ids = train_data.fetch_batched_data_by_id_optimized(i)
            loss = model.computeLoss(batch_corpus_tensors, batch_query_tensors, batch_target, batch_query_ids)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.item()   
        end_time = time.time()
        logger.info(f"Epoch: {run} loss: {epoch_loss} time: {end_time-start_time}")
        
        start_time = time.time()
        tr_ap_score,tr_all_ap,tr_map_score, tr_sign_ap_score,tr_all_sign_ap,tr_sign_map_score, tr_tan_ap_score,tr_all_tan_ap,tr_tan_map_score  = evaluate_embeddings_similarity(model,train_data,mode="train")
        logger.info(f"Run: {run} TRAIN ap_score: {tr_ap_score} map_score: {tr_map_score} sign_ap_score: {tr_sign_ap_score} sign_map_score: {tr_sign_map_score} tan_ap_score: {tr_tan_ap_score} tan_map_score: {tr_tan_map_score} Time: {time.time()-start_time}")
        start_time = time.time()
        ap_score,all_ap,map_score, sign_ap_score,all_sign_ap,sign_map_score, tan_ap_score,all_tan_ap,tan_map_score = evaluate_embeddings_similarity(model,train_data, mode="val")
        logger.info(f"Run: {run} VAL ap_score: {ap_score} map_score: {map_score} best_val_map_score: {best_val_map} sign_ap_score: {sign_ap_score} sign_map_score: {sign_map_score} tan_ap_score: {tan_ap_score} tan_map_score: {tan_map_score} Time: {time.time()-start_time}")

        state_dict = {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": run,
            "best_val_map": best_val_map,
            "val_map_score": map_score,
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'np_rng_state': np.random.get_state(),
            'random_state': random.getstate(),
            'patience': es.patience,
            'best_scores': es.best_scores,
            'num_bad_epochs': es.num_bad_epochs,
            'should_stop_now': es.should_stop_now,
        }

        state_dict =  es.check([map_score], state_dict)
        best_val_map = state_dict["best_val_map"]

        if es.should_stop_now:
            break
        run+=1
    
    #generate and dump fmap  pickles
    #IMP: Load best validation model here
    checkpoint = es.load_best_model()
    model.load_state_dict(checkpoint['model_state_dict'])      

    all_fmaps = {}
    corpus_fmaps = torch.zeros((train_data.corpus_embeds.shape[0], conf.fmap_training.tr_fmap_dim))
    bsz = 40000
    for i in tqdm.tqdm(range(0, train_data.corpus_embeds.shape[0],bsz)):
        corpus_fmaps[i:i+bsz,:] = model.forward(train_data.corpus_fmaps[i:i+bsz,:],isQ=False).data
    query_fmaps = {}
    #for mode in ["train", "test", "val"]:
    for mode in ["train", "val"]:
        query_fmaps[mode] =  model.forward(train_data.query_fmaps[mode],isQ=True).data
    all_fmaps['query'] = query_fmaps
    all_fmaps['corpus'] = corpus_fmaps
    logger.info(f"Dumping trained fmap pickle at {pickle_fp}")
    with open(pickle_fp, 'wb') as f:
        pickle.dump(all_fmaps, f)
        
        
if __name__ == "__main__":

    main_conf = OmegaConf.load("configs/config.yaml")
    cli_conf = OmegaConf.from_cli()
    data_conf = OmegaConf.load(f"configs/data_configs/{cli_conf.dataset.rel_mode}/{cli_conf.dataset.name}.yaml")
    # model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    hash_conf = OmegaConf.load(f"configs/hash_configs/{cli_conf.hashing.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, hash_conf, cli_conf)

    # NOTE: Below 3 lines should be same as first three in "check_pretrained_fmaps" function in fhash_trained.py/fhash.py
    temp_IN_ARCH = "L" +  "".join([f"RL_{dim}_" for dim in conf.fmap_training.hidden_layers])
    hashing_config_name_removal_set = {'device', 'embed_dim', 'subset_size', 'classPath'}
    hashing_conf_str = ",".join("{}{}".format(*i) for i in conf.hashing.items() if (i[0] not in hashing_config_name_removal_set))
    fmap_training_config_name_removal_set = {'model_name', 'classPath', 'device', 'hidden_layers'}
    fmap_training_conf_str = ",".join("{}{}".format(*i) for i in conf.fmap_training.items() if (i[0] not in fmap_training_config_name_removal_set))
    curr_task = conf.dataset.name + "," + hashing_conf_str + "," + fmap_training_conf_str + ","+ temp_IN_ARCH

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

    pickle_fp = conf.base_dir + "allPklDumps/fmapPickles/"+curr_task +"_fmap_mat.pkl"
    if not os.path.exists(pickle_fp):
        run_fmap_gen(conf, curr_task)

 
 