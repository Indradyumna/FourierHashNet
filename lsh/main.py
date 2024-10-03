from src.embeddings_loader import fetch_corpus_embeddings, fetch_query_embeddings, fetch_ground_truths
from utils.utils import *
import torch 
import random
import numpy as np
from loguru import logger


def set_seed():
  seed = 4
  random.seed(seed)
  np.random.seed(seed + 1)
  torch.manual_seed(seed + 2)
  torch.backends.cudnn.deterministic = False


def run_lsh(conf):
    corpus_embeds = fetch_corpus_embeddings(conf)

    query_embeds = fetch_query_embeddings(conf, "test")
    ground_truth = fetch_ground_truths(conf, "test")
  
    num_qitems = query_embeds.shape[0]
    assert(len(ground_truth.keys()) == num_qitems)

    set_seed()
    #This will init the k hash functions
    lsh = get_class(f"{conf.hashing.classPath}.{conf.hashing.name}")(conf)#.to(conf.hashing.device)

    #This will generate feature maps and index corpus items
    lsh.index_corpus(corpus_embeds)

    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash']= [] #TODO: replace asymhash everywhere. It is there if we need to put additional markers
    
    for qid,qemb in enumerate(query_embeds): 
        #reshape qemb to 1*d
        all_hashing_info_dict['asymhash'].append(lsh.retrieve(qemb[None,:],conf.K, no_bucket=False,qid=qid))


    time_logger_dict = time_dict_parser(all_hashing_info_dict,len(query_embeds))
    
    return all_hashing_info_dict, time_logger_dict


def time_dict_parser(all_hashing_info_dict,num_q):
    # =====================Time Analysis================================
    time_logger_dict = {}
    # "asymhash", "symhash"
    for k in all_hashing_info_dict.keys(): 
        time_logger_dict[k] = {}
        #"real", "user", "process_time"
        for k1 in all_hashing_info_dict[k][0][3].keys(): 
            time_logger_dict[k][k1] = {}
            # "score_computation_time", "heap_procedure_time", "take_time", "sort_procedure_time",  "hashcode_gen_time", "candidate_list_gen_time"
            for k2 in all_hashing_info_dict[k][0][3][k1].keys(): 
                time_logger_dict[k][k1][k2] = 0
                for qidx in range(num_q):
                    try:
                        time_logger_dict[k][k1][k2] += all_hashing_info_dict[k][qidx][3][k1][k2]
                    except:
                        raise ValueError(f"Error in all_hashing_info_dict[{k}] [{qidx}] [{3}] [{k1}] [{k2}]")

    
    return  time_logger_dict



def compute_all_scores(op_hash,ground_truth):
    all_hashing_info_dict = {}
    all_hashing_info_dict['asymhash'] = op_hash[0]['asymhash'] #time_logger_dict unused now
 
    all_topK_score_10 = []
    num_evals = []
    all_customap_hash = []


    num_qitems = len(op_hash[0]['asymhash'])

    for qidx in range(num_qitems):
        all_topK_score_10.append(compute_topK_score(all_hashing_info_dict['asymhash'][qidx][1], 10))
        num_evals.append(all_hashing_info_dict['asymhash'][qidx][0])
        all_customap_hash.append(custom_ap(set(ground_truth[qidx]),all_hashing_info_dict['asymhash'][qidx][2],all_hashing_info_dict['asymhash'][qidx][1],-1, len(set(ground_truth[qidx]))))
    
    
    return np.mean(np.array(all_topK_score_10)),\
            np.mean(num_evals), np.mean(all_customap_hash)

def custom_ap(ground_truth, pred_cids, pred_scores, K, len_gt = None):
    """
        ground_truth : set of relevant corpus ids
        pred_cids : list of predicted corpus ids
        pred_scores: list of predicted scores for the pred_cids
        K : required top K items (only needed to check and throw exception)
    """
    if K>=0:
        try:
            assert len(pred_cids)==K
        except Exception as e:
            logger.exception(e)
            logger.info(f"# ground truth={len(ground_truth)}, # preds={len(pred_cids)}")
        
    sorted_pred_scores = sorted(((e, i) for i, e in enumerate(pred_scores)), reverse=True)
    sum_precision= 0 
    positive_count = 0
    position_count = 0
    for sc, idx in sorted_pred_scores:
        position_count += 1
        #check if label=1
        if pred_cids[idx] in ground_truth: 
             positive_count +=1 
             sum_precision += (positive_count/position_count)
    if len_gt is not None: 
        average_precision = sum_precision/len_gt
    else:
        average_precision = sum_precision/len(ground_truth)
    return average_precision

def compute_topK_score( hash_scores, K, M=0):
    """
    """
    if len(hash_scores)==0:
        #TODO: Discuss
        total_hash_score = 0# min(nohash_scores)
    else:
        total_hash_score= np.sum(np.array(hash_scores[:K]) + M)

    return total_hash_score