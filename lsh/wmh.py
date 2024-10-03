from lsh.lsh_base import BaseLSH
from loguru import logger
import numpy as np
from drhash import WeightedMinHash
from scipy.sparse import csc_matrix
import time

def get_wmh_hcode(wmh_type, embeds, hcode_dim):
    wmh = WeightedMinHash.WeightedMinHash(csc_matrix(embeds.T), hcode_dim)
    if wmh_type =="minhash":   
        wmh_op = wmh.minhash()
    elif wmh_type =="gollapudi2":   
        wmh_op = wmh.gollapudi2() 
    elif wmh_type =="icws":   
        wmh_op = wmh.icws()  
    elif wmh_type =="licws":   
        wmh_op = wmh.licws()  
    elif wmh_type =="pcws":   
        wmh_op = wmh.pcws()  
    elif wmh_type =="ccws":   
        wmh_op = wmh.ccws()  
    elif wmh_type =="i2cws":   
        wmh_op = wmh.i2cws()  
    elif wmh_type =="chum":   
        wmh_op = wmh.chum()  
    elif wmh_type =="shrivastava":   
        wmh_op = wmh.shrivastava()  
    else:
        raise NotImplementedError()
        
    if len(wmh_op) ==3 : 
        hcodes = np.array([str(x)+str(y) for x,y  \
                  in zip(wmh_op[0].flatten(),wmh_op[1].flatten())]).reshape(wmh_op[0].shape)
    elif len(wmh_op) ==2 : 
        hcodes = np.array([str(x) for x in wmh_op[0].flatten()]).reshape(wmh_op[0].shape)
    else:
        raise NotImplementedError()

    return hcodes.squeeze()

class WMH(BaseLSH):
    """
       Random Hyperplane LSH -- Cosine simimlarity hashing
    """
    def __init__(self, conf): 
        super(WMH, self).__init__(conf)
        
        self.wmh_type = conf.hashing.wmh_type


    # DrHash misbehaves if there is a zero vector embedding in the corpus
    # Empty set is represented by zero vector embedding in our models
    # So we remove zero vector embeddings from the corpus in a preprocessing step when applying  DrHash
    def index_corpus(self, corpus_embeds):
        """
            corpus_embeds: (N,d) numpy array
        """
        s = time.time()
        rows_idx = []
        for idx in range(len(corpus_embeds)):
            if all(corpus_embeds[idx]==0):
                rows_idx.append(idx)
        self.corpus_embeds = np.delete(corpus_embeds,rows_idx,axis=0)

        self.corpus_hashcodes = get_wmh_hcode(self.wmh_type, self.corpus_embeds, self.hcode_dim)
        assert(self.corpus_embeds.shape[0] == self.corpus_hashcodes.shape[0])
        self.num_corpus_items = self.corpus_embeds.shape[0]
        #generates self.hashcode_mat (containing +1/-1, used for bucketing)
        self.hashcode_mat = self.preprocess_hashcodes(self.corpus_hashcodes)
        #Assigns corpus items to buckets in each of the tables
        #generates dict self.all_hash_tables containing bucketId:courpusItemIDs
        self.bucketify()
        logger.info(f"Corpus indexed. Time taken {time.time()-s:.3f} sec")
        
    def preprocess_hashcodes(self,all_hashcodes):
        # No preprocessing needed
        return all_hashcodes

    def assign_bucket(self,function_id,node_hash_code):
        func = self.hash_functions[function_id]
        
        return '_'.join(node_hash_code[func])

    def pretty_print_hash_tables(self,topk):
        """
            I've found this function useful to visualize corpus distribution across buckets
        """
        for table_id in range(self.num_hash_tables): 
            len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in self.all_hash_tables[table_id].keys()])[::-1] [:topk]
            len_list_str = [str(i) for i in len_list]
            lens = '|'.join(len_list_str)
            print(lens)
            

    def retrieve(self, q_embed, K, no_bucket=False, qid=None): 
        """
            Input : query_embed : to compute actual scores/distances
                      shape is (1*d)
            Input : K : top K similar items to return
            Output : top K items, time taken for retrieval, accuracy? 

            given query and a number k, find the top k closest corpus items 
            loop over al hash_tables: 
              map query to corr bucket: 
                compute Asymmetric similarity between query and each corpus item in bucket and update min heap
        """
        #Given input query_embed: generate query_hashcode : to ID query bucket 
        start_hashcode_gen = {}
        end_hashcode_gen = {}
        if no_bucket:
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = 0
                end_hashcode_gen[tm] = 0
        else:
            for tm in self.timing_methods:
                start_hashcode_gen[tm] = self.timing_funcs[tm]()
                
            #WMH(DrHash) toolkit throws exception for zero vectors. Adding small random noise
            if all(q_embed[0]==0):
                q_embed = q_embed + np.random.normal(0,1e-8,len(q_embed[0]))
            q_hashcode = get_wmh_hcode(self.wmh_type, q_embed, self.hcode_dim)
            for tm in self.timing_methods:
                end_hashcode_gen[tm] = self.timing_funcs[tm]()

        
        start_candidate_list_gen = {}
        end_candidate_list_gen = {}
        if no_bucket:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]() 
            #We consider all corpus items  
            candidate_list = list(range(self.num_corpus_items))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()
        else:
            for tm in self.timing_methods:
                start_candidate_list_gen[tm] = self.timing_funcs[tm]()
            #We use q hashcode to identify buckets, and take union of corpus items into candidate set
            candidate_list = []
            for table_id in range(self.num_hash_tables): 
                #identify bucket 
                bucket_id = self.assign_bucket(table_id,q_hashcode)
                candidate_list.extend(self.all_hash_tables[table_id][bucket_id])

            #remove duplicates from candidate_list
            candidate_list = list(set(candidate_list))
            for tm in self.timing_methods:
                end_candidate_list_gen[tm] = self.timing_funcs[tm]()

            if self.DEBUG:
                print("No. of candidates found", len(candidate_list))


        scores, corpus_ids, time_dict = self.heapify (q_embed,candidate_list, K)

        for tm in self.timing_methods:
            time_dict[tm]['candidate_list_gen_time'] = end_candidate_list_gen[tm] - start_candidate_list_gen[tm]  
            time_dict[tm]['hashcode_gen_time'] = end_hashcode_gen[tm] - start_hashcode_gen[tm]
        return len(candidate_list),  scores,corpus_ids, time_dict        

