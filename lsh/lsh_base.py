import torch 
import numpy as np
import random
import time 
from loguru import logger
from collections import defaultdict
import psutil
import heapq
from lsh.scoring import optimized_cosine_similarity, asym_sim, dot_sim, sigmoid_asym_sim, wjac_sim

def user_time():
  return psutil.Process().cpu_times().user  


scoring_func_dict = {"cos": optimized_cosine_similarity,
                    'hinge': asym_sim,
                    'dot': dot_sim,
                    'sighinge': sigmoid_asym_sim,
                    "wjac": wjac_sim}


class BaseLSH(object):
    def __init__(self,conf):
        self.num_hash_tables = conf.hashing.num_hash_tables
        self.K = conf.K  #TODO: Decide if to use this
        self.embed_dim = conf.dataset.embed_dim
        self.hcode_dim = conf.hashing.hcode_dim
        # No. of buckets in a hashTable is 2^subset_size
        self.subset_size = conf.hashing.subset_size
        assert(self.subset_size<=self.hcode_dim)
        
        self.device = conf.hashing.device
        self.DEBUG = conf.DEBUG  #Additional print statements
        
        self.powers_of_two = 1 << np.arange(self.subset_size - 1, -1, -1)


        # generates self.hash_functions, containing indices for hash functions
        self.init_hash_functions()
        
        self.scoring_func = scoring_func_dict[conf.hashing.FUNC]
        self.conf = conf
        
        ### Timing methods and funcs
        self.timing_methods = ["real", "user", "process_time"]
        self.timing_funcs = {"real": time.time, "user": user_time, "process_time": time.process_time}

    def init_hash_functions(self):
        """
            Each hash function is a random subset of the hashcode. 
        """
        #TODO: Revisit if this is needed
        if self.subset_size == self.hcode_dim:
            self.hash_functions = np.arange(self.hcode_dim)[None,:].astype(np.int64)
            return 
        
        hash_functions = []

        hash_code_dim = self.hcode_dim
        indices = list(range(hash_code_dim))
        for i in range(self.num_hash_tables):
            random.shuffle(indices)
            hash_functions.append(np.array( indices[:self.subset_size] ).astype(np.int64))
        self.hash_functions = np.stack(hash_functions)


    def index_corpus(self, corpus_embeds):
        """
            corpus_embeds: (N,d) numpy array
        """
        s = time.time()
        self.corpus_embeds = corpus_embeds
        self.corpus_hashcodes = self.fetch_RH_hashcodes(self.corpus_embeds,isQuery=False)
        assert(self.corpus_embeds.shape[0] == self.corpus_hashcodes.shape[0])
        self.num_corpus_items = self.corpus_embeds.shape[0]
        #generates self.hashcode_mat (containing +1/-1, used for bucketing)
        self.hashcode_mat = self.preprocess_hashcodes(self.corpus_hashcodes)
        #Assigns corpus items to buckets in each of the tables
        #generates dict self.all_hash_tables containing bucketId:courpusItemIDs
        self.bucketify()
        logger.info(f"Corpus indexed. Time taken {time.time()-s:.3f} sec")
        
    def fetch_RH_hashcodes(self, embeds, isQuery, qid=None):
        raise NotImplementedError()
    
    
    def preprocess_hashcodes(self,all_hashcodes): 
        all_hashcodes = np.sign(all_hashcodes)
        #edge case
        if (np.sign(all_hashcodes)==0).any(): 
            logger.info("Hashcode had 0 bits. replacing all with 1")
            all_hashcodes[all_hashcodes==0]=1
        return all_hashcodes

    def assign_bucket(self,function_id,node_hash_code):
        func = self.hash_functions[function_id]
        # convert sequence of -1 and 1 to binary by replacing -1 s to 0
        binary_id = np.take(node_hash_code,func)
        binary_id[binary_id<0] = 0
        bucket_id = (self.powers_of_two@binary_id).astype(self.powers_of_two.dtype)  
        return bucket_id

    def bucketify(self): 
        """
          For all hash functions: x
            Loop over all corpus items
              Assign corpus item to bucket in hash table corr. to hash function 
        """ 
        s = time.time()
        self.all_hash_tables = []
        for func_id in range(self.num_hash_tables): 
            hash_table = defaultdict(list)#{}
            #for idx in range(2**self.subset_size): 
            #    hash_table[idx] = []
            for item in range(self.num_corpus_items):
                hash_table[self.assign_bucket(func_id,self.hashcode_mat[item])].append(item)
            self.all_hash_tables.append(hash_table)
    
    def pretty_print_hash_tables(self,topk): 
        """
            I've found this function useful to visualize corpus distribution across buckets
        """
        for table_id in range(self.num_hash_tables): 
            len_list = sorted([len(self.all_hash_tables[table_id][bucket_id]) for bucket_id in range(2**self.subset_size)])[::-1] [:topk]
            len_list_str = [str(i) for i in len_list]
            lens = '|'.join(len_list_str)
            print(lens)
            
            
    def heapify(self, q_embed, candidate_list, K):
        """
            use q_embed , candidate_list, corpus_embeds to fetch top K items
        """
        time_dict = {tm: {} for tm in self.timing_methods}
        #other_data_dict = {}
        
        if len(candidate_list) == 0:
            for k in time_dict.keys():
                time_dict[k]['score_computation_time'] = 0.0
                time_dict[k]['heap_procedure_time'] = 0.0
                time_dict[k]['take_time'] = 0.0
                time_dict[k]['sort_procedure_time'] = 0.0
            #other_data_dict['len_candidate_list'] = 0
            return list(), list(), time_dict#, other_data_dict
        
        score_timer_start = {}
        for tm in self.timing_methods:
            score_timer_start[tm] = self.timing_funcs[tm]()
            
        take_start = {}
        for tm in self.timing_methods:
            take_start[tm] = self.timing_funcs[tm]()
        candidate_corpus_embeds = np.take(self.corpus_embeds,candidate_list,axis=0)
        for tm in self.timing_methods:
            time_dict[tm]['take_time'] = self.timing_funcs[tm]() - take_start[tm]
        
        scores = self.scoring_func((self.conf, q_embed, candidate_corpus_embeds))
        if len(scores.shape)==2:
            scores = scores.squeeze(0)


        for tm in self.timing_methods:
            time_dict[tm]['score_computation_time'] = self.timing_funcs[tm]() - score_timer_start[tm]
        heap_timer_start = {}
        for tm in self.timing_methods:
            heap_timer_start[tm] = self.timing_funcs[tm]()

        if K >= 0:    
            score_heap = []
            heap_size = 0

            for i in range(len(candidate_list)):
                if heap_size<K: 
                    heap_size = heap_size+1
                    heapq.heappush(score_heap,(scores[i],candidate_list[i]))
                else:
                    heapq.heappushpop(score_heap,(scores[i],candidate_list[i]))

            for tm in self.timing_methods:
                time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
            scores,corpus_ids =  list(zip (*score_heap))

        else:
            corpus_ids = candidate_list
            for tm in self.timing_methods:
                time_dict[tm]['heap_procedure_time'] = self.timing_funcs[tm]() - heap_timer_start[tm]
        sort_timer_start = {}
        for tm in self.timing_methods:
            sort_timer_start[tm] = self.timing_funcs[tm]()

        scores_arr = np.array(scores)
        corpus_ids_arr = np.array(corpus_ids)
        sorted_ids = np.argsort(-scores_arr)
        sorted_scores = scores_arr[sorted_ids]
        sorted_corpus_ids = corpus_ids_arr[sorted_ids]
        for tm in self.timing_methods:
            time_dict[tm]['sort_procedure_time'] = self.timing_funcs[tm]() - sort_timer_start[tm]
        #other_data_dict['len_candidate_list'] = len(sorted_corpus_ids)
        return list(sorted_scores), list(sorted_corpus_ids), time_dict#, other_data_dict


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
                
            q_hashcode =  self.preprocess_hashcodes(self.fetch_RH_hashcodes(q_embed,isQuery=True)).squeeze()

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


