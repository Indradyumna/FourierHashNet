import multiprocessing
import pickle
import time
from omegaconf import OmegaConf
from lsh.main import run_lsh, compute_all_scores
from src.embeddings_loader import  fetch_ground_truths
from src.embeddings_loader import fetch_corpus_embeddings


if __name__ == "__main__":
    main_conf = OmegaConf.load("configs/config.yaml")
    cli_conf = OmegaConf.from_cli()
    data_conf = OmegaConf.load(f"configs/data_configs/{cli_conf.dataset.rel_mode}/{cli_conf.dataset.name}.yaml")
    hash_conf = OmegaConf.load(f"configs/hash_configs/{cli_conf.hashing.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, hash_conf, cli_conf)  
  
    ### cli-inputs #
    # conf.hashing.FUNC = "sighinge"
    # conf.dataset.name = "msnbc294_3"
    # conf.hashing.name = "wmh"

    conf.hashing.wmh_type = "chum"

    SENTINEL = None 
    
    s = time.time()

    # ============== INFO  =================
    # WMH tool cannot handle zero embedding in corpus. 
    # Our model produces zero embeddings for empty set.
    # So we check for zero embeddings and remove them before scoring.
    corpus_embeds = fetch_corpus_embeddings(conf)
    row_map = {}
    curr_idx = 0
    for idx in range(len(corpus_embeds)):
        if not all(corpus_embeds[idx]==0):
            row_map[curr_idx] = idx
            curr_idx = curr_idx+1

    def adjust_for_wmh_gt(op):
        for qidx in range(len(op[0]['asymhash'])):
            op[0]['asymhash'][qidx]=op[0]['asymhash'][qidx][:2]+\
                                    ([row_map[idx] for idx in op[0]['asymhash'][qidx][2]],)+\
                                    op[0]['asymhash'][qidx][3:]
        return op


    ground_truth = fetch_ground_truths(conf, "test")

    def inner_foo(conf,wmh_type,d):
      conf.hashing.wmh_type = wmh_type
      tmp_list = []

      for subset_size in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30]:
          conf.hashing.subset_size = subset_size
          hash_op  = run_lsh(conf)       
          tmp_list.append(compute_all_scores(adjust_for_wmh_gt(hash_op),ground_truth))
      
      d.put((wmh_type, tmp_list))
      d.put(SENTINEL)



    queue = multiprocessing.Queue()
    procs = []
 
    all_hashing_result = {}

    algo_list = ["minhash", "gollapudi2", "icws", "pcws", "ccws", "i2cws", "chum","licws"] 
    
    for ver in algo_list:   
      p =  multiprocessing.Process(target=inner_foo, args=(conf,ver,queue))
      procs.append(p)
      p.start()

    seen_sentinel_count = 0
    while seen_sentinel_count < len(algo_list):
        a = queue.get()
        if a is SENTINEL:
            seen_sentinel_count += 1
        else:
            all_hashing_result[a[0]] = a[1]

    for p in procs: 
        p.join()

    
    fp = f"{conf.base_dir}allPklDumps/hashResultPickles/{conf.dataset.name}_{conf.dataset.rel_mode}_{conf.hashing.FUNC}_{conf.hashing.name}.pkl"

    pickle.dump(all_hashing_result, open(fp,"wb"))


# python -m scripts.lsh_scripts.hashing_wmh dataset.name="msweb294" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="WMH"
# python -m scripts.lsh_scripts.hashing_wmh dataset.name="msweb294" dataset.rel_mode="fhash"  hashing.FUNC="wjac" hashing.name="WMH"
