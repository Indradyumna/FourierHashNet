import multiprocessing
import pickle
import time
from omegaconf import OmegaConf
from lsh.main import run_lsh, compute_all_scores
from src.embeddings_loader import  fetch_ground_truths


if __name__ == "__main__":
    
    main_conf = OmegaConf.load("configs/config.yaml")
    cli_conf = OmegaConf.from_cli()
    data_conf = OmegaConf.load(f"configs/data_configs/{cli_conf.dataset.rel_mode}/{cli_conf.dataset.name}.yaml")
    # model_conf = OmegaConf.load(f"configs/model_configs/{cli_conf.model.name}.yaml")
    hash_conf = OmegaConf.load(f"configs/hash_configs/{cli_conf.hashing.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, hash_conf, cli_conf)


    ### cli-inputs #
    # conf.fmap_training.tr_fmap_loss = "BCE3"
    # conf.hashing.FUNC = "sighinge"
    # conf.hashcode_training.LOSS_TYPE = "query_aware"
    # conf.hashcode_training.QA_subset_size = 8
    # conf.hashing.m_use = 10
    # conf.dataset.name = "msnbc294_3"
    # conf.hashing.name = "Fhash_Trained"


    SENTINEL = None 
    
    s = time.time()

    #NOTE
    # conf.hashing.T = 3.0
    #conf.hashing.T = 38.0
 

    variations = [(0.05, 10,"Asym","AsymFmapCos")] #TODO check that always this. Hardcoded in conf.fmap_training

    c1_val = [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55,
           0.6 , 0.65, 0.7 , 0.75, 0.8 , 0.85]
    
    # TODO: dataset.name=ptc_fr dataset.rel_mode=ged 
    fp = f"{conf.base_dir}allPklDumps/hashResultPickles/{conf.dataset.name}_{conf.dataset.rel_mode}_{conf.hashing.FUNC}_{conf.hashing.name}_{conf.hashcode_training.LOSS_TYPE}{conf.hashcode_training.QA_subset_size }{conf.fmap_training.tr_fmap_loss}muse{conf.hashing.m_use}.pkl"
    ground_truth = fetch_ground_truths(conf, "test")

    def inner_foo(conf,dval,d):
        if conf.hashcode_training.LOSS_TYPE == "query_aware":
            conf.hashcode_training.C1 = dval
        else:
            conf.hashcode_training.DECORR = dval

        tmp_list = []
        for subset_size in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
            conf.hashing.subset_size = subset_size   
            hash_op = run_lsh(conf)
            tmp_list.append(compute_all_scores(hash_op,ground_truth))
            
        d.put((dval, tmp_list))
        d.put(SENTINEL)
        return
    

    queue = multiprocessing.Queue()
    procs = []
    all_dict = {} 
    all_metric_list_parallel = []
    
    for margin,v1,v2,v3 in variations: # unnecessary, but OK
        for dval in c1_val:

            p =  multiprocessing.Process(target=inner_foo, args=(conf,dval,queue))
            procs.append(p)
            p.start()
            
        seen_sentinel_count = 0
        while seen_sentinel_count < len(c1_val):
            a = queue.get()
            if a is SENTINEL:
                seen_sentinel_count += 1
            else:
                all_dict [a[0]] = a[1]



        for p in procs: 
            p.join()
    
        for dval in c1_val:
            all_metric_list_parallel.extend(all_dict[dval])
            
    pickle.dump(all_metric_list_parallel, open(fp,"wb"))    
    print(time.time()-s)



# python -m scripts.lsh_scripts.hashing_fhash dataset.name="msweb294" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="Fhash_Trained" hashcode_training.QA_subset_size=8 fmap_training.tr_fmap_loss="BCE3" hashcode_training.LOSS_TYPE="query_aware"
