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
    hash_conf = OmegaConf.load(f"configs/hash_configs/{cli_conf.hashing.name}.yaml")
    conf = OmegaConf.merge(main_conf, data_conf, hash_conf, cli_conf)  
    ### cli-inputs #
    # conf.hashing.FUNC = "sighinge"
    # conf.dataset.name = "msnbc294_3"
    # conf.hashing.name = "RH"

    ground_truth = fetch_ground_truths(conf, "test")

    all_hashing_result = []

    for subset_size in [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,25,30]:
        conf.hashing.subset_size = subset_size
        hash_op  = run_lsh(conf)   
        all_hashing_result.append(compute_all_scores(hash_op,ground_truth))
    
    
    fp = f"{conf.base_dir}allPklDumps/hashResultPickles/{conf.dataset.name}_{conf.dataset.rel_mode}_{conf.hashing.FUNC}_{conf.hashing.name}.pkl"

    pickle.dump(all_hashing_result, open(fp,"wb"))

# python -m scripts.lsh_scripts.hashing_untrained_baselines dataset.name="msweb294" dataset.rel_mode="fhash"  hashing.FUNC="sighinge" hashing.name="RH"
# python -m scripts.lsh_scripts.hashing_untrained_baselines dataset.name="msweb294" dataset.rel_mode="fhash"  hashing.FUNC="dot" hashing.name="DPRH"
