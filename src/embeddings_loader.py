import pickle
from loguru import logger
import numpy as np

def fetch_ground_truths(conf, mode):
    logger.info('Fetching ground truth.')
    fp = f"{conf.base_dir}fhash_data/{conf.dataset.name}_embeds_{conf.hashing.FUNC}.pkl"
    all_d = pickle.load(open(fp,"rb"))
    logger.info(f"Loading ground truth labels from {fp}")
    return all_d[f'{mode}_positive_labels']


def fetch_query_embeddings(conf, mode):
    logger.info('Fetching query embeddings.')
    embed_fp = f"{conf.base_dir}fhash_data/{conf.dataset.name}_embeds_{conf.hashing.FUNC}.pkl"
    all_d = pickle.load(open(embed_fp,"rb"))
    logger.info(f"From {embed_fp}")
    return all_d[f'{mode}_q']

def fetch_corpus_embeddings(conf):
    logger.info('Fetching corpus embeddings.')
    embed_fp = f"{conf.base_dir}fhash_data/{conf.dataset.name}_embeds_{conf.hashing.FUNC}.pkl"
    all_d = pickle.load(open(embed_fp,"rb"))
    logger.info(f"From {embed_fp}", embed_fp)
    return all_d['all_c'].astype(dtype=np.float32)
