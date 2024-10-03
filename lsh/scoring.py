import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


def optimized_cosine_similarity(inputs):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the cosine similarity cos(a[i],b[j]) for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    (_, a, b1) = inputs
    eps = 1e-8
    return (a@b1.T)/((np.linalg.norm(a,axis=1)[:,None] + eps)*(np.linalg.norm(b1,axis=1)[None,:]+eps))

def pairwise_cosine_sim(inputs):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    """
    (_, a, b) = inputs
    return cosine_similarity(a, b)

def np_sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_asym_sim(inputs):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    (conf, query_embeds, corpus_embeds) = inputs
    sig_in = np.sum(np.maximum(0, query_embeds-corpus_embeds), axis=1)
    sig_out = np_sigmoid(conf.dataset.sigmoid_a*sig_in + conf.dataset.sigmoid_b)
    return sig_out


def asym_sim(inputs):
    '''
        query_embeds: 1 x d or m x d
        corpus_embeds: m x d
        output sim: m x 1
    '''
    (_, query_embeds, corpus_embeds) = inputs
    sim =  -np.sum(np.maximum(0, query_embeds-corpus_embeds), axis=1)
    return sim

def dot_sim(inputs):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the dot similarity a[i]*b[j] for all i and j.
    :return: Matrix with res[i][j]  = \sum(a[i]*b[j])
    """
    (_, a, b) = inputs 
    return (a@b.T) 

def hinge_sim(inputs):
    """
    input dim: a (m x d), b (n x d)
    output dim: m x n
    Computes the asym hinge similarity -max(0,a[i]- b[j]) for all i and j.
    :return: Matrix with res[i][j]  = -max(0,a[i]- b[j])
    """
    (_, a, b) = inputs 
    return -(np.maximum((a[:,None,:]-b[None,:,:]),0)).sum(-1)

def wjac_sim(inputs):
    '''
        query_embeds: 1 x d 
        corpus_embeds: m x d
        output sim: m x 1
    '''
    (_, a, b) = inputs 
    return np.minimum(a,b).sum(-1)/(np.maximum(a,b).sum(-1)+1e-8)


def pairwise_ranking_loss_similarity(predPos, predNeg, margin):

    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    ell = margin + expanded_2 - expanded_1
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(n_1*n_2)

def pairwise_ranking_loss_similarity_per_query(predPos, predNeg, qidPos, qidNeg, margin):
    
    assert qidPos.shape == predPos.shape and qidNeg.shape == predNeg.shape, f"qidPos.shape: {qidPos.shape}, predPos.shape: {predPos.shape}, qidNeg.shape: {qidNeg.shape}, predNeg.shape: {predNeg.shape}"
    
    n_1 = predPos.shape[0]
    n_2 = predNeg.shape[0]
    dim = predPos.shape[1]

    expanded_1 = predPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_qid_1 = qidPos.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_2 = predNeg.unsqueeze(0).expand(n_1, n_2, dim)
    expanded_qid_2 = qidNeg.unsqueeze(0).expand(n_1, n_2, dim)

    ell = margin + expanded_2 - expanded_1
    loss = torch.nn.ReLU()(ell) * (expanded_qid_1 == expanded_qid_2)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss/(torch.sum(expanded_qid_1 == expanded_qid_2))#, torch.sum(expanded_qid_1 == expanded_qid_2) / (n_1 * n_2)

