import torch
from GMN.segment import unsorted_segment_sum
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from sklearn.metrics import average_precision_score
import numpy as np

def dot_product_score(x,y):
    return (x*y).sum(-1)

def cosine_score(x,y):
    return torch.nn.functional.cosine_similarity(x,y)

def euclidean_distance_score(x, y):
    """This is the squared Euclidean distance based score.
    Returning negative of the distance as score"""
    return -torch.sum((x - y) ** 2, dim=-1)

def hinge_distance_score(x, y):
    """ Hinge distance based score.
    """
    return -torch.sum(torch.nn.ReLU()(x-y),dim=-1)

def min_score(x, y):
    """ Min score between two vectors.
    """
    return torch.min(x,y).sum(-1)

def l1_score(x,y):
    return -torch.sum(torch.abs(x-y),dim=-1)


def sigmoid_hinge_sim(sigmoid_a, sigmoid_b, x, y):
    sig_in= torch.sum(torch.nn.ReLU()(x-y),dim=-1)

    return torch.nn.Sigmoid()(sigmoid_a*sig_in + sigmoid_b)

#create a function dictionary for scoring layers. map names to functions.
scoring_functions = {
    "dot": dot_product_score,
    "cos": cosine_score,
    "euc": euclidean_distance_score,   
    "hinge": hinge_distance_score,
    "sighinge": sigmoid_hinge_sim,
    "min": min_score,
    "l1": l1_score,
}



