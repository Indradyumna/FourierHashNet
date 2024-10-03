import os
import torch

from sklearn.metrics import average_precision_score
import time
import numpy as np
import tqdm

class EarlyStoppingModule(object):
    """
    Module to keep track of validation score across epochs
    Stop training if score not imroving exceeds patience
    """

    def __init__(
        self, save_dir=".", task_name="TASK", patience=100, delta=0.005, logger=None
    ):
        self.save_dir = save_dir
        self.task_name = task_name
        self.patience = patience
        self.delta = delta
        self.logger = logger
        self.create_dirs()
        self.best_scores = None
        self.num_bad_epochs = 0
        self.should_stop_now = False

    def create_dirs(self):
        # Initial
        save_dir = os.path.join(self.save_dir, "initialModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.initial_model_path = os.path.join(save_dir, self.task_name)

        # Latest
        save_dir = os.path.join(self.save_dir, "latestModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.latest_model_path = os.path.join(save_dir, self.task_name)

        # Best
        save_dir = os.path.join(self.save_dir, "bestValidationModels")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.best_model_path = os.path.join(save_dir, self.task_name)

    def save_initial_state(self, state_dict):
        assert not os.path.exists(self.initial_model_path), f"{self.initial_model_path} already exists. Do you mean to resume"
        self.logger.info(f"saving initial model to {self.initial_model_path}")
        output = open(self.initial_model_path, mode="wb")
        
        torch.save(state_dict, output)
        output.close()

    def save_latest_model(self, state_dict):
        output = open(self.latest_model_path, mode="wb")
        torch.save(state_dict,output)
        output.close()

    def load_latest_model(self):
        assert os.path.exists(self.latest_model_path), f"{self.latest_model_path} does not exist. Do you mean to start fresh"

        self.logger.info(f"loading latest trained model from {self.latest_model_path}",)
        checkpoint = torch.load(self.latest_model_path)
        self.patience = checkpoint["patience"]
        self.best_scores = checkpoint["best_scores"]
        self.num_bad_epochs = checkpoint["num_bad_epochs"]
        self.should_stop_now = checkpoint["should_stop_now"]
        return checkpoint

    def save_best_model(self,state_dict):
        self.logger.info(f"saving best validated model to {self.best_model_path}")
        output = open(self.best_model_path, mode="wb")
        torch.save(state_dict,output)
        output.close()

    def load_best_model(self, device="cuda"):
        self.logger.info(f"loading best validated model from {self.best_model_path}")
        checkpoint = torch.load(self.best_model_path, map_location=device)
        return checkpoint

    def diff(self, curr_scores):
        return sum([cs - bs for cs, bs in zip(curr_scores, self.best_scores)])

    def check(self, curr_scores, state_dict):
        if self.best_scores is None:
            self.best_scores = curr_scores
            if "best_val_ap" in state_dict:
                state_dict["best_val_ap"] = state_dict["val_ap_score"]
            if "best_val_map" in state_dict:
                state_dict["best_val_map"] = state_dict["val_map_score"]
            if "best_neg_val_loss" in state_dict:
                state_dict["best_neg_val_loss"] = state_dict["neg_val_loss"]
            state_dict["best_scores"] = self.best_scores
            state_dict["num_bad_epochs"] = self.num_bad_epochs
            self.save_best_model(state_dict)
        elif self.diff(curr_scores) >= self.delta:
            self.num_bad_epochs = 0
            self.best_scores = curr_scores
            if "best_val_ap" in state_dict:
                state_dict["best_val_ap"] = state_dict["val_ap_score"]
            if "best_val_map" in state_dict:
                state_dict["best_val_map"] = state_dict["val_map_score"]
            if "best_neg_val_loss" in state_dict:
                state_dict["best_neg_val_loss"] = state_dict["neg_val_loss"]
            state_dict["best_scores"] = self.best_scores
            state_dict["num_bad_epochs"] = self.num_bad_epochs
            self.save_best_model(state_dict)
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs > self.patience:
                self.should_stop_now = True
        self.save_latest_model(state_dict)
        return state_dict
    
def pairwise_ranking_loss(pred_pos, pred_neg, margin):
    num_pos, dim = pred_pos.shape
    num_neg, _ = pred_neg.shape

    expanded_pred_pos = pred_pos.unsqueeze(1)
    expanded_pred_neg = pred_neg.unsqueeze(0)
    relu = torch.nn.ReLU()
    loss = relu(margin + expanded_pred_neg - expanded_pred_pos)
    mean_loss = torch.mean(loss, dim=(0, 1))

    return mean_loss


def compute_average_precision(model, pos_pairs, neg_pairs, dataset, return_pred_and_labels=False, return_running_time=False):
    assert not(return_running_time and return_pred_and_labels)
    all_pairs = pos_pairs + neg_pairs
    num_pos_pairs, num_neg_pairs = len(pos_pairs), len(neg_pairs)

    if return_running_time:
        total_running_time = 0
        total_batches = 0
    predictions = []
    num_batches = dataset.create_eval_batches(all_pairs)
    for batch_idx in range(num_batches):
        batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes, _ = dataset.fetch_batch_by_id(batch_idx)

        if return_running_time:
            start_time = time.time()

        model_output = model(batch_graphs, batch_graph_node_sizes, batch_graph_edge_sizes)

        if return_running_time:
            end_time = time.time()
            total_running_time += end_time - start_time
            total_batches += 1

        predictions.append(model_output.data)
    all_predictions = torch.cat(predictions, dim=0)
    all_labels = torch.cat([torch.ones(num_pos_pairs), torch.zeros(num_neg_pairs)])

    average_precision = average_precision_score(all_labels, all_predictions.cpu())
    if return_pred_and_labels:
        return average_precision, all_labels, all_predictions
    elif return_running_time:
        return average_precision, total_running_time, total_batches
    else:
        return average_precision
