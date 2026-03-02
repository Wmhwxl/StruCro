"""Evaluate the model on semantic similarity task"""
import os
import pickle
import json
import torch
import pandas as pd
from sklearn.metrics import ndcg_score
from src.dataset import load_split_data
from src.utils import build_model_config, bulid_graph_val, bulid_graph_train
from src.collator import ValCollator, TrainCollator
from src.StruCro import StruCro
from src.GraphBindingModelv3 import GraphBindingModelv3
from src.trainer import BindingTrainer
from transformers import TrainingArguments
from transformers.trainer_utils import (
    EvalPrediction,
)
import fire
import numpy as np
from typing import Dict, List, Optional
from src.dataset import TrainDataset, ValDataset
import csv
from collections import defaultdict

def compute_metrics(inputs: EvalPrediction) -> Dict:
    """Compute the metrics for the prediction, including node type verification."""
    metrics = defaultdict(list)
    predictions_list = inputs.predictions
    num_samples = len(predictions_list)

    predictions_dict = {
        "node_index": [],
        "node_type": [],
        "prediction": defaultdict(list),
        "label": defaultdict(list)
    }

    # Extracting predictions and labels
    for key, tensor in predictions_list.items():
        if key == 'prediction':
            predictions_dict["prediction"] = tensor
        elif key == 'label':
            predictions_dict["label"] = tensor
        elif key == 'node_index':
            predictions_dict["node_index"] = tensor.tolist()
        elif key == 'node_type':
            predictions_dict["node_type"] = tensor.tolist()

    node_indices = predictions_dict["node_index"]
    node_types = predictions_dict["node_type"]
    all_tail_types = list(predictions_dict['prediction'].keys())

    # Track node type correctness
    node_type_accuracies = defaultdict(list)

    # For each tail type (i.e., each relation)
    for tail_type in all_tail_types:
        preds, labels = predictions_dict['prediction'][tail_type], predictions_dict['label'][tail_type]
        total_ranks = []
        total_weights = []
        hit_at_1 = []
        hit_at_3 = []
        hit_at_10 = []

        for i in range(num_samples):
            node_types_i = node_types[i]  # This is the true node type
            pred, label = preds[i], labels[i]
            label = label[label != -100]  # Filter out invalid labels
        
            
            ###### MRR #####
            pred = pred.numpy().flatten()
            label = label.numpy().flatten()
            ranks = []
            for true_label in label:
                if true_label in pred:
                    rank = np.where(pred == true_label)[0][0] + 1  # Rank starts from 1
                    #ranks.append(rank / len(pred))  # Add the reciprocal rank for MRR
                    ranks.append(1 / rank)
                    # Hit@K calculation
                    hit_at_1.append(1 if rank <= 1 else 0)
                    hit_at_3.append(1 if rank <= 3 else 0)
                    hit_at_10.append(1 if rank <= 10 else 0)

            if ranks:
                avg_rank = np.mean(ranks)
                total_ranks.append(avg_rank)
                total_weights.append(len(label))  # Weight based on the number of labels

        # Compute MRR, Hit@K metrics
        if total_ranks and total_weights:
            weighted_mrr = np.sum(np.array(total_ranks) * np.array(total_weights)) / np.sum(total_weights)
            metrics[f"head_{node_types_i}_tail_{tail_type}_MRR"].append(weighted_mrr)

        if hit_at_1:
            metrics[f"head_{node_types_i}_tail_{tail_type}_Hit@1"].append(np.mean(hit_at_1))
        if hit_at_3:
            metrics[f"head_{node_types_i}_tail_{tail_type}_Hit@3"].append(np.mean(hit_at_3))
        if hit_at_10:
            metrics[f"head_{node_types_i}_tail_{tail_type}_Hit@10"].append(np.mean(hit_at_10))


    # Return the metrics
    new_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return new_metrics


# write the data loading module here
def main(
    data_dir="./data/BindData", # the data directory
    split_dir="./data/BindData/train_test_split", # the train/test split directory
    hidden_dim=384, # the hidden dimension of the transformation model
    n_layer=4, # the number of transformer layers
    batch_size=1024, # the training batch size
    gnn_num_layers=2,  # the number of  GNN Layers
    num_hops=2,  # the number of edge_node hops
    learning_rate=1.6e-3, # the learning ratesss
    n_epoch=1, # the number of training epochs
    weight_decay=1e-4, # the weight decay
    eval_steps=861, # the number of steps to evaluate the model
    save_dir="./checkpoints/bind", # the directory to save the model
    dataloader_num_workers=4, # the number of workers for data loading
    use_wandb=False, # whether to use wandb
    target_relation = 2, # the target relation
    target_node_type_index = 1, # the target node type index
    seed=42,
    ):
    # load embedding
    with open(os.path.join(data_dir, "embedding_dict.pkl"), "rb") as f:
        embedding_dict = pickle.load(f)
        
    
    # load data config
    with open(os.path.join(data_dir, "data_config.json"), "r") as f:
        data_config = json.load(f)

    # load train/test split
    split_data = load_split_data(split_dir)

    # build dataset
    train_data = TrainDataset(**{"triplet":split_data["train"], "node":split_data["node_train"]})
    val_data = ValDataset(**{"triplet_all":split_data["all"], 
                               "node_test":split_data["node_test"],
                               "node_all":split_data["node_all"],
                               "target_relation":target_relation, # only consider the evaluation on one relation, 2: `interact with`
                               "target_node_type_index": target_node_type_index, # the index of the target node type: protein/gene is 1
                               "frequent_threshold": 50, # the threshold of the frequent node
                               })
    
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    # bulid torchdrug.data.Graph  
    train_graph = bulid_graph_train(split_data)
    val_graph = bulid_graph_val(split_data)
    # build the model
    print("### Model Configuration ###")
    # build model config
    model_config = build_model_config(data_config)
    model_config["hidden_dim"] = hidden_dim
    model_config["n_layer"] = n_layer
    model_config["gnn_num_layers"] = gnn_num_layers
    model_config["num_hops"] = num_hops
    print(json.dumps(model_config, indent=4))
    model = StruCro(**model_config)
    # load model from checkpoint_dir
    model.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model_gnn2_h384_b1024_lr0.0016_20250213_200812.bin")))
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save model config to the save directory
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)

    # debug
    # debug_train_dataloader(model, train_data, train_collate_fn)
    # debug_test_dataloader(model, val_data, ValCollator(embedding_dict))
    

        

# Try to create TrainingArguments with only the basic parameters first
    # build trainer
    eval_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=n_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # every node corresponds to multiple tail nodes
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        max_grad_norm=1.0, # gradient clipping
        warmup_ratio=0.1,
        dataloader_num_workers=dataloader_num_workers, # number of processes to use for dataloading
        report_to="wandb" if use_wandb else "none",
        )
    


    print("### Training Arguments ###")
    #print(train_args)
    print(json.dumps(eval_args.to_dict(), indent=4))

    print("### Number of Trainable Parameters ###")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # build trainer
    trainer = BindingTrainer(
        model=model,
        args=eval_args,
        train_dataset=None,
        eval_dataset=val_data,
        data_collator=TrainCollator(train_graph, embedding_dict),
        test_data_collator=ValCollator(val_graph, embedding_dict),
        compute_metrics=compute_metrics,
        )
    
    evaluation_results = trainer.evaluate()
    print("### Evaluation Results ###")
    print(evaluation_results)

    print("### Model Evaluation Done ###")

if __name__ == "__main__":
    fire.Fire(main)
