"""Train the StruCro model with InfoNCE loss"""
import os
import pdb
from typing import Dict, List, Optional
from collections import defaultdict
import fire
import pickle
import json
import time
import math
from src.utils import bulid_graph_eval_train_only, bulid_graph_train
from torch_geometric.data import Data
# solve the error "too many open files" when data_num_workers > 0
# ref: https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import torch
import pandas as pd
import numpy as np
from transformers import TrainingArguments
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    EvalPrediction,
)
from src.utils import build_model_config, compute_metrics
from src.strucro import StruCro
from src.dataset import TrainDataset, ValDataset
from src.collator import TrainCollator, ValCollator
from src.trainer import BindingTrainer
from src.dataset import load_split_data
from datetime import datetime
import psutil


def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# write the data loading module here
def main(
    data_dir="./data/BindData", # the data directory
    split_dir="./data/BindData/train_test_split", # the train/test split directory
    hidden_dim=64, # the hidden dimension of the transformation model
    n_layer=6, # the number of transformer layers
    batch_size=256, # the training batch size
    edge_sampling_ratio=0.35, # the ratio of sampling edges
    gnn_num_layers=3,  # the number of  GNN Layers
    num_hops=1,  # the number of edge_node hops
    learning_rate=1.6e-3, # the learning ratesss
    n_epoch=1, # the number of training epochs
    weight_decay=1e-4, # the weight decay
    eval_steps=1, # the number of steps to evaluate the model
    save_dir="./checkpoints/bind", # the directory to save the model
    dataloader_num_workers=4, # the number of workers for data loading
    use_wandb=False, # whether to use wandb
    seed=42,
    model_name="strucro",
    relation_modeling_mode="transformer",
    contrastive_mode="dual",
    augmentation_mode="wavelet_noise",
    noise_level=0.2,
    ):

    # 设置随机种子
    start_time = time.time()
    set_seed(seed)
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
    val_data = ValDataset(
            node_test=split_data["node_test"],
            triplet_all=split_data["all"],
            node_all=split_data["node_all"],
            target_relation=2,
            target_node_type_index=1,
            frequent_threshold=50)


    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    
 
    train_graph = bulid_graph_train(split_data)
    eval_graph = bulid_graph_eval_train_only(split_data)

    # build the model
    print("### Model Configuration ###")
    # build model config
    model_config = build_model_config(data_config)
    model_config["edge_sampling_ratio"] = edge_sampling_ratio
    model_config["hidden_dim"] = hidden_dim
    model_config["n_layer"] = n_layer
    model_config["gnn_num_layers"] = gnn_num_layers
    model_config["num_hops"] = num_hops
    if model_name.lower() == "strucro":
        model_config["relation_modeling_mode"] = relation_modeling_mode
        model_config["contrastive_mode"] = contrastive_mode
        model_config["augmentation_mode"] = augmentation_mode
        model_config["noise_level"] = noise_level
    print(json.dumps(model_config, indent=4))
    if model_name.lower() == "strucro":
        model = StruCro(**model_config)
    elif model_name.lower() == "rgcn":
        model = RGCNModel(**model_config)
    elif model_name.lower() == "graphsage":
        model = GraphSAGEModel(**model_config)
    elif model_name.lower() == "gat":
        model = GATModel(**model_config)
    elif model_name.lower() == "compgcn":
        model = CompGCNModel(**model_config)
    elif model_name.lower() == "dgi":
        model = DGIModel(**model_config)
    elif model_name.lower() == "hgt":
        model = HGTModel(**model_config)
    elif model_name.lower() == "han":
        model = HANModel(**model_config)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # save model config to the save directory
    with open(os.path.join(save_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    

# Try to create TrainingArguments with only the basic parameters first


    # build trainer
    train_args = TrainingArguments(
        output_dir=save_dir,
        overwrite_output_dir=True,
        num_train_epochs=n_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, # every node corresponds to multiple tail nodes
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=10,
        save_steps=861,
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        max_grad_norm=1.0, # gradient clipping
        warmup_ratio=0.1,
        dataloader_num_workers=dataloader_num_workers, # number of processes to use for dataloading
        report_to="wandb" if use_wandb else "none",
        save_safetensors=False,
        )
    

    print("### Training Arguments ###")
    #print(train_args)
    print(json.dumps(train_args.to_dict(), indent=4))

    print("### Number of Trainable Parameters ###")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # 检查 val_data 中是否有头节点6
    head_6_data = [row for row in val_data if row["x_index"] == 6]
    print(f"Number of head node 6 entries in val_data: {len(head_6_data)}")

    # build trainer
    trainer = BindingTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=TrainCollator(train_graph, embedding_dict),  
        test_data_collator=ValCollator(eval_graph, embedding_dict),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    end_time = time.time()
    total_time = end_time - start_time

    process = psutil.Process(os.getpid())
    process_memory_gb = process.memory_info().rss / 1024**3
    cpu_percent = psutil.cpu_percent(interval=0.2)

    print("\n### Resource Usage Summary ###")
    print(f"Total time: {total_time:.2f} s ({total_time/60:.2f} min)")
    print(f"CPU usage: {cpu_percent:.2f}%")
    print(f"Process memory: {process_memory_gb:.2f} GB")

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
        current_allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        current_reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        peak_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device_id) / 1024**3

        print(f"GPU: {gpu_name}")
        print(f"GPU current allocated memory: {current_allocated:.2f} GB")
        print(f"GPU current reserved memory: {current_reserved:.2f} GB")
        print(f"GPU peak allocated memory: {peak_allocated:.2f} GB")
        print(f"GPU peak reserved memory: {peak_reserved:.2f} GB")
    else:
        print("GPU: not available")

    # Save the model
    trainer.save_model(save_dir)


    # Save the model
    trainer.save_model(save_dir)

    # Generate model save name with timestamp and key parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_info = f"{model_name}_gnn{gnn_num_layers}_h{hidden_dim}_b{batch_size}_lr{learning_rate}_epo{n_epoch}_edge{edge_sampling_ratio}"

    # Create the model save name
    save_name = f"pytorch_model_{params_info}_{timestamp}.bin"

    # Save the model's state_dict using torch.save()
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))

    print(f"### Model Saved as {save_name} ###")

if __name__ == "__main__":
    fire.Fire(main)
