import pdb
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass
from transformers.trainer_utils import (
    EvalPrediction,
)
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np
from transformers.utils import ModelOutput
from tqdm import tqdm
from .losses import InfoNCE, SupConLoss
from .encoder import CompGATv5 
from .utils import wavelet_transform, add_noise


@dataclass
class StruCroModelOutput(ModelOutput):
    """Class for the output of the StruCro model.
    """
    loss: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None
    tail_embeddings: Optional[torch.FloatTensor] = None


class StruCro(nn.Module):
    def __init__(self,
                 n_node: int,
                 n_relation: int, 
                 proj_dim: dict,
                 hidden_dim: int = 768,
                 n_layer: int = 6,
                 gnn_num_layers: int = 2,
                 encoder_drop: float = 0.2,
                 beta: float = 0.1,
                 num_hops: int = 2,
                 edge_sampling_ratio: float = 0.4,
                 hid_drop: float = 0.2) -> None:
        super().__init__()
        
        # Initialize model parameters
        self._init_parameters(gnn_num_layers, num_hops, edge_sampling_ratio, hidden_dim)
        
        # Initialize embeddings
        self._init_embeddings(n_node, n_relation, hidden_dim)
        
        # Initialize GNN layers
        self._init_gnn_layers(gnn_num_layers, hidden_dim, encoder_drop, beta, hid_drop)
        
        # Initialize projection layers
        self._init_proj_layers(proj_dim, hidden_dim)
        
        # Initialize transformer encoder
        self._init_transformer(hidden_dim, n_layer)

    def _init_parameters(self, gnn_num_layers, num_hops, edge_sampling_ratio, hidden_dim):
        self.gnn_num_layers = gnn_num_layers
        self.num_hops = num_hops
        self.edge_sampling_ratio = edge_sampling_ratio
        self.paired_loss_fn = InfoNCE(negative_mode="paired")
        self.unpaired_loss_fn = InfoNCE(negative_mode="unpaired")
        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))

    def _init_embeddings(self, n_node, n_relation, hidden_dim):
        self.node_type_embed = nn.Sequential(
            nn.Embedding(n_node, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.relation_type_embed = nn.Sequential(
            nn.Embedding(n_relation, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def _init_gnn_layers(self, gnn_num_layers, hidden_dim, encoder_drop, beta, hid_drop):
        self.layer = gnn_num_layers
        self.gnn_layers = nn.ModuleList([
            CompGATv5(hidden_dim, hidden_dim, num_heads=1, drop=encoder_drop, bias=True, beta=beta)
            for _ in range(gnn_num_layers)
        ])
        self.hid_drop = nn.Dropout(hid_drop)

    def _init_proj_layers(self, proj_dim, hidden_dim):
        self.proj_layer = nn.ModuleDict({
            str(node_type): nn.Linear(dim, hidden_dim, bias=False)
            for node_type, dim in proj_dim.items()
        })

    def _init_transformer(self, hidden_dim, n_layer):
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)

    def forward(self, graph, node_emb, node_type, edge_info, tail_type_ids, head_indices, 
                tail_indices, tail_emb=None, neg_indices=None, return_loss=False, **kwargs):
        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process edge information
        edge_data = self._process_edge_data(graph, edge_info, device)
        
        # Get node embeddings
        node_input_embs = self._groupby_and_project(node_emb, node_type)
        tail_type_emb = self.node_type_embed(tail_type_ids).unsqueeze(1)
        
        # Apply data augmentation
        aug_embs = self._apply_augmentations(node_input_embs)
        tail_emb = self._groupby_and_project(tail_emb, tail_type_ids)
        tail_embs = self._apply_augmentations(tail_emb)
        
        # Process through GNN
        outputs = [self._process_gnn(emb, head_indices, tail_indices, edge_data, edge_info) 
                  for emb in aug_embs]
        
        # Combine outputs
        x, _ = self._combine_outputs(outputs)
        
        # Calculate loss if required
        loss = self._calculate_loss(outputs, tail_embs, neg_indices) if return_loss else None

        return StruCroModelOutput(
            embeddings=x,
            loss=loss,
            tail_embeddings=tail_emb
        )
    
    def projection(self, node_emb, node_type_id) -> torch.Tensor:
        return self.proj_layer[str(node_type_id)](node_emb)
    
    @torch.no_grad()
    def encode(self,
               head_emb,
               head_type_id,
               rel_type_id,
               tail_type_id,
               batch_size=None,
               return_projected_head=False):
                # encode type ids
        tgt_device = head_emb.device
        head_type_id = torch.tensor([head_type_id]).to(tgt_device)
        head_type_emb = self.node_type_embed(head_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)
        rel_type_id = torch.tensor([rel_type_id]).to(tgt_device)
        rel_type_emb = self.relation_type_embed(rel_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)
        tail_type_id = torch.tensor([tail_type_id]).to(tgt_device)
        tail_type_emb = self.node_type_embed(tail_type_id).unsqueeze(1) # (batch_size, 1, hidden_dim)

        num_samples = head_emb.size(0)
        if batch_size is None:
            batch_size = num_samples

        outputs = []
        projected_inputs = []
        for i in tqdm(range(0, num_samples, batch_size), "encoding..."):
            # project head embeddings
            head_input_emb = self.projection(head_emb[i:i+batch_size], head_type_id.item())
            projected_inputs.append(head_input_emb.cpu().detach().numpy())
            head_input_emb = head_input_emb.unsqueeze(1)
            head_type_input_emb = head_type_emb.repeat(len(head_input_emb), 1, 1)
            rel_type_input_emb = rel_type_emb.repeat(len(head_input_emb), 1, 1)
            tail_type_input_emb = tail_type_emb.repeat(len(head_input_emb), 1, 1)
            input_embs = torch.cat([head_input_emb, head_type_input_emb, rel_type_input_emb, tail_type_input_emb], dim=1)
            output_embs = self._encode(input_embs)
            outputs.append(output_embs.cpu().detach().numpy())

        outputs = np.concatenate(outputs, axis=0)
        projected_inputs = np.concatenate(projected_inputs, axis=0)
        print({"tail_emb":outputs, "head_emb":projected_inputs})
        if return_projected_head:
            return {"tail_emb":outputs, "head_emb":projected_inputs}
        else:
            return outputs
    
    def _groupby_and_project(self, head_emb, head_type_ids):
        head_type_id_uniq = torch.unique(head_type_ids)
        sample_index_groupby_head_type = defaultdict(list)
        if isinstance(head_type_ids, torch.Tensor) and head_type_ids.dim() == 0:
                    head_type_ids = head_type_ids.unsqueeze(0)
        batch_indexes = torch.arange(head_type_ids.size(0)).to(head_type_ids.device)
        for i, head_type_id in enumerate(head_type_id_uniq):
            sample_index_groupby_head_type[head_type_id.item()] = batch_indexes[head_type_ids == head_type_id].tolist()
        # forward for each head type
        head_input_embs, sample_indexes = [], []
        for head_type_id in head_type_id_uniq:
            subsample_index = sample_index_groupby_head_type[head_type_id.item()]
            head_emb_subsample = torch.cat([torch.tensor(head_emb[i])[None] for i in subsample_index]).to(head_type_ids.device)
            # projection
            head_emb_subsample = self.projection(head_emb_subsample, head_type_id.item())
            head_input_embs.append(head_emb_subsample)
            sample_indexes.extend(subsample_index)

        # sort head_input_embs by sample index from 0 to batch_size
        head_input_embs = torch.cat(head_input_embs, dim=0)
        head_input_embs = head_input_embs[torch.argsort(torch.tensor(sample_indexes))]


        return head_input_embs

    def _process_edge_data(self, graph, edge_info, device):
        edge_index = graph.edge_list[:, :2].t().to(device)
        edge_type = graph.edge_list[:, 2].to(device)
        
        # Sample edges if needed
        if self.edge_sampling_ratio < 1.0:
            num_edges = edge_index.size(1)
            sample_size = int(num_edges * self.edge_sampling_ratio)
            if sample_size < num_edges:
                idx = torch.randperm(num_edges)[:sample_size].to(device)
                edge_index = edge_index[:, idx]
                edge_type = edge_type[idx]
                
        return {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'relation_type_emb': self.relation_type_embed(
                torch.arange(graph.num_relation).to(device)
            )
        }

    def _apply_augmentations(self, node_input_embs):
        return [
            wavelet_transform(node_input_embs),
            add_noise(node_input_embs, noise_level=0.01)
        ]

    def _process_gnn(self, node_embs, head_indices, tail_indices, edge_data, edge_info):
        """Process node embeddings through GNN layers."""
        curr_embs = node_embs
        rel_encs = [None, None]  # Placeholder for relation encodings
        
        for gnn_layer in self.gnn_layers:
            # Encode relations using edge_info
            rel_agu1, rel_inv_agu1, rel_selfloop = self._encode_relation(
                curr_embs, 
                head_indices, 
                tail_indices,
                edge_info
            )
            
            # Process through GNN layer
            head_emb, tail_emb, curr_embs, rel_agu1, rel_inv_agu1 = gnn_layer(
                curr_embs,
                edge_data['edge_index'],
                edge_data['relation_type_emb'],
                edge_data['edge_type'],
                head_indices,
                tail_indices,
                rel_agu1,
                rel_inv_agu1,
                rel_selfloop
            )
            rel_encs = [rel_agu1, rel_inv_agu1]
            
        return head_emb, tail_emb, curr_embs

    def _combine_outputs(self, outputs):
        head_embs = [out[0] for out in outputs]
        tail_embs = [out[1] for out in outputs]
        
        x = self.a * head_embs[0] + (1 - self.a) * head_embs[1]
        tail_emb = self.b * tail_embs[0] + (1 - self.b) * tail_embs[1]
        
        return x, tail_emb

    def _calculate_loss(self, outputs, tail_embs, neg_indices=None):
        """Calculate contrastive loss with two augmented views"""
        head_embs = [out[0] for out in outputs]
        #tail_embs = [out[1] for out in outputs]
        node_embs = [out[2] for out in outputs]
        
        if neg_indices is not None:
            # Handle negative samples
            neg_tail_indices = neg_indices[:, :, 1].reshape(-1)
            neg_tail_embs = [emb[neg_tail_indices].view(neg_indices.size(0), neg_indices.size(1), -1) 
                            for emb in node_embs]
            
            # Calculate loss with negative samples
            loss_w = (self.paired_loss_fn(head_embs[0], tail_embs[0], neg_tail_embs[0]) + 
                     self.paired_loss_fn(head_embs[0], tail_embs[0], neg_tail_embs[1]))
            loss_m = (self.paired_loss_fn(head_embs[1], tail_embs[1], neg_tail_embs[0]) + 
                     self.paired_loss_fn(head_embs[1], tail_embs[1], neg_tail_embs[1]))
        else:
            # Calculate loss without negative samples (e.g., during testing)
            loss_w = (self.unpaired_loss_fn(head_embs[0], tail_embs[0]) + 
                     self.unpaired_loss_fn(head_embs[1], tail_embs[1]))
            loss_m = (self.unpaired_loss_fn(head_embs[0], tail_embs[1]) + 
                     self.unpaired_loss_fn(head_embs[1], tail_embs[0]))
            
        return loss_w + loss_m

    def _encode_relation(self, node_embs, head_indices, tail_indices, edge_info):
        """Encode relations between nodes using type information from edge_info"""
        # Get type information from edge_info
        head_type_ids = edge_info[:, 0]  # head node type
        tail_type_ids = edge_info[:, 1]  # tail node type
        rel_type_ids = edge_info[:, 2]   # relation type
        
        # Get embeddings for types
        head_type_emb = self.node_type_embed(head_type_ids).unsqueeze(1)
        tail_type_emb = self.node_type_embed(tail_type_ids).unsqueeze(1)
        relation_emb = self.relation_type_embed(rel_type_ids).unsqueeze(1)
        
        # Get node embeddings
        head_node_emb = node_embs[head_indices]
        tail_node_emb = node_embs[tail_indices]
        
        # Create input embeddings for relation encoding
        head_input = torch.cat([
            head_node_emb.unsqueeze(1),
            head_type_emb,
            tail_type_emb,
            relation_emb
        ], dim=1)
        
        tail_input = torch.cat([
            tail_node_emb.unsqueeze(1),
            tail_type_emb,
            head_type_emb,
            relation_emb
        ], dim=1)
        
        # Encode relations
        rel_forward = self._encode(head_input)
        rel_backward = self._encode(tail_input)
        rel_self = self._encode(torch.cat([
            head_node_emb.unsqueeze(1),
            head_type_emb,
            head_type_emb,
            relation_emb
        ], dim=1))
        
        return rel_forward, rel_backward, rel_self

    def _encode(self, input_embs):
        """Encode input embeddings using transformer encoder"""
        output_embs = self.encoder(input_embs)
        rel_embs = output_embs[:, 0, :]
        # TransE-style addition
        output_embs = rel_embs + input_embs[:, 0, :]
        return rel_embs