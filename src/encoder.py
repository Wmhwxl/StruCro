import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

class CompGATv5(MessagePassing):
    """
    Compositional Graph Attention Layer with three types of attention: 
    forward, backward and self-loop.
    """
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        num_heads: int,
        drop: float,
        bias: bool = True,
        beta: float = 0.1
    ) -> None:
        super(CompGATv5, self).__init__(aggr='add')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.beta = beta
        
        # Forward attention layers
        self.w_q_forward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_k_forward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_v_forward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        # Backward attention layers
        self.w_q_backward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_k_backward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_v_backward = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        # Self-loop attention layers
        self.w_q_selfloop = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_k_selfloop = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        self.w_v_selfloop = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        # Regularization layers
        self.dropout = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Compute scaled dot-product attention."""
        # Compute attention scores
        attn_score = (q * k).sum(dim=-1) / math.sqrt(self.out_channels)
        return attn_score, v

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_index: torch.Tensor,
        relation_embedding: torch.Tensor,
        edge_type: torch.Tensor,
        head: torch.Tensor,
        tail: torch.Tensor,
        rel_agu1: torch.Tensor,
        rel_inv_agu1: torch.Tensor,
        rel_selfloop: torch.Tensor
    ) -> tuple:
        # Forward pass
        x_i = node_emb[edge_index[1]]  # Target nodes
        x_j = node_emb[edge_index[0]]  # Source nodes
        
        # Get relation embeddings
        rel_emb = torch.index_select(rel_inv_agu1, 0, edge_type)
        
        # Forward attention
        q_f = self.w_q_forward(x_i).view(-1, self.num_heads, self.out_channels)
        k_f = self.w_k_forward(x_j + rel_emb).view(-1, self.num_heads, self.out_channels)
        v_f = self.w_v_forward(x_j + rel_emb).view(-1, self.num_heads, self.out_channels)
        attn_score_f, _ = self._compute_attention(q_f, k_f, v_f)
        
        # Backward attention
        edge_index_backward = edge_index.flip(0)
        rel_emb_backward = torch.index_select(rel_inv_agu1, 0, edge_type)
        x_i_b = node_emb[edge_index_backward[0]]
        x_j_b = node_emb[edge_index_backward[1]]
        
        q_b = self.w_q_backward(x_i_b).view(-1, self.num_heads, self.out_channels)
        k_b = self.w_k_backward(x_j_b + rel_emb_backward).view(-1, self.num_heads, self.out_channels)
        v_b = self.w_v_backward(x_j_b + rel_emb_backward).view(-1, self.num_heads, self.out_channels)
        attn_score_b, _ = self._compute_attention(q_b, k_b, v_b)
        
        # Self-loop attention
        rel_emb_selfloop = torch.index_select(rel_selfloop, 0, edge_type)
        x_i_s = node_emb[0]
        x_j_s = node_emb[0]
        
        q_s = self.w_q_selfloop(x_i_s).view(-1, self.num_heads, self.out_channels)
        k_s = self.w_k_selfloop(x_j_s + rel_emb_selfloop).view(-1, self.num_heads, self.out_channels)
        v_s = self.w_v_selfloop(x_j_s + rel_emb_selfloop).view(-1, self.num_heads, self.out_channels)
        attn_score_s, _ = self._compute_attention(q_s, k_s, v_s)
        
        # Combine attention scores
        attn_score = torch.cat([attn_score_f, attn_score_b, attn_score_s], dim=0)
        attention_weights = F.softmax(attn_score, dim=0)
        
        # Split attention weights
        f_size = attn_score_f.size(0)
        b_size = attn_score_b.size(0)
        
        attn_score_f_slice = attention_weights[:f_size]
        attn_score_b_slice = attention_weights[f_size:f_size + b_size]
        attn_score_s_slice = attention_weights[f_size + b_size:]
        
        # Compute weighted outputs
        output_f = (attn_score_f_slice.unsqueeze(-1) * v_f).sum(dim=1)
        output_b = (attn_score_b_slice.unsqueeze(-1) * v_b).sum(dim=1)
        output_s = (attn_score_s_slice.unsqueeze(-1) * v_s).sum(dim=1)
        
        # Combine outputs
        output_nodes = output_f + output_b + output_s
        output_nodes = self.leaky_relu(output_nodes)
        
        # Get final embeddings
        head_embs = output_nodes[head]
        tail_embs = output_nodes[tail]
        
        return head_embs, tail_embs, output_nodes, rel_agu1, rel_inv_agu1