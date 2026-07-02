import torch
import numpy as np
from collections import defaultdict

class TrainCollator:
    def __init__(self, graph, embedding_dict, neg_sample_size=5) -> None:
        """
        :param graph: The full train graph
        :param embedding_dict: Dictionary mapping node indices to their embeddings
        """
        self.graph = graph  # 完整图
        self.embedding_dict = embedding_dict  # 节点索引到嵌入的映射
        self.neg_sample_size = neg_sample_size

    def generate_relation_flipped_negative_samples(self, head_indices, tail_indices, num_neg):

        num_samples = len(head_indices)
        all_neg_samples = []

        for i in range(num_samples):
            head = head_indices[i]
            tail = tail_indices[i]

        # Generate random negative samples
            neg_samples = []
            for _ in range(num_neg):
                if torch.rand(1).item() > 0.5:  # Flip head or tail with 50% probability
                    neg_samples.append((torch.randint(0, num_samples, (1,)).item(), tail))
                else:
                    neg_samples.append((head, torch.randint(0, num_samples, (1,)).item()))

            all_neg_samples.append(neg_samples)

        # Convert to a tensor of shape [num_samples, num_neg, 2]
        neg_indices = torch.tensor(all_neg_samples, dtype=torch.long).view( num_samples , num_neg, 2)
        return neg_indices

    def __call__(self, batch):
        outputs = defaultdict(list)   
        # 获取 batch 中所有节点对和它们的索引
        node_indices, edge_info, node_types, head_indices, tail_indices = self.get_node_indices_and_types(batch)
    
        # 根据节点索引在全局图中提取子图
        # 在 __call__ 方法中使用
        # subgraph, subgraph_indices = self.get_subgraph_with_k_hop(node_indices, k=2)  # 2-hop 邻居
        subgraph, subgraph_indices = self.get_subgraph(node_indices)
        # 确保 subgraph_indices 和 node_indices 的长度一致，并且不超出 907 节点的限制
        assert len(subgraph_indices) <= subgraph.num_node, "子图节点索引数超过了最大节点数 907"
        # 创建全局索引到子图索引的映射
        global_to_local_map = {idx: i for i, idx in enumerate(subgraph_indices)}

        # 使用映射将 head_indices 和 tail_indices 从全局索引转换为子图索引
        head_indices = [global_to_local_map[idx] for idx in head_indices]
        tail_indices = [global_to_local_map[idx] for idx in tail_indices]
        # 确保 head_indices 和 tail_indices 长度正确
        assert len(head_indices) == len(tail_indices), "头节点和尾节点数量不匹配"
        # 将索引和类型转换为张量
        node_emb = self.get_node_embeddings(subgraph_indices)
        node_types = torch.tensor(node_types, dtype=torch.long)
        head_indices = torch.tensor(head_indices, dtype=torch.long)
        tail_indices = torch.tensor(tail_indices, dtype=torch.long)
        edge_info = torch.tensor(edge_info, dtype=torch.long)
        
        # 根据 batch 生成负样本
        neg_indices = self.generate_relation_flipped_negative_samples(head_indices, tail_indices, num_neg=self.neg_sample_size)
        neg_indices = torch.tensor(neg_indices, dtype=torch.long)
        for row in batch:
            x_index = row["x_index"]
            y_index = row["y_index"]

            if x_index not in self.embedding_dict:
                print("[WARN] node index {} not in embedding dict, skip it.".format(x_index))
                continue

            if y_index not in self.embedding_dict:
                print("[WARN] node index {} not in embedding dict, skip it.".format(y_index))
                continue

            head_emb = torch.tensor(self.embedding_dict[x_index])
            tail_emb = torch.tensor(self.embedding_dict[y_index])
            #tail_embeddings[x_index].append(tail_emb)


            # Append to outputs
            outputs["head_emb"].append(head_emb)
            outputs["head_type_ids"].append(row["x_type"])
            outputs["rel_type_ids"].append(row["display_relation"])
            outputs["tail_type_ids"].append(row["y_type"])

        # Combine tail embeddings for each unique x_index
        
            outputs["tail_emb"].append(tail_emb)

        outputs["head_type_ids"] = torch.tensor(outputs["head_type_ids"])
        outputs["rel_type_ids"] = torch.tensor(outputs["rel_type_ids"])
        outputs["tail_type_ids"] = torch.tensor(outputs["tail_type_ids"])


        # 将所有数据存入 outputs 中
        outputs["graph"] = subgraph
        outputs["node_emb"] = node_emb
        outputs["node_type"] = node_types  # 返回节点类型
        outputs["edge_info"] = edge_info  # 返回边的类型信息
        outputs["head_indices"] = head_indices  # 头节点的索引
        outputs["tail_indices"] = tail_indices  # 尾节点的索引
        outputs["neg_indices"] = neg_indices
        
        # 返回构造好的子图以及节点的嵌入和类型
        return dict(outputs)
    
    def get_subgraph(self, node_indices):
        """利用 torchdrug 的 API 从全局图中提取子图"""
        subgraph = self.graph.subgraph(node_indices)
        return subgraph, node_indices  # 返回子图和对应的节点索引

    def get_node_indices_and_types(self, batch):
        """从 batch 中提取 x_index 和 y_index 的节点集合，并获取节点类型"""
        node_indices = []  # 用于存储唯一的节点索引
        node_types = []  # 用于存储节点对应的类型
        edge_info = []  # 用于存储边的信息
        head_indices = []  # 用于存储每条边的头节点索引
        tail_indices = []  # 用于存储每条边的尾节点索引

        index_to_type = {}  # 用于确保节点索引与类型一致

        for row in batch:
            x_idx = row['x_index']
            y_idx = row['y_index']
            x_type = row['x_type']
            y_type = row['y_type']

            # 添加 x_idx 和 y_idx 到对应的头节点和尾节点列表
            head_indices.append(x_idx)
            tail_indices.append(y_idx)

            # 如果 x_idx 不在 node_indices 列表中，则添加
            if x_idx not in index_to_type:
                node_indices.append(x_idx)
                node_types.append(x_type)
                index_to_type[x_idx] = x_type

            # 如果 y_idx 不在 node_indices 列表中，则添加
            if y_idx not in index_to_type:
                node_indices.append(y_idx)
                node_types.append(y_type)
                index_to_type[y_idx] = y_type

            # 保留边信息 (x_index, y_index, display_relation)
            edge_info.append((x_type, y_type, row['display_relation']))

        return node_indices, edge_info, node_types, head_indices, tail_indices

    def get_subgraph_with_k_hop(self, node_indices, k=2):
        """利用 k-hop 邻居扩展子图
        
        Args:
            node_indices (list): 初始节点索引列表
            k (int): 跳数限制，即考虑节点的k跳邻居
            
        Returns:
            tuple: (子图, 扩展后的节点索引列表)
        """
        # 将初始节点索引转换为集合，用于快速查找
        all_nodes = set(node_indices)
        frontier = set(node_indices)
        
        # 迭代k次，每次向外扩展一跳
        for hop in range(k):
            next_frontier = set()
            # 对当前边界中的每个节点
            for node in frontier:
                # 获取节点的所有邻居
                neighbors = self.graph.neighbors(node)
                # 将新发现的邻居节点添加到下一轮的边界集合中
                next_frontier.update(n.item() for n in neighbors if n.item() not in all_nodes)
            # 更新总节点集合和当前边界
            all_nodes.update(next_frontier)
            frontier = next_frontier
            
            # 如果没有新的节点被添加，提前结束
            if not frontier:
                break
        
        # 将集合转换回列表并提取子图
        final_indices = list(all_nodes)
        subgraph = self.graph.subgraph(final_indices)
        
        return subgraph, final_indices

    def get_node_embeddings(self, node_indices):
        """根据节点索引获取对应的嵌入"""
        node_emb = [self.embedding_dict[node_idx] for node_idx in node_indices]       
        return node_emb
    



import torch
from collections import defaultdict

class ValCollator:
    def __init__(self, graph, embedding_dict) -> None:
        """
        :param graph: The full train graph
        :param embedding_dict: Dictionary mapping node indices to their embeddings
        """
        self.graph = graph
        self.embedding_dict = embedding_dict
        self.is_triplet = True

    def get_node_indices_and_types(self, batch):
        """从 batch 中提取 x_index 和 y_index 的节点集合，并获取节点类型"""
        node_indices = []
        node_types = []
        edge_info = []
        head_indices = []
        tail_indices = []

        index_to_type = {}

        for row in batch:
            x_idx = row['x_index']
            y_idx = row['y_index']
            x_type = row['x_type']
            y_type = row['y_type']
            display_relation = row['display_relation']

            if x_idx not in index_to_type:
                node_indices.append(x_idx)
                # 如果 x_type 是标量张量，将其转换为有维度的张量
                if isinstance(x_type, torch.Tensor) and x_type.dim() == 0:
                    x_type = x_type.unsqueeze(0)
                node_types.append(torch.tensor(x_type))
                index_to_type[x_idx] = x_type

            if isinstance(y_idx, (list, tuple)) and isinstance(y_type, (list, tuple)) and isinstance(display_relation, (list, tuple)):
                num_y = len(y_idx)
                head_indices.extend([x_idx] * num_y)

                for single_y_idx, single_y_type, single_display_relation in zip(y_idx, y_type, display_relation):
                    tail_indices.append(single_y_idx)

                    if single_y_idx not in index_to_type:
                        if isinstance(single_y_type, torch.Tensor) and single_y_type.dim() == 0:
                            single_y_type = single_y_type.unsqueeze(0)
                        node_indices.append(single_y_idx)
                        node_types.append(torch.tensor(single_y_type))
                        index_to_type[single_y_idx] = single_y_type

                    edge_info.append((x_type, single_y_type, single_display_relation))
            else:
                tail_indices.append(y_idx)
                head_indices.append(x_idx)
        return node_indices, edge_info, node_types, head_indices, tail_indices
    
    def get_subgraph(self, node_indices):
        """利用 torchdrug 的 API 从全局图中提取子图"""
        subgraph = self.graph.subgraph(node_indices)
        return subgraph, node_indices

    def get_node_embeddings(self, node_indices):
        """根据节点索引获取对应的嵌入"""
        node_emb = [torch.tensor(self.embedding_dict[node_idx]) for node_idx in node_indices]

        return node_emb

    def __call__(self, batch):
        """aggregate the data to a batch for model inference.
        """
        if self.is_triplet:
            return self._collate_triplet(batch)
        else:
            return self._collate_node(batch)

    def set_mode(self, is_triplet):
        """Set the mode of the collator.
        """
        self.is_triplet = is_triplet

    def _collate_triplet(self, batch):
        """Collate the triplet data.
        """
        outputs = defaultdict(list)
        node_indices, edge_info, node_types, head_indices, tail_indices = self.get_node_indices_and_types(batch)
 
        subgraph, subgraph_indices = self.get_subgraph(node_indices)
        assert len(subgraph_indices) <= subgraph.num_node, "子图节点索引数超过了最大节点数 907"
        global_to_local_map = {idx: i for i, idx in enumerate(subgraph_indices)}

        graph_head_indices = [global_to_local_map[idx] for idx in head_indices]
        graph_tail_indices = [global_to_local_map[idx] for idx in tail_indices]
        assert len(graph_head_indices) == len(graph_tail_indices), "头节点和尾节点数量不匹配"

        node_emb = self.get_node_embeddings(subgraph_indices)
        # 将节点类型列表中的标量张量转换为有维度的张量
        node_types = torch.tensor([t.squeeze() if isinstance(t, torch.Tensor) and t.dim() == 1 else t for t in node_types], dtype=torch.long)
        graph_head_indices = torch.tensor(graph_head_indices, dtype=torch.long)
        graph_tail_indices = torch.tensor(graph_tail_indices, dtype=torch.long)
        edge_info = torch.tensor(edge_info, dtype=torch.long)
        embedding_dict = self.embedding_dict
        head_type_ids = edge_info[:,0]
        tail_type_ids = edge_info[:,1]
        for row in batch:
            tail_emb = [torch.tensor(embedding_dict[i]) for i in row["y_index"]]
            num_tail = len(tail_emb)
            head_emb = [torch.tensor(embedding_dict[row["x_index"]]) for _ in range(num_tail)]
            outputs["head_emb"].append(head_emb)
            outputs["tail_emb"].append(tail_emb)

        outputs["graph"] = subgraph
        outputs["node_emb"] = node_emb
        outputs["node_type"] = node_types  # 返回节点类型
        #outputs["head_embnode_type"] = node_types
        outputs["edge_info"] = edge_info
        outputs["head_indices"] = graph_head_indices
        outputs["tail_indices"] = graph_tail_indices
        outputs["head_index"] = head_indices
        outputs["tail_index"] = tail_indices
        outputs["tail_type_ids"] = torch.tensor(tail_type_ids)
        outputs["head_type_ids"] = torch.tensor(head_type_ids)
        return dict(outputs)

    def _collate_node(self, batch):
        """Collate for the input nodes
        """
        outputs = defaultdict(list)
        embedding_dict = self.embedding_dict
        for row in batch:
            node_index = row["node_index"]
            node_emb = torch.tensor(embedding_dict[node_index])
            outputs["node_emb"].append(node_emb)
            # 如果节点类型是标量张量，将其转换为有维度的张量
            if isinstance(row["node_type"], torch.Tensor) and row["node_type"].dim() == 0:
                row["node_type"] = row["node_type"].unsqueeze(0)
            outputs["node_type_id"].append(row["node_type"])
            outputs["node_index"].append(node_index)

        return dict(outputs)

def has_length(inputs):
    return getattr(inputs, "__len__", None) is not None
