"""Implement the dataset class for training and evaluation of the binding model.
"""
import pdb
import os
import torch
import pandas as pd

def load_split_data(split_dir):
    train_split = pd.read_csv(os.path.join(split_dir, "triplet_train_negative.csv"))

    # uncomment if you want to convert the string negative_y_index_list to list at the beginning
    # print("Converting string negative_y_index_list to list...")
    # train_split["negative_y_index"] = train_split["negative_y_index"].map(eval)
    
    test_split = pd.read_csv(os.path.join(split_dir, "triplet_test.csv"))
    node_train_split = pd.read_csv(os.path.join(split_dir, "node_train.csv"))
    node_test_split = pd.read_csv(os.path.join(split_dir, "node_test.csv"))

    df_all = pd.read_csv(os.path.join(split_dir, "triplet_full.csv"))
    df_node_all = pd.concat([node_train_split, node_test_split], axis=0).reset_index(drop=True)

    # drop duplicate nodes and triples
    train_split = train_split.drop_duplicates(subset=["x_index", "y_index", "display_relation"]).reset_index(drop=True)
    # test_split = test_split.drop_duplicates(subset=["x_index", "y_index", "display_relation"]).reset_index(drop=True)
    node_train_split = node_train_split.drop_duplicates(subset=["node_index"]).reset_index(drop=True)
    node_test_split = node_test_split.drop_duplicates(subset=["node_index"]).reset_index(drop=True)
    df_all = df_all.drop_duplicates(subset=["x_index", "y_index", "display_relation"]).reset_index(drop=True)
    df_node_all = df_node_all.drop_duplicates(subset=["node_index"]).reset_index(drop=True)
    return {
        "train": train_split,
        "test": test_split,
        "node_train": node_train_split,
        "node_test": node_test_split,
        "all": df_all,
        "node_all": df_node_all,
    }

# class ValDataset(torch.utils.data.Dataset):
#     """
#     Use the test set to evaluate the retrieval performance of the model.
#     This version supports multiple pairs of target_relation and target_node_type_index.
#     """

#     def __init__(self,
#                  node_test,
#                  triplet_all,
#                  node_all,
#                  target_relation=None,
#                  target_node_type_index=None,
#                  frequent_threshold=None,
#                  ):
#         self.frequent_threshold = frequent_threshold

#         # Check input consistency
#         if target_relation is not None and target_node_type_index is not None:
#             if len(target_relation) != len(target_node_type_index):
#                 raise ValueError("target_relations and target_node_type_indices must have the same length.")
#             self.target_pairs = list(zip(target_relation, target_node_type_index))
#         else:
#             self.target_pairs = None

#         # Prepare to collect filtered data
#         filtered_triplets = []
#         filtered_node_tests = []
#         filtered_node_alls = []

#         # If target_pairs are specified, filter for each pair
#         if self.target_pairs is not None:
#             for rel, node_type in self.target_pairs:
#                 triplet_subset = triplet_all[
#                     (triplet_all['display_relation'] == rel) &
#                     (triplet_all['x_type'] == node_type)
#                 ].reset_index(drop=True).copy()
                
#                 all_node_index = pd.concat([triplet_subset["x_index"], triplet_subset["y_index"]]).unique()
                
#                 node_all_subset = node_all[node_all["node_index"].isin(all_node_index)].reset_index(drop=True).copy()
#                 node_test_subset = node_test[node_test["node_index"].isin(all_node_index) &
#                                              (node_test["node_type"] == node_type)].reset_index(drop=True).copy()

#                 filtered_triplets.append(triplet_subset)
#                 filtered_node_tests.append(node_test_subset)
#                 filtered_node_alls.append(node_all_subset)

#             # Concatenate all filtered subsets
#             self.triplet = pd.concat(filtered_triplets).drop_duplicates().reset_index(drop=True)
#             self.node = pd.concat(filtered_node_tests).drop_duplicates().reset_index(drop=True)
#             self.node_all = pd.concat(filtered_node_alls).drop_duplicates().reset_index(drop=True)
#         else:
#             # If no target pairs are specified, use all data
#             self.triplet = triplet_all.reset_index(drop=True)
#             self.node = node_test.reset_index(drop=True)
#             self.node_all = node_all.reset_index(drop=True)

#         # Apply frequent threshold filtering
#         if self.frequent_threshold is not None:
#             val_counts = self.triplet["y_index"].value_counts()
#             frequent_node_index = val_counts[val_counts >= self.frequent_threshold].index
#             self.triplet = self.triplet[self.triplet["y_index"].isin(frequent_node_index)].reset_index(drop=True).copy()
#             all_node_index = pd.concat([self.node["node_index"], self.triplet["y_index"]]).unique()
#             self.node_all = self.node_all[self.node_all["node_index"].isin(all_node_index)].reset_index(drop=True).copy()

#         # Ensure test nodes have corresponding triplets
#         node_test_new = self.node[self.node["node_index"].isin(self.triplet["x_index"])].reset_index(drop=True).copy()
#         if len(node_test_new) != len(self.node):
#             print(f"Warning: {len(self.node) - len(node_test_new)} test nodes are removed because they do not have a tail node in the triplet.")
#             diff_index = self.node["node_index"][~self.node["node_index"].isin(node_test_new["node_index"])]
#             self.node_all = self.node_all[~self.node_all["node_index"].isin(diff_index)].reset_index(drop=True).copy()
#             self.node = node_test_new

#         self.tail_node_types = self.triplet["y_type"].unique()

#     def __getitem__(self, index):
#         # Get the positive y_index and all candidate y_index from the same type
#         row = self.node.iloc[index]
#         triplet = self.triplet[self.triplet["x_index"] == row["node_index"]]
#         outputs = {
#             "x_index": row["node_index"],
#             "x_type": row["node_type"],
#             "y_index": triplet["y_index"].tolist(),
#             "y_type": triplet["y_type"].tolist(),
#             "display_relation": triplet["display_relation"].tolist(),
#             "relation": triplet["relation"].tolist(),
#         }
#         return outputs

#     def __len__(self):
#         return len(self.node)

#     def get_all_node(self):
#         return self.node_all

#     def get_all_triplet(self):
#         return self.triplet
    

    
class TrainDataset(torch.utils.data.Dataset):
    """Use the train set for contrastive learning with InfoNCE loss.
    """
    def __init__(self, triplet, node):
        self.triplet = triplet
        self.node = node
        self.triplet_df = pd.DataFrame(triplet).reset_index(drop=True)

    def __getitem__(self, index):
        return self.triplet.iloc[index]

    def __len__(self):
        return len(self.triplet)
    
    def get_data(self):
        return self.triplet_df


class ValDataset(torch.utils.data.Dataset):
    """Use the test set to evaluate the retrieval performance of the model. The evaluation is performed in this way:
    1. get the raw embeddings of all nodes in `self.node_all`.
    2. for each relation type, transform head node embedding to tail node embedding using the transformation model.
    3. match the transformed embedding with the raw embedding of the target node.

    Args:
        node_test (pd.DataFrame): the test node dataframe
        triplet (pd.DataFrame): the **all** triplet dataframe
        node_all (pd.DataFrame): the **all** node dataframe. Need to encode them all when evaluating.
        target_node_type_index (int): the target node type to consider for evaluation and prediction.
            Defaults to None and use all node types.
        target_relation (int): the `display relation` type to consider for evaluation. 
        frequent_threshold (int, optional): the tail node that appears less than this threshold will be removed in the evaluation.
            Defaults to None and use all nodes.
    """
    def __init__(self,
                 node_test,
                 triplet_all,
                 node_all,
                 target_node_type_index=None,
                 target_relation=None,
                 frequent_threshold=None,
                 ):
        self.target_relation = target_relation
        self.frequent_threshold = frequent_threshold
        self.target_node_type_index = target_node_type_index
        if target_relation is not None:

            # filter the triplet and node dataframe by the relation
            # only maintain triplets with the relation in the relation list
            # only maintain the nodes that appear in the triplets
            # only maintain the test nodes that appear in the triplets
            triplet_all = triplet_all[triplet_all['display_relation'].isin([target_relation])].reset_index(drop=True).copy()
            all_node_index  = pd.concat([triplet_all["x_index"], triplet_all["y_index"]]).unique()
            node_all = node_all[node_all["node_index"].isin(all_node_index)].reset_index(drop=True).copy()
            node_test = node_test[node_test["node_index"].isin(all_node_index)].reset_index(drop=True).copy()

        if target_node_type_index is not None:
            # filter the triplet that has x_type equal to the target node type
            triplet_all = triplet_all[triplet_all["x_type"] == target_node_type_index].reset_index(drop=True).copy()
            # only choose the test nodes that are the target node type
            node_test = node_test[node_test["node_type"] == target_node_type_index].reset_index(drop=True).copy()
            # only choose node_all that are the head node in node_test and the tail node in triplet_all
            all_node_index = pd.concat([node_test["node_index"], triplet_all["y_index"]]).unique()
            node_all = node_all[node_all["node_index"].isin(all_node_index)].reset_index(drop=True).copy()

        # filter out the target node in the triplet that is not frequent enough
        if self.frequent_threshold is not None:
            val_counts = triplet_all["y_index"].value_counts()
            frequent_node_index = val_counts[val_counts >= self.frequent_threshold].index
            triplet_all = triplet_all[triplet_all["y_index"].isin(frequent_node_index)].reset_index(drop=True).copy()
            all_node_index = pd.concat([node_test["node_index"], triplet_all["y_index"]]).unique()
            node_all = node_all[node_all["node_index"].isin(all_node_index)].reset_index(drop=True).copy()

        # filter out the test node that does have a tail node in the triplet
        node_test_new = node_test[node_test["node_index"].isin(triplet_all["x_index"])].reset_index(drop=True).copy()
        if len(node_test_new) != len(node_test):
            print(f"Warning: {len(node_test) - len(node_test_new)} test nodes are removed because they do not have a tail node in the triplet.")
            # find the difference between the two dataframes
            diff_index = node_test["node_index"][~node_test["node_index"].isin(node_test_new["node_index"])]
            # filter out node all
            node_all = node_all[~node_all["node_index"].isin(diff_index)].reset_index(drop=True).copy()
            node_test = node_test_new

        # save the filtered dataframes
        self.triplet = triplet_all
        self.node = node_test
        self.node_all = node_all   
        self.tail_node_types = self.triplet["y_type"].unique()

    def __getitem__(self, index):
        # get the positive y_index and all candidate y_index from the same type
        row = self.node.iloc[index]
        triplet = self.triplet[self.triplet["x_index"] == row["node_index"]]
        return {
            "x_index": row["node_index"],
            "x_type": row["node_type"],
            "y_index": triplet["y_index"].tolist(),
            "y_type": triplet["y_type"].tolist(),
            "display_relation": triplet["display_relation"].tolist(),
            "relation": triplet["relation"].tolist(),
        }
    
    def __len__(self):
        return len(self.node)
    
    def get_all_node(self):
        return self.node_all
    
    def get_all_triplet(self):
        return self.triplet