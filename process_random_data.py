import os
import torch
import pandas as pd
from utils import get_encode, MultiDataset
from sklearn.model_selection import train_test_split
import random 
import numpy as np




def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_data(path, tasks):
    # Get smiles and labels for each task 
    task_smiles = []
    task_labels = []
    all_smiles  = []
    print('Load dataset')
    for task in tasks:
        path_task = path + "/refined_merged_{}.csv".format(task)
        data      = pd.read_csv(path_task)
        smiles    = data['SMILES'].tolist()
        label     = data['Label'].tolist()
        task_smiles.append(smiles)
        task_labels.append(label)
        all_smiles.extend(smiles)

    # labeling for all smiles 
    all_smiles  = list(set(all_smiles))
    list_labels = [[] for i in range(len(tasks))]
    for i in range(len(tasks)):
        for smiles in all_smiles:
            if smiles in task_smiles[i]:
                idx = task_smiles[i].index(smiles)
                list_labels[i].append(task_labels[i][idx])
            else:
                # smiles in not labeled in this task
                list_labels[i].append(6)
    # print("encode dataset")
    data_list = get_encode(all_smiles, list_labels)
    return data_list

def get_statistic(X_test):
    labels = []
    for data in X_test:
        labels.append(data.y[0])
    labels = np.array(labels).transpose()
    for i in range(9):
        print(len(np.where(labels[i]==0)[0]), len(np.where(labels[i]==1)[0]))


def split_data(path, data_list, seed=2):
    "Mối chất sẽ được lưu dưới dạng Data(x=[26, 39], edge_index=[2, 56], edge_attr=[0], y=[1, 9], smiles='COc1cccc2c(N=NC(=O)c3ccc(S(N)(=O)=O)cc3)c(O)[nH]c12)"
    "Chúng ta sẽ có dánh sách của 46k phần tử đã được mã hóa, sau đó mới đem đi chia danh sách thành 3 tập train, val, test"
    # Load processed dataset 
    X_, X_test = train_test_split(data_list, test_size=0.1, random_state=seed)
    X_train, X_val = train_test_split(X_, test_size=0.15, random_state=1)
    trn  = MultiDataset(root=path, data_list=X_train, dataset="train")
    val  = MultiDataset(root=path, data_list=X_val, dataset="val")


    print("Num_of_sample train_set={}, val_set={}, test_set={}".format(len(trn), len(val), len(test)))
    return trn, val, test

if __name__ == "__main__":
    tasks = ['BRE','CNS','COL','LEU','LNS','MEL','OVA','PRO','REN']
    path = './raw_data'
    data_list = process_data(path, tasks) 
    print(data_list[0])

    tran, val , test = split_data(path='./processed_data', data_list=data_list)
    