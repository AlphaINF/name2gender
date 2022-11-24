import logging
import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def csv_loader(file_name):
    fp = open(file_name, "r", encoding='utf-8', errors='ignore')
    all = fp.read().split('\n')
    fp.close()
    output = []
    for line in all:
        line_element = str(line).split(',')
        # print(line_element)

        try:
            label = ['女', '男'].index(line_element[1])
            output.append((line_element[0], ['F', 'M'][label]))
        except:
            label = 0
    logging.info('[DataLoader]: finished loading, len = ', len(output))
    return output

def data_loader(file_name):
    logging.info('[DataLoader]: loading data, path =' + file_name)
    fp = open(file_name, "r", encoding='utf-8', errors='ignore')
    next(fp)
    all = fp.read().split('\n')
    fp.close()
    output = []
    for line in all:
        line_element = str(line).split('\t')
        # print(line_element)

        try:
            label = ['F', 'M'].index(line_element[3])
            output.append((line_element[2], line_element[3]))
        except:
            label = 0
    logging.info('[DataLoader]: finished loading, len = ', len(output))
    return output


class NameDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        data = self.data[item]
        name = data[0]
        label = ['F', 'M'].index(data[1])
        return name, label

    def __len__(self):
        return len(self.data)


def name_to_list(name):
    character_list = []
    for ch in name:
        try:
            val = ch.encode(encoding='ansi')
            character_list.append((val[0] - 128) << 8 | val[1])
        except:
            character_list.append(0)
    return character_list


def name_to_tensor(name):
    character_list = name_to_list(name)
    name_tensor = torch.zeros(1, len(character_list))
    for i, e in enumerate(character_list):
        name_tensor[0][i] = e
    name_tensor = name_tensor.to(torch.long)
    len_tensor = torch.zeros(1).to(torch.long)
    len_tensor[0]=len(name)
    return name_tensor, len_tensor


def value_to_tensor(names, labels):
    name_list = []
    for name in names:
        name_list.append(name_to_list(name))

    lengths = np.array([len(name) for name in name_list])
    max_len = max(lengths)
    name_tensors = torch.zeros(len(names), max_len)

    for i, idx in enumerate(name_list):
        for j, e in enumerate(idx):
            name_tensors[i][j] = e

    name_tensors = name_tensors.to(torch.long)
    lengths = torch.from_numpy(lengths).to(torch.long)
    return name_tensors, labels, lengths
