# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import pandas as pd
import torch
from pandas.core.frame import DataFrame
from torch.utils.data import Dataset
from transformers import BertTokenizer


def list_pipei(txt, sp):
    p, start, end = 0, 0, 0
    for i in range(len(txt)):
        if txt[i] == sp[0]:
            ii = i
            p = 1
            for j in range(len(sp)):
                if txt[ii] != sp[j]:
                    p = 0
                    break
                else:
                    ii = ii+1
        if p == 1:
            start = i
            end = i+len(sp)
            break
    if p == 0:
        print(txt)
        print(sp)
        print("error")
    return start+1, end+1


def get_ids(train_set, t1_t):
    tokenizer = BertTokenizer.from_pretrained('./src/model/bert-base-uncased/' )
    t1_pos_test = []
    for i in range(len(train_set)):
        sentence_token = tokenizer.tokenize(train_set[i])
        t1 = tokenizer.tokenize(t1_t[i])
        if t1 == []:
            t1_pos_test.append([0, 1])
        else:
            start, end = list_pipei(sentence_token, t1)
            t1_pos_test.append([start, end])
    return torch.tensor(t1_pos_test, dtype=torch.long)


def get_coqe(path):
    data1 = pd.read_csv(path, encoding='utf-8')
    sentence = []
    label = []
    t1 = []
    t2 = []
    aspects = []
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
        else:
            label.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = ""
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = ""
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = ""
        t1.append(data1["object_a"][i])
        t2.append(data1["object_b"][i])
        aspects.append(data1["aspect"][i])
        sentence.append(data1["sentence"][i])
    dict_d = {0: sentence, 1: label, 2: t1, 3: t2, 4: aspects}
    
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    t1 = train_set[2].values
    t2 = train_set[3].values
    aspects = train_set[4].values
    train_inputs = sentences
    train_targets = targets
    t1_pos = get_ids(sentences, t1)
    t2_pos = get_ids(sentences, t2)
    aspect_pos = get_ids(sentences, aspects)
    return train_inputs, train_targets, t1_pos, t2_pos, aspect_pos


def get_data(path, special_token):
    data1 = pd.read_csv(path, encoding='utf-8')
    sentence = []
    label = []
    for i in range(len(data1["id"])):
        if(data1["most_frequent_label"][i] == "better"):
            label.append(1)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
        else:
            label.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = " "
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = " "
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"
        sentence.append(data1["sentence"][i] + special_token + data1["object_a"][i] + special_token + data1["object_b"][
            i] + special_token + data1["aspect"][i])
    dict_d = {0: sentence, 1: label}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    return train_inputs, train_targets


def get_data_prompt(path, special_token, MASK):
    data1 = pd.read_csv(path, encoding = 'utf-8')
    sentence = []
    label = []
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
        else:
            label.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = "others"
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = "others"
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + data1["aspect"][i])
    dict_d = {0: sentence, 1: label}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    return train_inputs, train_targets


def get_data_prompt_2(path, special_token, MASK):
    data1 = pd.read_csv(path, encoding='utf-8')
    sentence = []
    label = []
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "BETTER":
            label.append(1)
        elif data1["most_frequent_label"][i] == "WORSE":
            label.append(2)
        else:
            continue
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + " genral.")
    dict_d = {0: sentence, 1: label}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    return train_inputs, train_targets


def get_data_prompt4(path, special_token, MASK, rate):
    data1 = pd.read_csv(path, encoding='utf-8')
    row = len(data1["id"]) * rate
    data1 = pd.read_csv(path, nrows =int(row))
    sentence = []
    label = []

    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
        else:
            label.append(0)

        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = "others"
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = "others"
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + data1["aspect"][i])
    dict_d = {0: sentence, 1: label}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets

    return train_inputs, train_targets


def get_data1(path1, special_token):
    sentence = []
    label = []

    data1 = pd.read_csv(path1, encoding='utf-8')
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "BETTER":
            label.append(1)
        elif data1["most_frequent_label"][i] == "WORSE":
            label.append(2)
        else:
            continue

        sentence.append(data1["sentence"][i] + special_token + data1["object_a"][i] + special_token + data1["object_b"][
            i] + special_token + "general")
    dict_d = {0: sentence, 1: label}

    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    return train_inputs, train_targets


def get_data4(path, special_token, rate):
    data1 = pd.read_csv(path, encoding='utf-8')
    row = len(data1["id"]) * rate
    data1 = pd.read_csv(path, nrows=int(row))
    sentence = []
    label = []
    print(len(data1["id"]))
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
        else:
            label.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = " "
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = " "
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"

        sentence.append(data1["sentence"][i] + special_token + data1["object_a"][i] + special_token + data1["object_b"][
            i] + special_token + data1["aspect"][i])
    dict_d = {0: sentence, 1: label}
    
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    return train_inputs, train_targets


def get_data_double(path, special_token, MASK=""):
    data1 = pd.read_csv(path, encoding='utf-8')
    sentence = []
    label = []
    sentence1 = []
    label1 = []
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
            label1.append(2)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
            label1.append(1)
        else:
            label.append(0)
            label1.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = "others"
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = "others"
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + data1["aspect"][i])
        sentence1.append(
            data1["sentence"][i] + special_token + data1["object_b"][i] + " is " + MASK + " than " + data1["object_a"][
                i] + " in " + data1["aspect"][i])
    dict_d = {0: sentence, 1: label, 2: sentence1, 3: label1}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    sentences1 = train_set[2].values
    targets1 = train_set[3].values
    train_inputs1 = sentences1
    train_targets1 = targets1
    return train_inputs, train_targets, train_inputs1, train_targets1


def get_data_double1(path1, special_token, MASK=""):
    sentence = []
    label = []
    sentence1 = []
    label1 = []
   
    data1 = pd.read_csv(path1, encoding = 'utf-8')
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "BETTER":
            label.append(1)
            label1.append(2)
        elif data1["most_frequent_label"][i] == "WORSE":
            label.append(2)
            label1.append(1)
        else:
            continue
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + " general")
        sentence1.append(
            data1["sentence"][i] + special_token + data1["object_b"][i] + " is " + MASK + " than " + data1["object_a"][
                i] + " in " + " general")

    dict_d = {0: sentence, 1: label, 2: sentence1, 3: label1}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    sentences1 = train_set[2].values
    targets1 = train_set[3].values
    train_inputs1 = sentences1
    train_targets1 = targets1
    return train_inputs, train_targets, train_inputs1, train_targets1


def get_data_double4(path, special_token, MASK, rate):
    data1 = pd.read_csv(path, encoding='utf-8')
    row = len(data1["id"]) * rate
    data1 = pd.read_csv(path, nrows=int(row))
    sentence = []
    label = []
    sentence1 = []
    label1 = []
    print(len(data1["id"]))
    for i in range(len(data1["id"])):
        if data1["most_frequent_label"][i] == "better":
            label.append(1)
            label1.append(2)
        elif data1["most_frequent_label"][i] == "worse":
            label.append(2)
            label1.append(1)
        else:
            label.append(0)
            label1.append(0)
        if type(data1["object_a"][i]) == float:
            data1["object_a"][i] = "others"
        if type(data1["object_b"][i]) == float:
            data1["object_b"][i] = "others"
        if type(data1["aspect"][i]) == float:
            data1["aspect"][i] = "general"
        sentence.append(
            data1["sentence"][i] + special_token + data1["object_a"][i] + " is " + MASK + " than " + data1["object_b"][
                i] + " in " + data1["aspect"][i])
        sentence1.append(
            data1["sentence"][i] + special_token + data1["object_b"][i] + " is " + MASK + " than " + data1["object_a"][
                i] + " in " + data1["aspect"][i])

    dict_d = {0: sentence, 1: label, 2: sentence1, 3: label1}
    train_set = DataFrame(dict_d)
    sentences = train_set[0].values
    targets = train_set[1].values
    train_inputs = sentences
    train_targets = targets
    sentences1 = train_set[2].values
    targets1 = train_set[3].values
    train_inputs1 = sentences1
    train_targets1 = targets1
    return train_inputs, train_targets, train_inputs1, train_targets1


def get_batch_data(train_inputs, train_targets):
    batch_train_inputs, batch_train_targets = [], []
    for i in range(batch_count):
        batch_train_inputs.append(train_inputs[i * batch_size: (i + 1) * batch_size])
        batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])
    return batch_train_inputs, batch_train_targets


def get_double_batch_data(train_inputs, train_targets, train_inputs1, train_targets1):
    batch_train_inputs, batch_train_targets,batch_train_inputs1, batch_train_targets1 = [], [], [], []
    for i in range(batch_count):
        batch_train_inputs.append(train_inputs[i * batch_size: (i + 1) * batch_size])
        batch_train_targets.append(train_targets[i * batch_size: (i + 1) * batch_size])
        batch_train_inputs1.append(train_inputs1[i * batch_size: (i + 1) * batch_size])
        batch_train_targets1.append(train_targets1[i * batch_size: (i + 1) * batch_size])
    return batch_train_inputs, batch_train_targets,batch_train_inputs1, batch_train_targets1


class Mydataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.idx = list()
        for item in x:
            self.idx.append(item)
        pass

    def __getitem__(self, index):
        input_data = self.idx[index]
        target = self.y[index]
        return input_data, target

    def __len__(self):
        return len(self.idx)


class Mydataset_coqe(Dataset):
    def __init__(self, x,  y,  t1, t2, a):
        self.x = x
        self.y = y
        self.t1 = t1
        self.t2 = t2
        self.a = a
        self.idx = list()
        for i in range(len(x)):
            self.idx.append([x[i], t1[i], t2[i], a[i]])
        pass

    def __getitem__(self, index):
        input_data, t1, t2, a = self.idx[index] 
        target = self.y[index]
        return input_data, target, t1, t2, a

    def __len__(self):
        return len(self.idx)


class Mydataset_double(Dataset):
    def __init__(self, x, y, x1, y1):
        self.x = x
        self.y = y
        self.x1 = x1
        self.y1 = y1
        self.idx = list()
        for it in range(len(x)):
            self.idx.append([x[it], x1[it]])
        pass

    def __getitem__(self, index):
        input_data, input_data1 = self.idx[index]
        target = self.y[index]
        target1 = self.y1[index]
        return input_data, target, input_data1, target1

    def __len__(self):
        return len(self.idx)
