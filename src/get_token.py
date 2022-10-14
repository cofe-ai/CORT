# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import torch
from transformers import BertTokenizer,XLNetTokenizer,RobertaTokenizer


def get_token(batch_sentences, model_type):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./src/model/bert-base-uncased/')
    if model_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("./src/model/xlnet-base-cased/")
    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("./src/model/roberta-base/")
    batch_tokenized = tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, pad_to_max_length=True)     
    input_ids = torch.tensor(batch_tokenized['input_ids'])
    attention_mask = torch.tensor(batch_tokenized['attention_mask'])
    return input_ids,attention_mask


def get_double_token(batch_sentences, batch_sentences1, model_type):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./src/model/bert-base-uncased/')
    if model_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("./src/model/xlnet-base-cased/")
    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("./src/model/roberta-base/")
    batch_tokenized = tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, pad_to_max_length=True)      
    input_ids = torch.tensor(batch_tokenized['input_ids'])
    attention_mask = torch.tensor(batch_tokenized['attention_mask'])
    
    batch_tokenized1 = tokenizer.batch_encode_plus(batch_sentences1, add_special_tokens=True, pad_to_max_length=True)    
    input_ids1 = torch.tensor(batch_tokenized1['input_ids'])
    attention_mask1 = torch.tensor(batch_tokenized1['attention_mask'])
    return input_ids, attention_mask, input_ids1, attention_mask1


def get_token_prompt(batch_sentences, model_type):
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('./src/model/bert-base-uncased/')
    if model_type == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained("./src/model/xlnet-base-cased/")
    if model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained("./src/model/roberta-base/")
    batch_tokenized = tokenizer.batch_encode_plus(batch_sentences, add_special_tokens=True, pad_to_max_length=True)      
    input_ids = torch.tensor(batch_tokenized['input_ids'])
    attention_mask = torch.tensor(batch_tokenized['attention_mask'])
    return input_ids, attention_mask


def get_mask_hidden(input_ids, attention_mask, model_type):
    if model_type == 'bert':
        id_tmp = 103
    if model_type == 'xlnet':
        id_tmp = 6
    if model_type == 'roberta':
        id_tmp = 50264
    b = torch.nonzero(input_ids == id_tmp).squeeze()
    a = torch.ones([input_ids.shape[0], 1]).long()
    if b.shape == torch.Size([2]):
         index_mask1 = b
    else:
        index_mask1 = b.gather(1, a)
    return index_mask1
