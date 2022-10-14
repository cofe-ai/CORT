# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import transformers as tfs
from transformers import BertModel,BertTokenizer
import torch.nn as nn
import torch
from src.utils.get_cls_mask import get

      
class BertClassificationModel(nn.Module):
    def __init__(self):
        super(BertClassificationModel, self).__init__()     
        self.bert = BertModel.from_pretrained('src/model/bert-base-uncased/' )
        self.dense = nn.Linear(768, 3) 
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        state = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(state)
        return linear_output


class prompt_bert(nn.Module):
    def __init__(self):
        super(prompt_bert, self).__init__()   
        model_class = tfs.BertModel      
        self.bert = model_class.from_pretrained('src/model/bert-base-uncased/')
        self.dense = nn.Linear(1536, 3)  
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, index_mask):
        bert_output = self.bert(input_ids, attention_mask = attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        return linear_output


class BertcomModel(nn.Module):
    def __init__(self):
        super(BertcomModel, self).__init__()   
        model_class = tfs.BertModel      
        self.bert = model_class.from_pretrained('src/model/bert-base-uncased/')
        self.dense = nn.Linear(1536, 3)  
        self.dense1 = nn.Linear(1536, 3)  
        self.dense2 = nn.Linear(768, 3)  
        self.dense3 = nn.Linear(768, 3)  
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)

        linear_output = self.dense(bert_state)

        bert_output1 = self.bert(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1),1)
        linear_output1 = self.dense(bert_state1)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4


class BertcomDModel(nn.Module):
    def __init__(self):
        super(BertcomDModel, self).__init__()   
        model_class = tfs.BertModel   
        self.bert = model_class.from_pretrained('src/model/bert-base-uncased/')
        self.dense = nn.Linear(1536, 3) 
        self.dense1 = nn.Linear(1536, 3)  
        self.dense2 = nn.Linear(768, 3)  
        self.dense3 = nn.Linear(768, 3)  
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)

        bert_output1 = self.bert(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1)
        state = self.dropout(bert_state1)
        linear_output1 = self.dense(state)

        return linear_output, linear_output1, bert_mask_hidden_state, bert_mask_hidden_state1


class Bertcom2Model(nn.Module):
    def __init__(self):
        super(Bertcom2Model, self).__init__()
        model_class = tfs.BertModel
        self.bert = model_class.from_pretrained('src/model/bert-base-uncased/')
        self.dense = nn.Linear(768, 3)
        self.dense2 = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get(bert_output, index_mask, input_ids)
        linear_output = self.dense(bert_mask_hidden_state)

        bert_output1 = self.bert(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get(bert_output1, index_mask1, input_ids1)
        linear_output1 = self.dense(bert_mask_hidden_state1)

        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4
