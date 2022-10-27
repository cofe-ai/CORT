# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import transformers as tfs
import torch.nn as nn
import torch
from src.utils.get_cls_mask import get_rep_cls_and_mask


class XLNetClassificationModel(nn.Module):
    def __init__(self):
        super(XLNetClassificationModel, self).__init__()   
        model_class = tfs.XLNetModel   
        self.XLNet = model_class.from_pretrained("src/model/xlnet-base-cased/")
        self.dropout = nn.Dropout(p=0.1)   
        self.dense = nn.Linear(768, 3)  
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.XLNet(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, -1, :]
        bert_cls_hidden = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(bert_cls_hidden)
        return linear_output


class prompt_xlnet(nn.Module):
    def __init__(self):
        super(prompt_xlnet, self).__init__()
        self.XLNet = tfs.XLNetModel.from_pretrained('src/model/xlnet-base-cased/')
        self.dense = nn.Linear(1536, 3)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, index_mask):
        bert_output = self.XLNet(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        return linear_output


class XLNetCoModel(nn.Module):
    def __init__(self):
        super(XLNetCoModel, self).__init__()
        self.XLNet = tfs.XLNetModel.from_pretrained('src/model/xlnet-base-cased/')
        self.dense = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.XLNet(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_cls_hidden_state = bert_output[0][:, -1, :]
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        bert_output1 = self.XLNet(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_cls_hidden_state1 = bert_output1[0][:, -1, :]
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1)
        hidden = self.dropout1(bert_state1)
        linear_output1 = self.dense(hidden)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4


class XLNetComDModel(nn.Module):
    def __init__(self):
        super(XLNetComDModel, self).__init__()
        self.XLNet = tfs.XLNetModel.from_pretrained('src/model/xlnet-base-cased/')
        self.dense = nn.Linear(1536, 3)
        self.dense1 = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dense3 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.XLNet(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_cls_hidden_state = bert_output[0][:, -1, :]
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1).cuda()
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        bert_output1 = self.XLNet(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_cls_hidden_state1 = bert_output1[0][:, -1, :]
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1).cuda()
        hidden = self.dropout1(bert_state1)
        linear_output1 = self.dense(hidden)
        return linear_output, linear_output1, bert_mask_hidden_state, bert_mask_hidden_state1


class XLNetCom2Model(nn.Module):
    def __init__(self):
        super(XLNetCom2Model, self).__init__()
        self.XLNet = tfs.XLNetModel.from_pretrained('src/model/xlnet-base-cased/')

        self.dense = nn.Linear(1536, 3)
        self.dense1 = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dense3 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)
        self.represent = torch.randn(3, 768)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.XLNet(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_cls_hidden_state = bert_output[0][:, -1, :]
        state = self.dropout(bert_mask_hidden_state)
        linear_output = self.dense(state)

        bert_output1 = self.XLNet(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_cls_hidden_state1 = bert_output1[0][:, -1, :]
        hidden = self.dropout1(bert_mask_hidden_state1)
        linear_output1 = self.dense(hidden)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4
