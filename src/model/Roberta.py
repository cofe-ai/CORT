# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import transformers as tfs
import torch.nn as nn
import torch
from src.utils.get_cls_mask import get_rep_cls_and_mask


class RobertaClassificationModel(nn.Module):
    def __init__(self):
        super(RobertaClassificationModel, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        bert_cls_hidden = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(bert_cls_hidden)
        return linear_output


class prompt_roberta(nn.Module):
    def __init__(self):
        super(prompt_roberta, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dropout = nn.Dropout(p=0.1)
        self.dense = nn.Linear(1536, 3)

    def forward(self, input_ids, attention_mask, index_mask):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        return linear_output

class roberta(nn.Module):
    def __init__(self):
        super(roberta, self).__init__()   
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
    
        self.dense = nn.Linear(768, 3)  
        self.dense1 = nn.Linear(1536, 768)  
        self.dense2 = nn.Linear(768, 3)  
        self.dropout = nn.Dropout(p=0.1) 
        self.dropout2 = nn.Dropout(p=0.1) 
        
    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense1(state)
        linear_output = self.dense(linear_output)
        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1,bert_mask_hidden_state1),1)
        hidden = self.dropout(bert_state1)
        linear_output1 = self.dense1(hidden)
        linear_output1 = self.dense(linear_output1)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3,linear_output4
    
    
class robert_double_Model(nn.Module):
    def __init__(self):
        super(robert_double_Model, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dense = nn.Linear(768, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        state = self.dropout(bert_cls_hidden_state)
        linear_output = self.dense(state)

        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        hidden = self.dropout(bert_cls_hidden_state1)
        linear_output1 = self.dense(hidden)

        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4


class robert_double_mask_Model(nn.Module):
    def __init__(self):
        super(robert_double_mask_Model, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dense = nn.Linear(768, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        state = self.dropout(bert_mask_hidden_state)
        linear_output = self.dense(state)

        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        hidden = self.dropout(bert_mask_hidden_state1)
        linear_output1 = self.dense(hidden)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3,linear_output4


class robertaCoModel(nn.Module):
    def __init__(self):
        super(robertaCoModel, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dense = nn.Linear(1536, 3)
        self.dense1 = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dense3 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)

        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1)
        hidden = self.dropout(bert_state1)
        linear_output1 = self.dense(hidden)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4


class robertaCoRModel(nn.Module):
    def __init__(self):
        super(robertaCoRModel, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dense = nn.Linear(1536, 3)
        self.dense1 = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dense3 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)

        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1)
        hidden = self.dropout(bert_state1)
        linear_output1 = self.dense1(hidden)

        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense3(bert_mask_hidden_state1)
        return linear_output, linear_output1, linear_output3, linear_output4


class robertaComDModel(nn.Module):
    def __init__(self):
        super(robertaComDModel, self).__init__()
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
        self.dense = nn.Linear(1536, 3)
        self.dense1 = nn.Linear(1536, 3)
        self.dense2 = nn.Linear(768, 3)
        self.dense3 = nn.Linear(768, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.dropout3 = nn.Dropout(p=0.1)

    def forward(self,input_ids, attention_mask, input_ids1, attention_mask1, index_mask, index_mask1):
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        bert_state = torch.cat((bert_cls_hidden_state, bert_mask_hidden_state), 1)
        state = self.dropout(bert_state)
        linear_output = self.dense(state)
        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        bert_state1 = torch.cat((bert_cls_hidden_state1, bert_mask_hidden_state1), 1)
        hidden = self.dropout(bert_state1)
        linear_output1 = self.dense(hidden)
        return linear_output, linear_output1, bert_mask_hidden_state, bert_mask_hidden_state1


class robertaCom2Model(nn.Module):
    def __init__(self):
        super(robertaCom2Model, self).__init__()   
        self.Roberta = tfs.RobertaModel.from_pretrained("src/model/roberta-base/")
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
        bert_output = self.Roberta(input_ids, attention_mask=attention_mask)
        bert_cls_hidden_state, bert_mask_hidden_state = get_rep_cls_and_mask(bert_output, index_mask, input_ids)
        state = self.dropout(bert_mask_hidden_state)
        linear_output = self.dense(state)

        bert_output1 = self.Roberta(input_ids1, attention_mask=attention_mask1)
        bbert_cls_hidden_state1, bert_mask_hidden_state1 = get_rep_cls_and_mask(bert_output1, index_mask1, input_ids1)
        hidden = self.dropout1(bert_mask_hidden_state1)
        linear_output1 = self.dense(hidden)
        linear_output3 = self.dense2(bert_mask_hidden_state)
        linear_output4 = self.dense2(bert_mask_hidden_state1)  ###试试不加
        return linear_output, linear_output1, linear_output3, linear_output4
