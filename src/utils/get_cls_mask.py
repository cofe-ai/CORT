import torch


def get(bert_output, index_mask, input_ids):
    bert_cls_hidden_state = bert_output[0][:, 0, :]
    bert_mask_hidden_state = bert_cls_hidden_state
    if index_mask.shape == torch.Size([2]):
        bert_mask_hidden_state = bert_output[0][:, index_mask[1], :]
    else:
        for i in range(input_ids.shape[0]):
            bert_mask_hidden_state[i] = bert_output[0][i, index_mask[i], :]
    return bert_cls_hidden_state, bert_mask_hidden_state
