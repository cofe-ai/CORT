import torch


def get_rep_cls_and_mask(bert_output, index_mask, input_ids):
    bert_cls_hidden_state = bert_output[0][:, 0, :]
    if index_mask.shape == torch.Size([2]):
        bert_mask_hidden_state = bert_output[0][:, index_mask[1], :]
    else:
        tmp = torch.range(0, input_ids.shape[0] - 1, device=input_ids.device, dtype=torch.int64)
        bert_mask_hidden_state = bert_output[0][tmp, index_mask.squeeze(), :]
    return bert_cls_hidden_state, bert_mask_hidden_state
