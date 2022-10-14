# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import sys
import argparse
import torch
import numpy as np
from src.get_token import get_token, get_double_token, get_mask_hidden
from src.data_loader import get_data_double, Mydataset_double, get_data
from src.evaluation import eva_classifier


sys.path.append(".")


def init(seed):
    init_seed = seed
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--num', type=str, default=89)
parser.add_argument('--n', type=str, default=1)
parser.add_argument('--name_model', type=str, default='roberta')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--special_token', type=str, default='</s>')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--Type', type=str, default='test')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--double', type=int, default=1)
parser.add_argument('--path', type=str, default='save_model/best/_M1.0_1.0_1.0AdamW1e-05/')
parser.add_argument('--mask', type=str, default='<mask>')
FLAGS = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device("cuda", FLAGS.device)
else:
    device = torch.device("cpu")
init(FLAGS.seed)

classifier_model = torch.load(str(FLAGS.path) + str(FLAGS.name_model) + '.pkl',
                              map_location=lambda storage, loc: storage.cuda(FLAGS.device))

classifier_model.eval()
if FLAGS.double == 0:
    test_inputs, test_targets = get_data(FLAGS.data_path 
                                         + "Kessler_all_test.csv", FLAGS.special_token, FLAGS.mask)
else:
    test_inputs, test_targets, test_inputs1, test_targets1 = get_data_double(
        FLAGS.data_path + "Kessler_all_test.csv", FLAGS.special_token, FLAGS.mask)
total = len(test_inputs)
hit = 0
pred_test = []
with torch.no_grad():
    for i in range(total):
        if FLAGS.double == 0:
            input_ids, attention_mask = get_token([test_inputs[i]], FLAGS.name_model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = classifier_model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
        else:
            input_ids, attention_mask, input_ids1, attention_mask1 = get_double_token(
                [test_inputs[i]], [test_inputs1[i]], FLAGS.name_model)
            index = get_mask_hidden(input_ids, attention_mask, FLAGS.name_model)
            index1 = get_mask_hidden(input_ids1, attention_mask1, FLAGS.name_model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            index = index.to(device)
            index1 = index1.to(device)
            outputs, outputs1, _, _ = classifier_model(input_ids, attention_mask, input_ids1, attention_mask1, index,
                                                       index1)
            m1, predicted1 = torch.max(outputs, 1)
            m2, predicted2 = torch.max(outputs1, 1)
            if m1.item() > m2.item():
                predicted = predicted1
            else:
                predicted = (3 - predicted2) % 3
        pred_test.append(predicted.item())

dict_weighted1 = eva_classifier(test_targets, pred_test, average="weighted", labels=[0, 1, 2])
dict_f = eva_classifier(test_targets, pred_test, labels=[0, 1, 2])
F1 = dict_f["f1"]
Pre = dict_f["pre"]
Rec = dict_f["rec"]
print("Original Order")
print('f1_weight:', dict_weighted1['f1'])
print('c_m:', dict_weighted1['c_m'])
print('acc: ' + str(dict_weighted1['acc']) + " | " + 'f1_weight: ' + str(dict_weighted1['f1']) + " | "
      + 'pre: ' + str(dict_weighted1['pre']) + " | " + 'rec: ' + str(dict_weighted1['rec']) + " | "
      + 'f1_micro: ' + str(dict_weighted1['f1_micro']) + " | " + 'f1_macro: ' + str(dict_weighted1['f1_macro']))
print('Better-f1:' + str(F1[1]) + " | " + 'Better-pre:' + str(Pre[1]) + " | " + 'Better-rec:' + str(Rec[1])
      + " | " + 'Worse-f1:' + str(F1[2]) + " | " + 'Worse-pre:' + str(Pre[2]) + " | " + 'Worse-rec:'
      + str(Rec[2]) + " | " + 'Same-f1:' + str(F1[0]) + " | " + 'Worse-pre:' + str(
    Pre[0]) + " | " + 'Worse-rec:' + str(Rec[0]))
