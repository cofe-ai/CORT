# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import os
import sys
import shutil
import argparse


import numpy as np
from torch.optim import AdamW, SGD
from src.evaluation import eva_classifier
from src.get_token import get_token
from src.data_loader import get_data, Mydataset
from torch.utils.tensorboard import SummaryWriter
from src.model.bert import *
from src.model.Roberta import *
from src.model.XLNet import *
from torch.utils.data import DataLoader


sys.path.append(".")


def _init_(seed):
    init_seed = seed
    torch.manual_seed(init_seed)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--n_label', type=int, default=3, choices=[2, 3, 5])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--special_token', type=str, default='</s>')
parser.add_argument('--name_model', type=str, default='roberta')
parser.add_argument('--baseline', type=int, default=0)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--optim', type=str, default='AdmW', choices=['AdmW', 'SGD'])
FLAGS = parser.parse_args()

if torch.cuda.is_available() and FLAGS.use_gpu > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    _init_(FLAGS.seed)
    max_value = 0
    if FLAGS.name_model == 'bert':
        classifier_model = BertClassificationModel().to(device)
    elif FLAGS.name_model == 'roberta':
        classifier_model = RobertaClassificationModel().to(device)
    else:
        classifier_model = XLNetClassificationModel().to(device)
    train_inputs, train_targets = get_data(
        FLAGS.data_path + "Kessler_all_train.csv", FLAGS.special_token)
    valid_inputs, valid_targets = get_data(
        FLAGS.data_path + "Kessler_all_valid.csv", FLAGS.special_token)
    test_inputs, test_targets = get_data(
        FLAGS.data_path + "Kessler_all_test.csv", FLAGS.special_token)
    dirs = "%s/%s" % ("./summary/com", FLAGS.name_model)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        shutil.rmtree(dirs)
        os.makedirs(dirs)
    writer = SummaryWriter(dirs)
    train_set = Mydataset(train_inputs, train_targets)
    print_every_batch = 5
    if FLAGS.optim == 'AdmW':
        optimizer = AdamW(classifier_model.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer = SGD(classifier_model.parameters(), lr=FLAGS.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    train_loader = DataLoader(dataset=train_set, batch_size=FLAGS.batch_size, shuffle=True)
    batch_count = int(len(train_loader) / FLAGS.batch_size)
    dirss = "%s/%s" % ("./model_all/com", FLAGS.name_model)
    if not os.path.exists(dirss):
        os.makedirs(dirss)
    else:
        shutil.rmtree(dirss)
        os.makedirs(dirss)
    for epoch in range(FLAGS.epochs):
        print_avg_loss = 0
        for i, data in enumerate(train_loader):
            classifier_model.train()
            inputs, labels = data
            labels = torch.as_tensor(labels).to(device)
            input_ids, attention_mask = get_token(inputs, FLAGS.name_model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            optimizer.zero_grad()
            outputs = classifier_model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
            total_loss = total_loss + loss.item()
            if i % print_every_batch == (print_every_batch - 1):
                print("Batch: %d, Loss: %.4f" % ((i + 1), print_avg_loss / print_every_batch))
                print_avg_loss = 0
        writer.add_scalar('Loss/training loss', total_loss / batch_count, epoch)
        classifier_model.eval()
        total = len(valid_inputs)
        pred_test = []
        with torch.no_grad():
            for i in range(total):
                input_ids, attention_mask = get_token([valid_inputs[i]], FLAGS.name_model)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = classifier_model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                pred_test.append(predicted.item())
        dict_weighted = eva_classifier(valid_targets, pred_test, average="weighted", labels=[0, 1, 2])
        print("valid")
        print('f1_weight:',dict_weighted['f1'])
        if dict_weighted['f1'] > max_value:
            torch.save(classifier_model, "%s/%s.pkl" % (dirss, epoch))
            max_value = dict_weighted['f1']

        writer.add_scalars("add_scalars/trigonometric", {'valid_acc': dict_weighted['acc'],
                                                         'valid_f1': dict_weighted['f1']}, epoch)
        total = len(test_inputs)
        pred_test = []
        with torch.no_grad():
            for i in range(total):
                input_ids, attention_mask = get_token([test_inputs[i]], FLAGS.name_model)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                outputs = classifier_model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                pred_test.append(predicted.item())
        dict_weighted = eva_classifier(test_targets, pred_test, average="weighted", labels=[0, 1, 2])
        dict_f = eva_classifier(test_targets, pred_test, labels=[0, 1, 2])
        print("test", ":", epoch)
        F1 = dict_f["f1"]
        Pre = dict_f["pre"]
        Rec = dict_f["rec"]
        print("Original Order")
        print('f1_weight:', dict_weighted['f1'])
        print('c_m:', dict_weighted['c_m'])
        print('acc: ' + str(dict_weighted['acc']) + " | " + 'f1_weight: ' + str(dict_weighted['f1']) + " | "
              + 'pre: ' + str(dict_weighted['pre']) + " | " + 'rec: ' + str(dict_weighted['rec']) + " | "
              + 'f1_micro: ' + str(dict_weighted['f1_micro']) + " | " + 'f1_macro: ' + str(dict_weighted['f1_macro']))
        print('Better-f1:' + str(F1[1]) + " | " + 'Better-pre:' + str(Pre[1]) + " | " + 'Better-rec:' + str(Rec[1])
              + " | " + 'Worse-f1:' + str(F1[2]) + " | " + 'Worse-pre:' + str(Pre[2]) + " | " + 'Worse-rec:'
              + str(Rec[2]) + " | " + 'Same-f1:' + str(F1[0]) + " | " + 'Worse-pre:' + str(
            Pre[0]) + " | " + 'Worse-rec:' + str(Rec[0]))
        total_loss = 0
    writer.close()
