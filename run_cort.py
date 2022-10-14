# coding: utf-8
#
# Copyright 2022 Hengran Zhang
# Author: Hengran Zhang
#
import os
import shutil
import sys
import argparse

import numpy as np
from torch.optim import AdamW, SGD
from src.evaluation import eva_classifier
from src.data_loader import get_data_double, Mydataset_double
from src.get_token import get_double_token, get_mask_hidden
from torch.utils.tensorboard import SummaryWriter 
from src.model.bert import *
from src.model.Roberta import *
from src.model.XLNet import *
from torch.utils.data import DataLoader

sys.path.append(".")


def init(seed):
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
parser.add_argument('--mask', type=str, default='<mask>')
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--n_label', type=int, default=3, choices=[2, 3, 5])
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--early_stop', type=str, default='loss')
parser.add_argument('--data_path', type=str, default='data/')
parser.add_argument('--special_token', type=str, default='</s>')
parser.add_argument('--name_model', type=str, default='roberta')
parser.add_argument('--baseline', type=int, default=0)
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--choice', type=int, default=0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--M', type=str, default='M')
parser.add_argument('--t1', type=float, default=1.0)
parser.add_argument('--t2', type=float, default=1.0)
parser.add_argument('--t3', type=float, default=1.0)
parser.add_argument('--droup', type=float, default=0.1)
parser.add_argument('--optim', type=str, default='AdamW', choices=['AdamW', 'Adam','SGD'])
FLAGS = parser.parse_args()


if torch.cuda.is_available() and FLAGS.use_gpu > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    max_value = 0
    min_value = 100000
    init(FLAGS.seed)
    if FLAGS.name_model == 'bert':
        classifier_model = BertcomModel().to(device)
        plm_params = list(map(id, classifier_model.bert.parameters()))
        plm = classifier_model.bert.parameters()
    elif FLAGS.name_model == 'roberta':
        classifier_model = robertaCoModel().to(device)
        plm_params = list(map(id, classifier_model.Roberta.parameters()))
        plm = classifier_model.Roberta.parameters()
    else: 
        classifier_model = XLNetCoModel().to(device)
        plm_params = list(map(id, classifier_model.XLNet.parameters()))
        plm = classifier_model.XLNet.parameters()

    train_inputs, train_targets, train_inputs1, train_targets1 = get_data_double(
        FLAGS.data_path + "Kessler_all_train.csv", FLAGS.special_token, FLAGS.mask)
    valid_inputs, valid_targets, valid_inputs1, valid_targets1 = get_data_double(
        FLAGS.data_path + "Kessler_all_valid.csv", FLAGS.special_token, FLAGS.mask)

    base_params = filter(lambda p: id(p) not in plm_params, classifier_model.parameters())
    if FLAGS.optim == 'AdamW':
        optimizer = AdamW(classifier_model.parameters(), lr=FLAGS.learning_rate)
    else:
        optimizer=SGD(classifier_model.parameters(),lr=FLAGS.learning_rate, momentum=FLAGS.momentum)
    dirs = "%s/%s" % ("./summary/best/" + "_" + FLAGS.M + str(FLAGS.t1) + "_" + str(FLAGS.t2) + "_" + str(FLAGS.t3) +
                      str(FLAGS.optim) + str(FLAGS.learning_rate), FLAGS.name_model)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    else:
        shutil.rmtree(dirs) 
        os.makedirs(dirs) 
    writer = SummaryWriter(dirs)
    
    train_set = Mydataset_double(train_inputs, train_targets, train_inputs1, train_targets1)
    print_every_batch = 5
    
    criterion = nn.CrossEntropyLoss()
    hinge_loss = nn.HingeEmbeddingLoss(margin=1.0)
    total_loss = 0
    train_loader = DataLoader(dataset=train_set, batch_size=FLAGS.batch_size, shuffle=True)
    batch_count = int(len(train_loader) / FLAGS.batch_size)
    print(next(classifier_model.parameters()).device)
    dirss = "%s/%s" % ("./save_model/best/" + "_" + FLAGS.M + str(FLAGS.t1) + "_" + str(FLAGS.t2) + "_" + str(FLAGS.t3)
                       + str(FLAGS.optim) + str(FLAGS.learning_rate), FLAGS.name_model)
    if not os.path.exists(dirss):
        os.makedirs(dirss)
    else:
        shutil.rmtree(dirss) 
        os.makedirs(dirss) 
    for epoch in range(FLAGS.epochs):
        print_avg_loss = 0
        for i, data in enumerate(train_loader):
            classifier_model.train()
            inputs, labels, inputs1, labels1 = data
            input_ids, attention_mask, input_ids1, attention_mask1 = get_double_token(inputs, inputs1, FLAGS.name_model)
            index = get_mask_hidden(input_ids, attention_mask, FLAGS.name_model)
            index1 = get_mask_hidden(input_ids1, attention_mask1, FLAGS.name_model)
            input_ids = input_ids.to(device)
            index = index.to(device)
            index1 = index1.to(device)
            attention_mask = attention_mask.to(device)
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            labels = torch.as_tensor(labels).to(device)
            labels1 = torch.as_tensor(labels1).to(device)
            optimizer.zero_grad()
            outputs, outputs1, mask, mask1 = classifier_model(input_ids, attention_mask, input_ids1, attention_mask1,
                                                              index, index1)
            x = 1 - torch.cosine_similarity(mask, mask1)
            x = x.to(device)
            y = torch.zeros(x.shape)-1
            for j in range(len(inputs)):
                if labels[j] == 0:
                    y[j] = 1
            y = y.to(device)

            loss = FLAGS.t1 * criterion(outputs, labels) + (FLAGS.t2) * criterion(outputs1, labels1) + (
                FLAGS.t3) * hinge_loss(x, y)
            loss.backward()
            optimizer.step()
            print_avg_loss += loss.item()
            total_loss = total_loss+loss.item()
            if i % print_every_batch == (print_every_batch-1):
                print("Batch: %d, Loss: %.4f" % ((i+1), print_avg_loss/print_every_batch))
                print_avg_loss = 0
        writer.add_scalar('Loss/training loss', total_loss / batch_count, epoch)
        total = len(valid_inputs)
        hit = 0
        pred_test = []
        classifier_model.eval()
        with torch.no_grad():
            for i in range(total):
                input_ids, attention_mask, input_ids1, attention_mask1 = get_double_token([valid_inputs[i]], [valid_inputs1[i]], FLAGS.name_model)
                index = get_mask_hidden(input_ids, attention_mask, FLAGS.name_model)
                index1 = get_mask_hidden(input_ids1, attention_mask1, FLAGS.name_model)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                input_ids1 = input_ids1.to(device)
                attention_mask1 = attention_mask1.to(device)
                index = index.to(device)
                index1 = index1.to(device)
                outputs, outputs1, _, _ = classifier_model(input_ids,attention_mask, input_ids1, attention_mask1, index, index1)
                m1, predicted1 = torch.max(outputs, 1)
                m2, predicted2 = torch.max(outputs1, 1)
                if m1.item() > m2.item():
                    predicted = predicted1
                else:
                    predicted = (3-predicted2)%3
                pred_test.append(predicted.item())
                if predicted == valid_targets[i]:
                    hit += 1

        dict_weighted = eva_classifier(valid_targets, pred_test, average="weighted", labels=[0, 1, 2])
        print(epoch)
        print('f1_weight:',dict_weighted['f1'])
        if FLAGS.early_stop == 'loss':
            if total_loss < min_value:
                torch.save(classifier_model, "%s/%s.pkl" % (dirss, FLAGS.name_model))
                min_value = total_loss
        else:
            if dict_weighted['f1'] > max_value:
                torch.save(classifier_model, "%s/%s.pkl" % (dirss, FLAGS.name_model))
                max_value = dict_weighted['f1']
        print('c_m',dict_weighted['c_m'])
        writer.add_scalars("add_scalars/1", {'train_loss': total_loss / batch_count, 'valid_acc': dict_weighted['acc'],
                                             'valid_f1': dict_weighted['f1']}, epoch)
        writer.add_scalars("add_scalars/2",
                           {'valid_micro': dict_weighted['f1_micro'], 'valid_acc': dict_weighted['acc'],
                            'valid_f1': dict_weighted['f1']}, epoch)
        total_loss = 0
    classifier_model = torch.load("%s/%s.pkl" % (dirss,FLAGS.name_model))
    classifier_model.eval()
    test_inputs, test_targets, test_inputs1, test_targets1 = get_data_double(FLAGS.data_path + "Kessler_all_test.csv",
                                                                             FLAGS.special_token, FLAGS.mask)
    total = len(test_inputs)
    hit = 0
    pred_test = []
    with torch.no_grad():
        for i in range(total):
            input_ids, attention_mask, input_ids1, attention_mask1 = \
                get_double_token([test_inputs[i]], [test_inputs1[i]], FLAGS.name_model)
            index = get_mask_hidden(input_ids, attention_mask, FLAGS.name_model)
            index1 = get_mask_hidden(input_ids1, attention_mask1, FLAGS.name_model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            index = index.to(device)
            index1 = index1.to(device)
            outputs, outputs1, _, _ = classifier_model(
                input_ids, attention_mask, input_ids1, attention_mask1, index, index1)
            m1, predicted1 = torch.max(outputs, 1)
            m2, predicted2 = torch.max(outputs1, 1)
            if m1.item() > m2.item():
                predicted = predicted1
            else:
                predicted = (3-predicted2) % 3
            pred_test.append(predicted.item())
            if predicted == test_targets[i]:
                hit += 1

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
          + " | " + 'Worse-f1:' + str(F1[2])+ " | " + 'Worse-pre:' + str(Pre[2]) + " | " + 'Worse-rec:' 
          + str(Rec[2]) + " | " + 'Same-f1:' + str(F1[0]) + " | " + 'Worse-pre:' + str(Pre[0]) + " | " + 'Worse-rec:' + str(Rec[0]))

    test_inputs, test_targets, test_inputs1, test_targets1 = get_data_double(
        FLAGS.data_path + "Kessler_all_reverse_test.csv", FLAGS.special_token, FLAGS.mask)
    total = len(test_inputs)
    hit = 0
    pred_test = []
    with torch.no_grad():
        for i in range(total):
            input_ids, attention_mask, input_ids1, attention_mask1 = \
                get_double_token([test_inputs[i]], [test_inputs1[i]], FLAGS.name_model)
            index = get_mask_hidden(input_ids, attention_mask, FLAGS.name_model)
            index1 = get_mask_hidden(input_ids1, attention_mask1, FLAGS.name_model)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            input_ids1 = input_ids1.to(device)
            attention_mask1 = attention_mask1.to(device)
            index = index.to(device)
            index1 = index1.to(device)
            outputs, outputs1, _, _ = classifier_model(
                input_ids, attention_mask, input_ids1, attention_mask1, index, index1)
            m1, predicted1 = torch.max(outputs, 1)
            m2, predicted2 = torch.max(outputs1, 1)
            if m1.item() > m2.item():
                predicted = predicted1
            else:
                predicted = (3-predicted2) % 3
            pred_test.append(predicted.item())
            if predicted == test_targets[i]:
                hit += 1

    dict_weighted1 = eva_classifier(test_targets, pred_test, average="weighted", labels=[0, 1, 2])
    dict_f = eva_classifier(test_targets, pred_test, labels=[0, 1, 2])
    F1 = dict_f["f1"]
    Pre = dict_f["pre"]
    Rec = dict_f["rec"]
    print("Reverse Order")
    print('f1_weight:', dict_weighted1['f1'])
    print('c_m:', dict_weighted1['c_m'])
    print('acc: ' + str(dict_weighted1['acc'])+" | "+'f1_weight: '+str(dict_weighted1['f1'])+" | "
          +'pre: '+str(dict_weighted1['pre'])+" | "+'rec: '+str(dict_weighted1['rec'])+" | "+'f1_micro: '+str(dict_weighted1['f1_micro'])+" | "+'f1_macro: '+str(dict_weighted1['f1_macro']))
    print('Better-f1:'+str(F1[1])+" | "+'Better-pre:'+str(Pre[1])+" | "+'Better-rec:'+str(Rec[1])+" | "+'Worse-f1:'+str(F1[2])
          +" | "+'Worse-pre:'+str(Pre[2])+" | "+'Worse-rec:'+str(Rec[2])+" | "+'Same-f1:'+str(F1[0])+" | "+'Worse-pre:'+str(Pre[0])+" | "+'Worse-rec:'+str(Rec[0]))
