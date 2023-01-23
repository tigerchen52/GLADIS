"""
This is the main file for pre-training
"""

import os
import sys
import logging
import torch
import numpy as np
import argparse
from math import exp
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from transformers import BertTokenizer, BertForNextSentencePrediction
import utils
import evaluation
import transformers
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)

#hyper-parameters
parser = argparse.ArgumentParser(description='acrobert')
parser.add_argument('-pre_train_path', help='the file path for the pre-training corpus', type=str, default="../input/pre_train_sample.txt")
parser.add_argument('-dictionary_path', help='the file path for the acronym dictionary', type=str, default="../input/acronym_kb.json")
parser.add_argument('-eval_path', help='validation data path', type=str, default='input/dev.txt')
parser.add_argument('-batch_size', help='the number of samples in each mini-batch', type=int, default=32)
parser.add_argument('-epoch', help='the number of epochs to train the model', type=int, default=100)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('-learning_rate', help='learning rate for training', type=float, default=2e-5)
parser.add_argument('-lr_decay', help='the decay rate for the learning rate', type=float, default=0.95)
parser.add_argument('-margin', help='the margin value lambda in the triplet loss', type=float, default=0.2)
parser.add_argument('-hard_neg_numbers', help='the number of hard negatives in each mini-batch', type=int, default=2)
parser.add_argument('-loss_check_step', help='print loss values for how many steps', type=float, default=1)
parser.add_argument('-check_step', help='save model for how many steps ', type=float, default=10000)
parser.add_argument('-model_path', help='the file path for storing models', type=str, default="../output/model_{a}_epoch.pt")
parser.add_argument('-device', help='which device to use', type=str, default="cuda:0")
parser.add_argument('-mode', help='training or evaluation', type=str, default="eval")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

# load acronym dictionary
kb = utils.load_acronym_kb(args.dictionary_path)


class AcronymBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", device='cpu'):
        super().__init__()
        self.device = device
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, pos_x, masked_pos_x=None, neg_x=None, train=True):
        loss, scores = 0.0, 0.0
        if train:
            pos_samples = self.tokenizer(pos_x, padding=True, return_tensors='pt', truncation=True)["input_ids"]
            neg_x = self.tokenizer(neg_x, padding=True, return_tensors='pt', truncation=True)["input_ids"]

            pos_samples = pos_samples.to(self.device)
            neg_x = neg_x.to(self.device)

            pos_outputs = self.model(pos_samples).logits
            neg_outputs = self.model(neg_x).logits
            pos_scores = 1 - nn.Softmax(dim=1)(pos_outputs)[:, 0]
            neg_scores = 1 - nn.Softmax(dim=1)(neg_outputs)[:, 0]
            loss = triplet_loss(pos_scores, neg_scores, args.margin)

        else:
            samples = self.tokenizer(pos_x, padding=True, return_tensors='pt', truncation=True)["input_ids"]
            samples = samples.to(self.device)
            outputs = self.model(samples).logits
            scores = nn.Softmax(dim=1)(outputs)[:, 0]

        return loss if train else scores


def triplet_loss(pos_score, neg_score, margin=0.2):
    losses = torch.relu(pos_score - neg_score + margin)
    return losses.mean()


def softmax(elements):
    total = sum([exp(e) for e in elements])
    return exp(elements[0]) / total


def cal_score(model, tokenizer, long_forms, contexts, batch_size):
    ps = list()
    for index in range(0, len(long_forms), batch_size):
        batch_lf = long_forms[index:index + batch_size]
        batch_ctx = [contexts] * len(batch_lf)
        encoding = tokenizer(batch_lf, batch_ctx, return_tensors="pt", padding=True, truncation=True, max_length=400).to(args.device)
        outputs = model(**encoding)
        logits = outputs.logits.cpu().detach().numpy()
        p = [softmax(lg) for lg in logits]
        ps.extend(p)
    return ps


def predict(model, short_form, context, batch_size, acronym_kb=kb):
    ori_candidate = utils.get_candidate(acronym_kb, short_form)
    long_terms = [str.lower(can) for can in ori_candidate]
    scores = cal_score(model.model, model.tokenizer, long_terms, context, batch_size)
    max_index = np.argmax(scores)
    return ori_candidate[max_index]


def eval(model, path, batch_size, acronym_kb=kb, train=True):
    data = utils.load_data(path)
    true_labels, pred_labels = list(), list()
    for index, sample in enumerate(data):
        if index % 100 == 0:
            logger.info('processing {a} lines '.format(a=index))
        if train and index > 200: break
        short_term = sample['short_term']
        long_term = sample['long_term']
        context_tokens = ' '.join([str.lower(t) for t in sample['tokens']])
        pred = predict(model, short_term, context_tokens, batch_size, acronym_kb=acronym_kb)
        true_labels.append(long_term)
        pred_labels.append(pred)
    macro_f1 = evaluation.macro_f1(true_labels, pred_labels)
    acc = evaluation.accuracy(true_labels, pred_labels)
    return macro_f1, acc


def train():
    model = AcronymBERT(device=args.device)
    model.to(args.device)
    loader = utils.AcroBERTLoader(batch_size=args.batch_size, tokenizer=model.tokenizer, kb=kb, shuffle=args.shuffle, hard_num=args.hard_neg_numbers)

    train_loader = loader(data_path=args.pre_train_path)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(trainable_num)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    max_f1, max_epoch = 0.0, 0
    for e in range(args.epoch):
        epoch_loss = 0
        batch_num = 0

        for pos_samples, masked_pos_samples, neg_samples in train_loader:
            model.train()
            optimizer.zero_grad()

            if batch_num % args.loss_check_step == 0 and batch_num != 0:
                logger.info('sample = {b}, loss = {a}'.format(a=epoch_loss / batch_num, b=batch_num * args.batch_size))

            if batch_num % args.check_step == 0 and batch_num != 0:
                for g in optimizer.param_groups:
                    g['lr'] *= args.lr_decay
            if batch_num % args.check_step == 0 and batch_num != 0:
                temp_path = args.model_path.format(a=str(e + 1) + '_' + str(batch_num))
                torch.save(model.state_dict(), temp_path)
            loss = model(pos_samples, masked_pos_samples, neg_samples)

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_num += 1

        scheduler.step()
        temp_path = args.model_path.format(a=e + 1)
        logger.info('the pre-training finished, saving model, path = {a}'.format(a=temp_path))
        torch.save(model.state_dict(), temp_path)
    return max_f1, max_epoch


def run_eval(model_path, device):
    model = AcronymBERT(device=args.device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)

    file_list = [
        '../evaluation/test_set/uad_test.json',
        '../evaluation/test_set/sciad_test.json',
        '../evaluation/test_set/bio_umls_test.json'
    ]

    dict_list = [
        '../evaluation/dict/uad_dict.json',
        '../evaluation/dict/sciad_dict.json',
        '../evaluation/dict/bio_umls_dict.json'
    ]
    f1s, accs = list(), list()
    for index, file in enumerate(file_list):
        dict_path = dict_list[index]
        # load acronym dictionary
        new_kb = utils.load_acronym_kb(dict_path)
        macro_f1, acc = eval(model, path=file, batch_size=args.batch_size, acronym_kb=new_kb, train=False)
        f1s.append(macro_f1)
        accs.append(acc)
        logger.info(file)
        logger.info('finished, macro_f1 = {a}, acc = {b}'.format(a=macro_f1, b=acc))

    logger.info('F1: {a}, ACC: {b}'.format(a=f1s, b=accs))


if __name__ == '__main__':
    import os
    logger.info("running %s", " ".join(sys.argv))
    if args.mode == 'training':
        train()
    else:
        run_eval(model_path='../input/acrobert.pt', device=args.device)

