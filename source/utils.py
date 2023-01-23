"""
This file contains functions for loading various needed data
"""

import json
import torch
import random
import nltk
import logging
import os
from random import random as rand
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def load_acronym_kb(kb_path='acronym_kb.json'):
    f = open(kb_path, encoding='utf8')
    acronym_kb = json.load(f)
    for key, values in acronym_kb.items():
        values = [v for v, s in values]
        acronym_kb[key] = values
    logger.info('loaded acronym dictionary successfully, in total there are [{a}] acronyms'.format(a=len(acronym_kb)))
    return acronym_kb


def get_candidate(acronym_kb, short_term):
    return acronym_kb[short_term]

def load_data(path):
    data = list()
    for line in open(path, encoding='utf8'):
        row = json.loads(line)
        data.append(row)
    return data


def load_dataset(data_path):
    all_short_term, all_long_term, all_context = list(), list(), list()
    for line in open(data_path, encoding='utf8'):
        obj = json.loads(line)
        short_term, long_term, context = obj['short_term'], obj['long_term'], ' '.join(obj['tokens'])
        all_short_term.append(short_term)
        all_long_term.append(long_term)
        all_context.append(context)

    return {'short_term': all_short_term, 'long_term': all_long_term, 'context':all_context}


def load_pretrain(data_path):
    all_short_term, all_long_term, all_context = list(), list(), list()
    cnt = 0
    for line in open(data_path, encoding='utf8'):
        cnt += 1
        # row = line.strip().split('\t')
        # if len(row) != 3:continue
        if cnt>200:continue
        obj = json.loads(line)
        short_term, long_term, context = obj['short_term'], obj['long_term'], ' '.join(obj['tokens'])
        all_short_term.append(short_term)
        all_long_term.append(long_term)
        all_context.append(context)

    return {'short_term': all_short_term, 'long_term': all_long_term, 'context': all_context}


class TextData(Dataset):
    def __init__(self, data):
        self.all_short_term = data['short_term']
        self.all_long_term = data['long_term']
        self.all_context = data['context']

    def __len__(self):
        return len(self.all_short_term)

    def __getitem__(self, idx):
        return self.all_short_term[idx], self.all_long_term[idx], self.all_context[idx]


def random_negative(target, elements):
    flag, result = True, ''
    while flag:
        temp = random.choice(elements)
        if temp != target:
            result = temp
            flag = False
    return result


class SimpleLoader():
    def __init__(self, batch_size, tokenizer, kb, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.kb = kb

    def collate_fn(self, batch_data):
        pos_tag, neg_tag = 0, 1
        batch_short_term, batch_long_term, batch_context = list(zip(*batch_data))
        batch_short_term, batch_long_term, batch_context = list(batch_short_term), list(batch_long_term), list(batch_context)
        batch_negative, batch_label, batch_label_neg = list(), list(), list()
        for index in range(len(batch_short_term)):
            short_term, long_term, context = batch_short_term[index], batch_long_term[index], batch_context[index]
            batch_label.append(pos_tag)
            candidates = [v[0] for v in self.kb[short_term]]
            if len(candidates) == 1:
                batch_negative.append(long_term)
                batch_label_neg.append(pos_tag)
                continue

            negative = random_negative(long_term, candidates)
            batch_negative.append(negative)
            batch_label_neg.append(neg_tag)

        prompt = batch_context + batch_context
        long_terms = batch_long_term + batch_negative
        label = batch_label + batch_label_neg

        encoding = self.tokenizer(prompt, long_terms, return_tensors="pt", padding=True, truncation=True)
        label = torch.LongTensor(label)

        return encoding, label

    def __call__(self, data_path):
        dataset = load_dataset(data_path=data_path)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


def mask_subword(subword_sequences, prob=0.15, masked_prob=0.8, VOCAB_SIZE=30522):
    PAD, CLS, SEP, MASK, BLANK = 0, 101, 102, 103, -100
    masked_labels = list()
    for sentence in subword_sequences:
        labels = [BLANK for _ in range(len(sentence))]
        original = sentence[:]
        end = len(sentence)
        if PAD in sentence:
            end = sentence.index(PAD)
        for pos in range(end):
            if sentence[pos] in (CLS, SEP): continue
            if rand() > prob: continue
            if rand() < masked_prob:  # 80%
                sentence[pos] = MASK
            elif rand() < 0.5:  # 10%
                sentence[pos] = random.randint(0, VOCAB_SIZE-1)
            labels[pos] = original[pos]
        masked_labels.append(labels)
    return subword_sequences, masked_labels


class AcroBERTLoader():
    def __init__(self, batch_size, tokenizer, kb, shuffle=True, masked_prob=0.15, hard_num=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.masked_prob = masked_prob
        self.hard_num = hard_num
        self.kb = kb
        self.all_long_terms = list()
        for vs in self.kb.values():
            self.all_long_terms.extend(list(vs))

    def select_negative(self, target):
        selected, flag, max_time = None, True, 10
        if target in self.kb:
            long_term_candidates = self.kb[target]
            if len(long_term_candidates) == 1:
                long_term_candidates = self.all_long_terms
        else:
            long_term_candidates = self.all_long_terms
        attempt = 0
        while flag and attempt < max_time:
            attempt += 1
            selected = random.choice(long_term_candidates)
            if selected != target:
                flag = False
        if attempt == max_time:
            selected = random.choice(self.all_long_terms)
        return selected

    def collate_fn(self, batch_data):
        batch_short_term, batch_long_term, batch_context = list(zip(*batch_data))
        pos_samples, neg_samples, masked_pos_samples = list(),  list(), list()
        for _ in range(self.hard_num):
            temp_pos_samples = [batch_long_term[index] + ' [SEP] ' + batch_context[index] for index in range(len(batch_long_term))]
            neg_long_terms = [self.select_negative(st) for st in batch_short_term]
            temp_neg_samples = [neg_long_terms[index] + ' [SEP] ' + batch_context[index] for index in range(len(batch_long_term))]
            temp_masked_pos_samples = [batch_long_term[index] + ' [SEP] ' + batch_context[index] for index in range(len(batch_long_term))]

            pos_samples.extend(temp_pos_samples)
            neg_samples.extend(temp_neg_samples)
            masked_pos_samples.extend(temp_masked_pos_samples)
        return pos_samples,  masked_pos_samples,  neg_samples

    def __call__(self, data_path):
        dataset = load_pretrain(data_path=data_path)
        logger.info('loaded dataset, sample = {a}'.format(a=len(dataset['short_term'])))
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // (2 * self.hard_num), shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


