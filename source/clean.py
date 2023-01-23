import json
import nltk


def load(path):
    ab_acronyms = dict()
    for line in open(path, encoding='utf8'):
        obj = json.loads(line)
        short_term, long_term = obj['short_term'], obj['long_term']
        tokens = nltk.word_tokenize(long_term)
        initials = ''.join([t[0] for t in tokens])
        if str.lower(short_term) == str.lower(initials):continue
        if (short_term, long_term) not in ab_acronyms:
            ab_acronyms[(short_term, long_term)] = 0
        ab_acronyms[(short_term, long_term)] += 1
    sorted_pairs = sorted(ab_acronyms.items(), key=lambda e:e[1],reverse=True)
    for key, value in sorted_pairs:
        print(key, value)


def deduplicate(path, out_path):
    wl = open(out_path, 'w', encoding='utf8')
    visited = set()
    for line in open(path, encoding='utf8'):
        obj = json.loads(line)
        short_term, long_term, context = obj['short_term'], obj['long_term'], ' '.join(obj['tokens'])
        key = (long_term, context)
        if key in visited:continue
        visited.add(key)
        json.dump(obj, wl, ensure_ascii=False)
        wl.write('\n')


if __name__ == '__main__':
    #deduplicate(path='../input/dataset/general/test.json', out_path='../input/new_dataset/general/test.json')
    load(path='../input/dataset/general/train.json')