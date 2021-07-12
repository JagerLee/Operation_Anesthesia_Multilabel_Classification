import pandas as pd
import re
import jieba
import json
import string
from zhon.hanzi import punctuation
import sys

global stop_words_list, label_dict


def delete_punctuation(s):
    trantab = str.maketrans({key: None for key in string.punctuation})
    s = s.translate(trantab)
    for i in punctuation:
        s = s.replace(i, '')
    return s


def stop_words(path):
    with open(path, encoding='UTF-8') as f:
        return [l.strip() for l in f]


def get_label(df, path, path2):
    label_set = set()
    for i, label in enumerate(df['doc_label']):
        label = label.split('：')[-1]
        label = label.split(',')
        df['doc_label'][i] = label
        for i in label:
            label_set.add(i)
    label_dict = {}
    for i, label in enumerate(label_set):
        label_dict[label] = str(i)
    with open(path, 'w') as f:
        f.write('Root')
        for i in range(len(label_dict)):
            f.write(' ' + str(i))
    with open(path2, 'w') as f:
        json_str = json.dumps(label_dict, ensure_ascii=False)
        f.write('%s\n' % json_str)
    return label_dict


def data2json(train_df, output_path):
    with open(output_path,"w+",encoding='utf-8') as f:
        for indexs in train_df.index:
            dict1 = {}
            tmp_label = train_df['doc_label'][indexs]
            tmp_label = [label_dict[l] for l in tmp_label]
            dict1['doc_label'] = tmp_label
            tmp_token = train_df['doc_token'][indexs]
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            tmp_token = re.sub(reg, '', tmp_token)
            tmp_token = delete_punctuation(tmp_token)
            tmp_token = jieba.lcut(tmp_token)
            content = [x for x in tmp_token if x not in stop_words_list]
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)


def train_validate_test_split(df, validate_size=.2, test_size=.2, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    validate_num = df.shape[0] - int(test_size * df.shape[0])
    train_num = validate_num - int(validate_size * df.shape[0])
    train_df = df[:train_num]
    validate_df = df[train_num:validate_num]
    test_df = df[validate_num:]

    return train_df, validate_df, test_df


if __name__ == '__main__':
    data_path = sys.argv[1]
    taxonomy_path = 'data/label.taxonomy'
    dict_path = 'data/label_dict.json'
    train_path = 'data/train_set.json'
    validate_path = 'data/validate_set.json'
    test_path = 'data/test_set.json'

    df = pd.read_excel(data_path, header=0, usecols=[0, 4], encodding='utf-8')
    df.columns = ['doc_token', 'doc_label']
    stop_words_list = stop_words('data/stopwords.txt')
    label_dict = get_label(df, taxonomy_path, dict_path)
    train_df, validate_df, test_df = train_validate_test_split(df)
    data2json(train_df, train_path)
    data2json(validate_df, validate_path)
    data2json(test_df, test_path)

