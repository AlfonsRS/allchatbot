from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import torch
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from transformers import BertConfig, AutoTokenizer, TFAutoModelForSequenceClassification
import matplotlib.pyplot as plt

train = json.load(open('is_train.json'))
valid = json.load(open(PATH_TO_VALID_DS, 'r'))
test = json.load(open(PATH_TO_TEST_DS, 'r'))
oos = json.load(open(PATH_TO_OOS_DATA, 'r'))

label_names = sorted(list(set([item[1] for item in train])))
id2label = {i:label for i,label in enumerate(label_names) if label != None}
label2id = {label:i for i,label in enumerate(label_names) if label != None}
print(len(label_names), "labels in total")

labels_train = [label2id[item[1]] for item in train]
ids_train = [tokenizer(item[0], padding='max_length', max_length = MAX_LENGTH, truncation=True)['input_ids'] for item in train]

labels_valid = [label2id[item[1]] for item in valid]
ids_valid = [tokenizer(item[0], padding='max_length', max_length = MAX_LENGTH, truncation=True)['input_ids'] for item in valid]

labels_test = [label2id[item[1]] for item in test]
ids_test = [tokenizer(item[0], padding='max_length', max_length = MAX_LENGTH, truncation=True)['input_ids'] for item in test]

ids_oos = [tokenizer(item[0], padding='max_length', max_length = MAX_LENGTH, truncation=True)['input_ids'] for item in oos]


print(f'{len(ids_train)} train examples\n{len(ids_test)} test examples')