# -*- coding: utf-8 -*

import argparse
import os
import time
import math
import nltk
import scipy
from itertools import count as Count
import itertools

import jsonlines
import numpy as np
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from common_func import print_time
from dataset_to_graph import (text_to_text_parsed,
                              text_parsed_to_graph_sparse_raw,
                              graph_sparse_raw_to_pos_encoded,
                              text_parsed_to_features,
                              )
from siamese_graph import (sparse_encoded_to_torch, datapair_format)
from siamese_graph_trainer import (GBSN_linear_adjust,
                                    # load_checkpoint, LinearAdjust,
                                    to_device)
from text_to_graph import _rc_dict


# ======================================== Data fucntions ===========

# ============================== Transform text to data ==========
# ==================== Auxiliar functions ==========

def parsed_to_torch(parsed, data_type):
    if data_type in ['short', 'med', 'full']:
        graph_type = 'DiGraph'
        reduce_classes = _rc_dict[data_type]

        sparse, node_names = \
            text_parsed_to_graph_sparse_raw(parsed, reduce_classes, graph_type)
        node_attr = graph_sparse_raw_to_pos_encoded((sparse, node_names))
        return sparse_encoded_to_torch((sparse, node_attr))
    else:
        return text_parsed_to_features(parsed)


def obj_text_to_model_format(obj_text, data_type_list):
    # Define text_parsed
    remove_punct = []
    s_parsed = text_to_text_parsed(obj_text['pair'][0], remove_punct)
    t_parsed = text_to_text_parsed(obj_text['pair'][1], remove_punct)

    s_dict = {data_type: parsed_to_torch(s_parsed, data_type)
              for data_type in set(data_type_list)}
    t_dict = {data_type: parsed_to_torch(t_parsed, data_type)
              for data_type in set(data_type_list)}
    pair_list = [datapair_format(s_dict[data_type], t_dict[data_type])
                 for data_type in data_type_list]
    return pair_list, obj_text['id']


# ============================== Datasets ==========

class TestDataset(torch.utils.data.Dataset):
    "In memory dataset. Transform text to graphs on the fly"""

    __slots__ = ('jsonl_text', 'transform', 'lim', 'obj_text')

    def __init__(self, jsonl_text, data_type_list, lim=None):
        super().__init__()
        self.jsonl_text = jsonl_text
        self.data_type_list = data_type_list
        self.lim = lim
        self.obj_list = self.define_lists()
        self.len = len(self.obj_list)

    def define_lists(self):
        with jsonlines.open(self.jsonl_text) as reader_text:
            r = Count(start=0) if self.lim is None else range(self.lim)
            obj_list = []
            for count in r:
                try:
                    obj_list.append(reader_text.read())
                except EOFError:
                    print('EOF en línea: ', count)
                    break
        return obj_list

    def __getitem__(self, i):
        obj_text = self.obj_list[i]
        return obj_text_to_model_format(obj_text, self.data_type_list)

    def __len__(self):
        return self.len


# ==================== Iterable dataset ==========

def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = (int(math.ceil((overall_end - overall_start) /
                  float(worker_info.num_workers))))
    worker_id = worker_info.id
    dataset.start = overall_start + worker_id * per_worker
    dataset.end = min(dataset.start + per_worker, overall_end)


class TestDatasetIterable(torch.utils.data.IterableDataset):
    "Transform text to graphs on the fly"""

    def __init__(self, jsonl_text, data_type_list, total_lines):
        super().__init__()
        # Crea un lector del archivo pero no lee todo a memoria. Cada worker
        # tiene copia de esta clase, por lo tanto copia de este lector
        self.jsonl_text = jsonl_text
        self.data_type_list = data_type_list
        self.start = 0
        self.end = total_lines

    def __iter__(self):
        # read lines from start to end
        self.iterator = \
            itertools.islice(iter(jsonlines.open(self.jsonl_text)),
                             self.start, self.end)
        return (obj_text_to_model_format(obj_text, self.data_type_list)
                for obj_text in self.iterator)


# ======================================== Predict functions ===========

def predict(EVALUATION_DIRECTORY, OUTPUT_DIRECTORY, train_dataset='med'):
    # True to use iterable dataset, save memory in large datasets
    iterable = True
#     iterable = False

    start_time = time.time()

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # ===== Parámeters
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model
    if train_dataset == 'short':
        model_path = '../PAN22_predict/best_model_.pth'
        th_best, margin_best = 0.65, 0.20
        data_type_list = ['short', 'med',
                          'short', 'med',
                          'text_feat']
    elif train_dataset == 'med':
        model_path = '../PAN22_predict/med_best_model_sa_.pth'
        th_best, margin_best = 0.65, 0.20
        data_type_list = ['short', 'med',
                          'short', 'med',
                          'text_feat']
    elif train_dataset == 'large':
        model_path = '../PAN22_predict/large_best_model_sa_.pth'
        th_best, margin_best = 0.55, 0.20
        data_type_list = ['short', 'med',
                          'short', 'med',
                          'text_feat']

    # ===== Dataset and dataloader
    jsonl_text = os.path.join(EVALUATION_DIRECTORY, 'pairs.jsonl')
    if iterable:
        batch_size = 10
#         batch_size = 20
#         num_workers = 2
        num_workers = 4
#         num_workers = 6
#         num_workers = 8
        total_lines = sum(1 for line in open(jsonl_text))
        ds_test = TestDatasetIterable(jsonl_text, data_type_list, total_lines)
        # Dataloader
        dl_test = DataLoader(ds_test, batch_size=batch_size,
                             follow_batch=['x_s', 'x_t'],
                             num_workers=num_workers,
                             worker_init_fn=worker_init_fn)
    else:
        batch_size = 10
        num_workers = 4
        ds_test = TestDataset(jsonl_text, data_type_list)
        # Dataloader
        dl_test = DataLoader(ds_test, batch_size=batch_size,
                             follow_batch=['x_s', 'x_t'],
                             num_workers=num_workers)

#     print(ds_test[0])
#     print(next(iter(dl_test)))

    # ===== Load model
    th_adjust = (th_best, margin_best)
    # Create final model. Siamese network + adjust
    GBSN_with_adjust = GBSN_linear_adjust(model_path, th_adjust,
                                          data_type_list, device)
    # Load model
    GBSN_with_adjust.load_model()

    # Predict with adjust
    prob_ids, pred_adjust = GBSN_with_adjust.predict(dl_test)
    # Free memory
    GBSN_with_adjust.free_memory()

    # save to jsonl
    path_pred = os.path.join(OUTPUT_DIRECTORY, 'answers.jsonl')
    with jsonlines.open(path_pred, mode='w') as writer:
        for out, prob_id in zip(pred_adjust, prob_ids):
            predicted = {'id': f'{prob_id}', 'value': out}
            writer.write(predicted)

    # print_time(start_time_p, 'Predictions')
    print_time(start_time, 'Total')


def main():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    # Parser para pasar argumentos desde línea de comandos con flags
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True, help='Eval dir')
    parser.add_argument('-o', type=str, required=True, help='Output dir')
    parser.add_argument('-s', type=str, required=True, help='Training dataset')
    args = parser.parse_args()

    predict(args.i, args.o, args.s)


if __name__ == "__main__":
    main()
