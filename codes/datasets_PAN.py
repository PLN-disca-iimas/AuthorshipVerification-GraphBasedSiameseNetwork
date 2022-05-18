#  -*- coding: utf-8 -*-

import os
import time
# import copy
import re
import string
from itertools import count as Count
from collections import Counter
import matplotlib.pyplot as plt
import jsonlines
import pandas as pd
import numpy as np
from unidecode import unidecode
from nltk import tokenize
from tqdm import tqdm, trange
import uuid

from multiprocessing import Pool, cpu_count

try:
    from ..utils.common_func import load_obj, save_obj, time_string, print_time
except:
    import sys
    sys.path.insert(1,os.path.join(os.path.abspath('.'),".."))
    from utils.common_func import load_obj, save_obj, time_string, print_time

# ============================== Manage texts datasets==========

def wrapper_TextDataset(dataset_name, lim=None, where_to_use='server'):
    """Auxiliar function to manage path to datasets"""

    relative_path = '../PAN_datasets'
    # ========== Original datasets ==============================
    # ===== to use the pan 2014 novels dataset, train
    if dataset_name == '14-nov-train':
        path = (relative_path +
                '/pan14-author-verification/'
                'pan14-author-verification-training-corpus/'
                'pan14-author-verification-training-corpus-english-novels-'
                '2014-04-22')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2014(path, lim), param_read

    # ===== To use the PAN 2014 novels dataset, test
    if dataset_name == '14-nov-test':
        path = (relative_path +
                '/pan14-author-verification/'
                'pan14-author-verification-test-corpus/'
                'pan14-author-verification-test-corpus2-english-novels-'
                '2014-04-22')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2014(path, lim), param_read

    # ===== To use the PAN 2015 dataset, train
    if dataset_name == '15-train':
        path = (relative_path +
                '/pan15-author-verification/'
                'training/'
                'pan15-authorship-verification-training-dataset-english-'
                '2015-04-19')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2014(path, lim), param_read

    # ===== To use the PAN 2020 dataset, small bal
    if dataset_name == '20-small-bal':
        folder = (relative_path +
                  '/pan20-author-verification-small/'
                  'pan20-author-verification-balanced-500/')
        jsonl_text = folder + 'PAN2020_balanced_500_text.jsonl'
        jsonl_truth = folder + 'PAN2020_balanced_500_truth.jsonl'

        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2020(jsonl_text, jsonl_truth, lim), param_read

    # ===== To use the PAN 2020 dataset, small
    if dataset_name == '20-small-train':
        folder = (relative_path +
                  '/pan20-author-verification-small/'
                  'pan20-author-verification-small-training/')
        jsonl_text = \
            folder + 'pan20-authorship-verification-training-small.jsonl'
        jsonl_truth = \
            (folder +
             'pan20-authorship-verification-training-small-truth.jsonl')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2020(jsonl_text, jsonl_truth, lim), param_read

    # ===== To use the PAN 2020 dataset, small
    if dataset_name == '20-large-train':
        folder = (relative_path +
                  '/pan20-author-verification-large/'
                  'pan20-author-verification-large-training/')
        jsonl_text = \
            folder + 'pan20-authorship-verification-training-large.jsonl'
        jsonl_truth = \
            (folder +
             'pan20-authorship-verification-training-large-truth.jsonl')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2020(jsonl_text, jsonl_truth, lim), param_read
    
    # ===== To use the PAN 2022 dataset,
    if dataset_name == '22-train':
        folder = (relative_path +
                  '/pan22-author-verification/'
                  'training/')
        jsonl_text = \
            folder + 'pan22-authorship-verification-training.jsonl'
        jsonl_truth = \
            (folder +
             'pan22-authorship-verification-training-truth.jsonl')
        # Variable to print
        param_read = (dataset_name + '-' + str(lim))
        return TextDataset2022(jsonl_text, jsonl_truth, lim), param_read


# ==================== Original datasets readers
class TextDataset2022:
    """Clase para leer el dataset del PAN 2022"""

    def __init__(self, jsonl_text, jsonl_truth, lim=None):
        self.jsonl_text = jsonl_text
        self.jsonl_truth = jsonl_truth
        self.lim = lim
        self.problem_list, self.truth_list = self.define_lists()
        self.len = len(self.problem_list)

    def define_lists(self):
        reader_text = jsonlines.open(self.jsonl_text)
        reader_truth = jsonlines.open(self.jsonl_truth)
        if self.lim is None:
            r = Count(start=0)
        else:
            r = range(self.lim)

        problem_list = []
        truth_list = []
        for count in r:
            try:
                obj_text = reader_text.read()
                obj_truth = reader_truth.read()
            except EOFError:
                print('EOF en línea: ', count)
                break
            except InvalidLineError:
                print('línea inválida en: ', count)
                continue

            # truth list en valores 0 y 1
            truth_dict = {True: 1, False: 0}

            # Verificar que listas tengan misma longitud
            texts_num = len(obj_text['pair'])
            if (texts_num == len(obj_text['discourse_types']) and
                    texts_num == len(obj_truth['authors'])):
                problem = {'prob_id': obj_text['id'],
                           'texts': obj_text['pair'],
                           'labels': [0, 1],
                           'topics': obj_text['discourse_types'],
                           'authors': obj_truth['authors']}
                problem_list.append(problem)
                truth_list.append(truth_dict[obj_truth['same']])
            else:
                print(f'Registro de línea {count} tiene listas de distinta'
                      'longitud')

        reader_text.close()
        reader_truth.close()
        return problem_list, truth_list

    def __getitem__(self, i):
        return self.problem_list[i], self.truth_list[i]

    def __len__(self):
        return self.len
    
class TextDataset2020:
    """Clase para leer el dataset del PAN 2020"""

    def __init__(self, jsonl_text, jsonl_truth, lim=None):
        self.jsonl_text = jsonl_text
        self.jsonl_truth = jsonl_truth
        self.lim = lim
        self.problem_list, self.truth_list = self.define_lists()
        self.len = len(self.problem_list)

    def define_lists(self):
        reader_text = jsonlines.open(self.jsonl_text)
        reader_truth = jsonlines.open(self.jsonl_truth)
        if self.lim is None:
            r = Count(start=0)
        else:
            r = range(self.lim)

        problem_list = []
        truth_list = []
        for count in r:
            try:
                obj_text = reader_text.read()
                obj_truth = reader_truth.read()
            except EOFError:
                print('EOF en línea: ', count)
                break
            except InvalidLineError:
                print('línea inválida en: ', count)
                continue

            # truth list en valores 0 y 1
            truth_dict = {True: 1, False: 0}

            # Verificar que listas tengan misma longitud
            texts_num = len(obj_text['pair'])
            if (texts_num == len(obj_text['fandoms']) and
                    texts_num == len(obj_truth['authors'])):
                problem = {'prob_id': obj_text['id'],
                           'texts': obj_text['pair'],
                           'labels': [0, 1],
                           'topics': obj_text['fandoms'],
                           'authors': obj_truth['authors']}
                problem_list.append(problem)
                truth_list.append(truth_dict[obj_truth['same']])
            else:
                print(f'Registro de línea {count} tiene listas de distinta'
                      'longitud')

        reader_text.close()
        reader_truth.close()
        return problem_list, truth_list

    def __getitem__(self, i):
        return self.problem_list[i], self.truth_list[i]

    def __len__(self):
        return self.len


class TextDataset2014:
    """Clase para leer el dataset del PAN 2014 y 2015"""

    def __init__(self, path, lim=None):
        self.path = path
        self.lim = lim
        self.problem_list = self.define_problem_list()
        self.truth_list = self.define_truth_list()
        assert len(self.problem_list) == len(self.truth_list)
        self.len = len(self.truth_list)

    def traverse_folders(self):
        folders = [entry for entry in os.scandir(self.path)
                   if entry.is_dir()]
        folders.sort(key=lambda x: x.name)
        # Use only some folders
        if self.lim is not None:
            folders = folders[: self.lim]

        return folders

    def define_problem_list(self):
        problem_list = []
        for folder in self.traverse_folders():
            with os.scandir(folder) as entries:
                texts = []
                text_labels = []
                for document in entries:
                    with open(document.path, 'r') as f:
                        texts.append(f.read())
                        text_labels.append(document.name)

                prob_id = [folder.name]
                problem = {'texts': texts,
                           'text_labels': text_labels,
                           'prob_id': prob_id}
                problem_list.append(problem)

        return problem_list

    def define_truth_list(self):
        with open(self.path + '/truth.txt', 'r') as f:
            truth_raw = f.readlines()
            dic_label = {'Y': True, 'N': False}
            if self.lim is not None:
                truth = [dic_label[line[-2:-1]] for line in
                         truth_raw[:self.lim]]
            else:
                truth = [dic_label[line[-2:-1]] for line in truth_raw]

        return truth

    def __getitem__(self, i):
        return self.problem_list[i], self.truth_list[i]

    def __len__(self):
        return self.len


# ==================== To explore text datasets==========

class ExploreText:
    """Clase para procesar un texto y obtener estadísticas importates"""

    def __init__(self):
        self.npa = re.compile(r'[^\x20-\x7E]')
        self.punct = r".,;:!?()[]{}`''\"@#$^&*+-|=~_"

    def char_explore(self, text):
        # contar y enlistar caracteres no ascii
        res_all = self.npa.findall(text)
        res_spaces = [ch for ch in res_all if ch in string.whitespace]
        res_others = [ch for ch in res_all if ch not in string.whitespace]
        res_others_c = Counter(res_others)
        ud = [(num, ch, unidecode(ch)) for ch, num in res_others_c.items()]
        return res_spaces, res_others, ud

    def define_statistics(self, text):
        res_spaces, res_others, _ = self.char_explore(text)
        tokens = tokenize.word_tokenize(text)
        words = [t for t in tokens if t not in self.punct]
        sentences = tokenize.sent_tokenize(text)
        text_stats = {'tokens': len(tokens),
                      'words': len(words),
                      'sentences': len(sentences),
                      'noascii_spaces': len(res_spaces),
                      'noascii_others': len(res_others)}
        return text_stats


def define_statistics_df(df):
    """Función auxiliar para procesar el dataframe de textos"""

    explore = ExploreText()
    return df.apply(lambda x:
                    pd.concat([x,
                               pd.Series(explore.define_statistics(
                                         x['text']))]),
                    axis=1)


# ==================== Auxiliar functions =========

def parallelize_dataframe(df, func, n_cores=4):
    """Para aplicar una función a un dataframe de forma paralela"""

    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


# ==================== Transform dataset to text dict =========

def dataset_to_texts_dict(dataset, f, dest_folder):
    """Transforma dataset a un diccionario de textos y una lista de problemas.
    Primero leemos datos del dataset,
    Segundo obtenemos estadísticas de cada texto (palabras, caracteres,
    enunciados, ...)
    Tercero creamos el diccionario y la lista
    Finalmente guardamos objetos pickle de lo obtenido
    """

    start_time_p = time.time()
    # ===== Get info from dataset
    print('Define dataset stats...')
    ds_stats = []
    problem_list = []
    truth_list = []
    for i, (problem, truth) in tqdm(enumerate(dataset)):
        # Transform to unique prob id
        prob_id = problem['prob_id'] + f'{i:05}'
        problem['prob_id'] = prob_id
        problem_list.append(problem)
        truth_list.append(truth)
        for i, text in enumerate(problem['texts']):
            text_stats = {'prob_id': prob_id,
                          'label': problem['labels'][i],
                          'truth': truth,
                          'text': text,
                          # start_str, end_str and chars as text identity
                          'start_str': text[:50],
                          'end_str': text[-50:],
                          'chars': len(text)}
            keys = problem.keys()
            if 'authors' in keys:
                text_stats['author'] = problem['authors'][i]

            if 'topics' in keys:
                text_stats['topic'] = problem['topics'][i]

            ds_stats.append(text_stats)

    # ===== Define df_texts
    print('Processing stats...')
    df_stats = pd.DataFrame(ds_stats)
    df_texts = df_stats.drop_duplicates(['chars', 'start_str',
                                         'end_str'])
    # -- remove texts from df_stats
    df_stats = df_stats.drop(columns=['text'])
    df_texts['text_id'] = df_texts.index
    df_texts = df_texts.drop(columns=['prob_id', 'label', 'truth'])

    start_time_0 = time.time()
    # Versión no paralela
    # df_texts = define_statistics_df(df_texts)
    cpu_c = cpu_count()
    print('Cores: ', cpu_c)
    df_texts = parallelize_dataframe(df_texts, define_statistics_df, cpu_c)
    # 200 problemas 0.94 min en acer 1 core
    # 200 problemas 0.52 min en acer 2 cores
    print_time(start_time_0, 'parallel', f)

    # Add text_id to df_stats
    df_stats = df_stats.merge(df_texts[['text_id', 'start_str',
                                        'end_str', 'chars']],
                              on=['start_str',
                                  'end_str', 'chars'])
    df_stats.drop(columns=['start_str', 'end_str', 'chars'],
                  inplace=True)

    # ===== Define ds_list
    print('Define dataset list with new prob ids...')
    df_prob = df_stats.set_index(['prob_id', 'label'])['text_id']
    for i, problem in tqdm(enumerate(problem_list)):
        prob_id = problem['prob_id']
        text_ids = []
        for label in problem['labels']:
            text_id = df_prob.loc[prob_id, label]
            text_ids.append(text_id)

        problem['text_ids'] = text_ids
        del problem['texts']

    # ===== Define texts_dict
    texts_dict = df_texts[['text_id', 'text']].set_index('text_id')[
            'text'].to_dict()
    df_texts.drop(columns='text', inplace=True)

    # Print time
    print_time(start_time_p, 'Process', f)
    start_time_s = time.time()

    # ===== Save
    folder = os.path.join(dest_folder, 'to_texts_dict')
    if not os.path.exists(folder):
        os.makedirs(folder)

    stats_dict = {'df_texts': df_texts,
                  'df_stats': df_stats}
    save_obj(stats_dict, os.path.join(folder, 'stats_dict'))
    ds_list = {'problem_list': problem_list,
               'truth_list': truth_list}
    save_obj(ds_list, os.path.join(folder, 'ds_list'))
    save_obj(texts_dict, os.path.join(folder, 'texts_dict'))

    # Print time
    print_time(start_time_s, 'Save', f)
    return stats_dict, ds_list, texts_dict


def count_stats(stats_dict, ds_list, f, dest_folder=None):
    """Función para graficar y obtener un resumen de las estadísticas
    obtenidas de los textos"""

    # ===== Unpack objects
    df_texts = stats_dict['df_texts']
    problem_list = ds_list['problem_list']
    truth_list = ds_list['truth_list']

    # ===== Print general stats
    print('\n=============== Problem stats', file=f)
    print(f"Total problems: {len(problem_list)}", file=f)
    print("Total problems per class:", file=f)
    print(f"True: {truth_list.count(True)}", file=f)
    print(f"False: {truth_list.count(False)}", file=f)
    print('\n=============== Texts stats', file=f)
    print(f"Columns in dataframe: {df_texts.columns}", file=f)
    print('==========', file=f)
    print(f"Total texts: {df_texts.shape[0]}", file=f)
    col_stats = ['chars',
                 'tokens',
                 'words',
                 'sentences',
                 'noascii_spaces',
                 'noascii_others']
    for col in col_stats:
        print(f"{str(col)}: {int(df_texts[col].mean())} / "
              f"{df_texts[col].min()} - {df_texts[col].max()}", file=f)

    # ===== Print author and topic stats, plot frequencies
    print('==========', file=f)
    time_stamp = time_string()
    if 'author' in df_texts.columns:
        print("Authors:", file=f)
        authors = df_texts['author'].value_counts()
        print(f"mean: {authors.mean()} / "
              f"{authors.min()} - {authors.max()}", file=f)
        print(authors, file=f)
        plt.bar(range(len(authors.values)), authors.values)
        name = 'authors_' + os.path.basename(f.name)[:-4] + time_stamp
        plt.savefig(os.path.join(str(dest_folder), name) + '.png')
        plt.clf()

    if 'topic' in df_texts.columns:
        print("Topics:", file=f)
        topics = df_texts['topic'].value_counts()
        print(f"Text per topic: {topics.mean()} / "
              f"{topics.min()} - {topics.max()}", file=f)
        print(topics, file=f)
        plt.bar(range(len(topics.values)), topics.values)
        name = 'topics_' + os.path.basename(f.name)[:-4] + time_stamp
        plt.savefig(os.path.join(str(dest_folder), name) + '.png')
        plt.clf()


def define_clean_dataframe(stats_dict, ds_list, texts_dict, f, dest_folder,
                           k=10, tokens_min=1):
    """Función que toma las estadísticas y limpia el dataset.
    Removemos problemas con mismo par de textos
    Removemos problemas con textos de menos de 200 tokens
    """

    start_time_p = time.time()

    # ===== Unpack objects
    df_texts = stats_dict['df_texts']
    df_stats = stats_dict['df_stats']
    problem_list = ds_list['problem_list']
    truth_list = ds_list['truth_list']
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # ===== Define clean stats
    # Problemas duplicados. Problemas con mismo par de textos
    problems = df_stats[['prob_id', 'text_id']].groupby('prob_id')[
            'text_id'].apply(frozenset)
    pair_count = problems.value_counts()
    dup_val = pair_count[pair_count > 1]
    dup = problems[problems.isin(dup_val.index)]
    print('Duplicated problems:', file=f)
    print(dup, file=f)

    # textos con pocos tokens
    few_tok_texts = df_texts[['text_id', 'start_str', 'end_str', 'chars']][
            df_texts['tokens'] < tokens_min]
    print(f"Texts with less than {tokens_min} tokens:", file=f)
    print(few_tok_texts, file=f)
    print(f"Problems with texts with less than {tokens_min} tokens:", file=f)
    ftt = few_tok_texts['text_id'].values
    few_tok_prob_ids = \
        df_stats[df_stats['text_id'].isin(ftt)]['prob_id'].unique()
    print(few_tok_prob_ids, file=f)
    
    # Define df_stats and df_texts clean
    exclude_ids = np.concatenate([dup.index.to_numpy(), few_tok_prob_ids])
    df_stats_clean = df_stats[~df_stats['prob_id'].isin(exclude_ids)]
    df_texts_clean = df_texts[~df_texts['text_id'].isin(ftt)]

    # ===== Define clean ds_list
    print('Define clean ds_list...')
    problem_clean = []
    truth_clean = []
    for problem, truth in zip(problem_list, truth_list):
        if problem['prob_id'] not in exclude_ids:
            truth_clean.append(truth)
            problem_clean.append(problem)

    # ===== Define new texts_dict
    for prob_id in list(ftt):
        del texts_dict[prob_id]

    # Print time
    print_time(start_time_p, 'Process', f)
    start_time_s = time.time()

    # ===== Save clean stats
    folder = os.path.join(dest_folder, 'clean')
    if not os.path.exists(folder):
        os.makedirs(folder)

    stats_dict_clean = {'df_stats': df_stats_clean,
                        'df_texts': df_texts_clean}
    save_obj(stats_dict_clean, os.path.join(folder, 'stats_dict_clean'))
    ds_list_clean = {'problem_list': problem_clean,
                     'truth_list': truth_clean}
    save_obj(ds_list_clean, os.path.join(folder, 'ds_list_clean'))
    # OJO, ES dest_folder
    save_obj(texts_dict, os.path.join(dest_folder, 'texts_dict_clean'))

    # Print time
    print_time(start_time_s, 'Save', f)
    return stats_dict_clean, ds_list_clean, texts_dict


# ============================== Split PAN 20 dataset in train, val, test ==

def asociated_ids_author(df, author_list):
    """Se utiliza en define_ids_partition
    Dado una lista de autores encuentra todos los problemas con algún
    texto de algún autor en la lista,
    luego la lista de todos los autores en estos problemas.
    Así recursivamente..."""
    
    
    # Prob_list from authors
    mask = df['author'].isin(author_list)
    if not mask.any():
        return []
    prob_list = df[mask]['prob_id'].unique()
    # Author_list from problems
    mask = df['prob_id'].isin(prob_list)
    author_list_2 = df[mask]['author'].unique()
    
    df = df[~mask]
    # Recursive call
    prob_ids_rec = asociated_ids_author(df, author_list_2)
    prob_ids = list(prob_list)
    prob_ids.extend(prob_ids_rec)
    return prob_ids


def asociated_ids_author_topic(df, author_list):
    """Se utiliza en define_ids_partition
    Dado una lista de autores encuentra todos los problemas con algún
    texto de algún autor en la lista,
    luego la lista de todos los temas en esos problemas,
    luego todos los problemas con algún texto con alguno de esos temas,
    luego la lista de todos los autores en estos problemas.
    Así recursivamente..."""

    # Prob_list from authors
    mask = df['author'].isin(author_list)
    if not mask.any():
        return []

    prob_list_1 = df[mask]['prob_id'].unique()
    # Topic_list from problems
    mask = df['prob_id'].isin(prob_list_1)
    topic_list = df[mask]['topic'].unique()
    # Prob_list from topics
    mask = df['topic'].isin(topic_list)
    prob_list_2 = df[mask]['prob_id'].unique()
    # Author_list from problems
    mask = df['prob_id'].isin(prob_list_2)
    author_list_2 = df[mask]['author'].unique()
    df = df[~mask]
    # Recursive call
    prob_ids_2 = asociated_ids_author_topic(df, author_list_2)
    prob_ids = list(prob_list_2)
    prob_ids.extend(prob_ids_2)
    return prob_ids


def define_ids_partition(stats_dict_clean, dest_folder, f,
                         asociated_fn=asociated_ids_author, label='authors'):
    """Particiona la lista de problemas en conjuntos tales que entre dos de
    ellos no compartan autores"""

    start_time_p = time.time()

    #----------------
    df_s = stats_dict_clean['df_stats']
    author_counts = df_s['author'].value_counts()
    cantidad_textos = []
    authorsByNumText = []
    #Se obtiene el numero de textos por autor
    for author in author_counts.index:
        mask = df_s['author'].isin([author])
        text_ids = df_s[mask]['text_id'].unique()
        cantidad_textos.append((author,len(text_ids)))
        
    #Se ordena del autor de mayor textos al de menor
    cantidad_textos.sort(key = lambda x: x[1], reverse=True) 
    authorsByNumText = [author for (author,n) in cantidad_textos]
    #El el segundo parametro del split define el numero de 
    #grupos con autores ajenos
    athorsToSplit = np.array_split(authorsByNumText, 12)

    toRemove = []
    print("Total de problemas limpios: ",len(df_s)/2)
    for i,group in enumerate(athorsToSplit):
        mask = df_s['author'].isin(group) #Autores solo de un grupo
        prob_list = df_s[mask]['prob_id'].unique() #Problemas asociados al grupo
    
        #todos los registros de esos probelmas pero que no pertenezcan al grupo
        mask = df_s['prob_id'].isin(prob_list) & ~df_s['author'].isin(group) 
        #Autores que comparten probelmas con el grupo pero que no son del grupo
        author_list_2 = df_s[mask]['author'].unique() 
    
        #Se debe de buscar los probelmas de estos autores que concuerden con los del grupo
        mask = df_s['author'].isin(author_list_2)
        prob_list2 = df_s[mask]['prob_id'].unique()
    
        #De esta lista se deben eliminar lo que compartan con los del grupo
        mask = df_s['prob_id'].isin(prob_list2) & df_s['author'].isin(group)
        toDelete = df_s[mask]['prob_id'].unique()
        print(len(toDelete),"eliminados del grupo ",i)
        toRemove.extend(toDelete)
    
        
    df_tmp = df_s[~df_s['prob_id'].isin(set(toRemove))]


    print("Despues de eliminar los problemas relacionados entre grupos")
    print("Total de problemas: ",len(df_tmp['prob_id'])/2)

    #----------------

    # ===== Split prob_ids in sets of disjoint authors (and maybe topics)
    df = df_tmp[['prob_id', 'author', 'topic']]
    df_truth = df_tmp[
            ['prob_id', 'truth']].drop_duplicates()   
    
    total_problems = len(df_truth)
    # Split algorithm
    author_counts = df['author'].value_counts()
    ids_partition = []
    part_num = 0      
    
    authorsByRelation = []
    for author in author_counts.index:
        mask = df['author'].isin([author])
        prob_list = df[mask]['prob_id'].unique()
        # Author_list from problems
        mask = df['prob_id'].isin(prob_list)
        author_list_2 = df[mask]['author'].unique()
        authorsByRelation.append((author,len(author_list_2)))
    
    authorsByRelation.sort(key=lambda tup: tup[1])  
    
    for (author,rel) in authorsByRelation:
        
        prob_ids = asociated_fn(df, [author])
        if len(prob_ids) == 0:
            continue

        truth_count = \
            df_truth[df_truth['prob_id'].isin(prob_ids)][
                    'truth'].value_counts()
        part = {'part_num': part_num,
                'prob_ids': prob_ids,
                'len': len(prob_ids),
                'truth': truth_count[1] if 1 in truth_count.index else 0}
        ids_partition.append(part)
        part_num += 1
        print('part:   len:', part['len'], 'truth: ', part['truth'], file=f)
        df = df[~df['prob_id'].isin(prob_ids)]
            
    print('Total partitions: ', len(ids_partition), file=f)
    assert sum([part['len'] for part in ids_partition]) == total_problems
    # Print time
    print_time(start_time_p, 'Process', f)
    start_time_s = time.time()

    # ===== Save
    folder = os.path.join(dest_folder, 'partitions')
    if not os.path.exists(folder):
        os.makedirs(folder)

    partition = (ids_partition, total_problems)
    save_obj(partition, os.path.join(folder, 'partition-' + label))
    # Print time
    print_time(start_time_s, 'Save', f)

    # Close log
    f.close()
    return partition


def define_splits_indexes(partition, dest_folder, f,
                          split_proportion=0.1, split_num=10,
                          bal_lim=0.45):
    """Utiliza la partición de problemas para crear conjuntos de cierto
    tamaño que sean balanceados (similar cantidad de problemas positivos y
    negativos)"""

    start_time_p = time.time()

    # ===== Unpack objects
    total_problems = partition[1]
    ids_partition = partition[0]
    
    # ===== Group parts in balanced splits
    df = pd.DataFrame(ids_partition)
    df.set_index('part_num')
    df_val = df[['len', 'truth']]
    df_ids = df[['prob_ids']]
    
    # Define positive and negative lists
    mask = (df_val['truth'] / df_val['len']) >= 0.5
    
    df_pos = df_val[mask].sort_values(['len', 'truth'], ascending=False)
    df_neg = df_val[~mask].sort_values('truth')
    df_neg = df_neg.sort_values('len', ascending=False)

    if isinstance(split_proportion, float):
        split_size = total_problems*split_proportion
    else:
        split_size = split_proportion
    verif_lim = split_size * (1 - bal_lim)
    part_splits = []
    part_splits_totals = []
    for i in trange(split_num):
        total_len = 0
        total_truth = 0
        partitions = []
        cont = True
        while total_len < split_size and cont:
            # Elige positivo o negativo
            if (total_len == 0) or (total_truth / total_len) <= 0.5:
                df_op = df_pos
            else:
                df_op = df_neg
            
            cont = False
            for index, part in df_op.iterrows():
                new_total_truth = total_truth + part['truth']
                new_total_len = total_len + part['len']
                if (new_total_truth <= verif_lim and
                    new_total_len - new_total_truth <= verif_lim and
                        new_total_len <= split_size):
                    cont = True
                    # Sí nos sirve
                    partitions.append(index)
                    df_op.drop(index=index, inplace=True)
                    total_truth = new_total_truth
                    total_len = new_total_len
                    print(i, 'Add:      ', index, total_truth, total_len,
                          file=f)
                    break

        part_splits.append(partitions)
        part_splits_totals.append((total_len, total_truth))

    # Add remaining problems in list
    part_splits.append(list(df_pos.index) + list(df_neg.index))
    part_splits_totals.append((df_pos['len'].sum() + df_neg['len'].sum(),
                               df_pos['truth'].sum() + df_neg['truth'].sum()))
    print(part_splits_totals, file=f)
    
    print(part_splits)
    # Define ids_splits
    ids_splits = [np.concatenate([df_ids['prob_ids'].iloc[index]
                                  for index in partition])
                  for partition in part_splits]
    
    # Print time
    print_time(start_time_p, 'Process', f)
    start_time_s = time.time()

    # ===== Save
    folder = os.path.join(dest_folder, 'partitions')
    save_obj(ids_splits, os.path.join(folder, 'ids_splits'))
    save_obj((part_splits, part_splits_totals),
             os.path.join(folder, 'part_splits'))

    # Print time
    print_time(start_time_s, 'Save', f)

    # Close log
    f.close()
    return ids_splits


def define_dataset_splits(ds_list, indexes, dest_folder):
    """Dadas tres listas con los ids de problemas para train, val y test,
    crea tres objetos ds_list con los correspondientes datos.
    Además cuenta y guarda número de problemas positivos.
    Además guarda lista de id de textos que se utilizan en cada lista"""

    # ===== Unpack objects
    problem_list = ds_list['problem_list']
    truth_list = ds_list['truth_list']
    split_train = indexes['train']
    split_val = indexes['val']
    split_test = indexes['test']

    # ===== Define ds_list for train, val and test
    ds_list_train = {'problem_list': [], 'truth_list': [],
                     'truth_count': 0}
    text_ids_train = []
    ds_list_val = {'problem_list': [], 'truth_list': [],
                   'truth_count': 0}
    text_ids_val = []
    ds_list_test = {'problem_list': [], 'truth_list': [],
                    'truth_count': 0}
    text_ids_test = []
    for problem, truth in tqdm(zip(problem_list, truth_list)):
        if problem['prob_id'] in split_train:
            ds_list_train['problem_list'].append(problem)
            ds_list_train['truth_list'].append(truth)
            text_ids_train.extend(problem['text_ids'])
            if truth == 1:
                ds_list_train['truth_count'] += 1

        if problem['prob_id'] in split_val:
            ds_list_val['problem_list'].append(problem)
            ds_list_val['truth_list'].append(truth)
            text_ids_val.extend(problem['text_ids'])
            if truth == 1:
                ds_list_val['truth_count'] += 1

        if problem['prob_id'] in split_test:
            ds_list_test['problem_list'].append(problem)
            ds_list_test['truth_list'].append(truth)
            text_ids_test.extend(problem['text_ids'])
            if truth == 1:
                ds_list_test['truth_count'] += 1

    ds_list_train['text_id_list'] = list(set(text_ids_train))
    ds_list_val['text_id_list'] = list(set(text_ids_val))
    ds_list_test['text_id_list'] = list(set(text_ids_test))

    # ===== Save
    save_obj(ds_list_train, os.path.join(dest_folder, 'ds_list_train'))
    save_obj(ds_list_val, os.path.join(dest_folder, 'ds_list_val'))
    save_obj(ds_list_test, os.path.join(dest_folder, 'ds_list_test'))
    return ds_list_train, ds_list_val, ds_list_test


def unique_authors_topics(ds_list, df_stats):
    prob_ids = [item['prob_id']
                for item in ds_list['problem_list']]
    mask = df_stats['prob_id'].isin(prob_ids)
#     authors = df_stats[mask]['author'].unique()
#     topics = df_stats[mask]['topic'].unique()
    authors = set(df_stats[mask]['author'])
    topics = set(df_stats[mask]['topic'])
    return authors, topics


def report_intersections(ds_list_1, ds_list_2, authors_1, authors_2,
                         topics_1, topics_2, f):
    int_texts = (set(ds_list_1['text_id_list']) &
                 set(ds_list_2['text_id_list']))
    print('Texts: ', len(int_texts), file=f)
    print(int_texts, file=f)

    int_authors = authors_1 & authors_2
    print('authors:', len(int_authors), file=f)
    print(int_authors, file=f)

    int_topics = topics_1 & topics_2
    print('topics:', len(int_topics), file=f)
    print(int_topics, file=f)


def verify_splits(ds_list_train, ds_list_val, ds_list_test,
                  stats_dict_clean, dest_folder, f):
    df_stats = stats_dict_clean['df_stats']

    authors_train, topics_train = \
        unique_authors_topics(ds_list_train, df_stats)
    authors_val, topics_val = \
        unique_authors_topics(ds_list_val, df_stats)
    authors_test, topics_test = \
        unique_authors_topics(ds_list_test, df_stats)

    # Print proportion
    print('\n----- Problems in split and positive examples:', file=f)
    print('Train:', file=f)
    print(len(ds_list_train['problem_list']),
          '. Truth ', ds_list_train['truth_count'], file=f)
    print('Val:', file=f)
    print(len(ds_list_val['problem_list']),
          '. Truth ', ds_list_val['truth_count'], file=f)
    print('Test:', file=f)
    print(len(ds_list_test['problem_list']),
          '. Truth ', ds_list_test['truth_count'], file=f)

    print('\n----- Intersections:', file=f)
    print('train and val', file=f)
    report_intersections(ds_list_train, ds_list_val,
                         authors_train, authors_val,
                         topics_train, topics_val, f)
    print('\ntrain and test', file=f)
    report_intersections(ds_list_train, ds_list_test,
                         authors_train, authors_test,
                         topics_train, topics_test, f)
    print('\ntest and val', file=f)
    report_intersections(ds_list_test, ds_list_val,
                         authors_test, authors_val,
                         topics_test, topics_val, f)

    f.close()


# ============================== Pipeline to process dataset ==========
def dataset_pipeline():
    """Función para procesar el dataset del pan 2022.
    Se obtienen estadísticas, se transforma a text_dict,
    se limpia,
    se obtiene particion y conjuntos train, val y test.
    """

    #dataset_name = '20-small-bal'
    dataset_name = '22-train'
#     dataset_name = '20-large-train'
    lim = None
    p_lim = '' if lim is None else ('_' + str(lim))

    dest_folder = os.path.join('../data/PAN22_text_split',
                               dataset_name + p_lim)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    start_time = time.time()
    dataset, param_read = wrapper_TextDataset(dataset_name, lim)

    print('Executing dataset_to_texts_dict...')
    log_name = os.path.join(dest_folder, '01_dataset_to_texts_dict.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    stats_dict, ds_list, texts_dict = \
        dataset_to_texts_dict(dataset, f, dest_folder)
    count_stats(stats_dict, ds_list, f, dest_folder)
    f.close()

    print('Executing define_clean_dataframe...')
    log_name = os.path.join(dest_folder, '02_define_clean_dataframe.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    stats_dict_clean, ds_list_clean, texts_dict_clean = \
        define_clean_dataframe(stats_dict, ds_list, texts_dict,
                                f, dest_folder)
    count_stats(stats_dict_clean, ds_list_clean, f, dest_folder)
    f.close()


    stats_dict_clean_path = os.path.join(dest_folder,
                                         'clean/stats_dict_clean')
    stats_dict_clean = load_obj(stats_dict_clean_path, fast=True)
    label = 'authors'
    print('Executing define_ids_partition (' + label + ')...')
    log_name = os.path.join(dest_folder, '03_define_ids_partition-' +
                            label + '.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    partition_authors = \
        define_ids_partition(stats_dict_clean, dest_folder, f)

# #     Uncomment next lines to try split dataset in authors and topics
#    print('Executing define_ids_partition (authors and topics)...')
#    label = 'authors_topics'
#    log_name = os.path.join(dest_folder, '04_define_ids_partition-' +
#                            label + '.txt')
#    print('log save in: ' + log_name)
#    f = open(log_name, 'w+')
#    partition_authors_topics = \
#        define_ids_partition(stats_dict_clean, dest_folder,f,
#                             asociated_fn=asociated_ids_author_topic,
#                             label=label)

    partition_authors_path = os.path.join(dest_folder, 'partitions/partition-authors')
    partition_authors = load_obj(partition_authors_path, fast=True)
    print('Executing define_splits_indexes...')
    log_name = os.path.join(dest_folder, '05_define_splits_indexes.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    
    ids_splits = define_splits_indexes(partition_authors, dest_folder, f,
                                     split_proportion=0.1,
                                       split_num=5, bal_lim=0.4)

    ds_list_clean = os.path.join(dest_folder, 'clean/ds_list_clean')
    ds_list_clean = load_obj(ds_list_clean, fast=True)
    ids_splits = os.path.join(dest_folder, 'partitions/ids_splits')
    ids_splits = load_obj(ids_splits, fast=True)
    indexes = {'test': ids_splits[0],
               'val': ids_splits[1],
               'train': np.concatenate(ids_splits[2:])}
    print('Executing define_dataset_splits...')
    ds_list_train, ds_list_val, ds_list_test = \
        define_dataset_splits(ds_list_clean, indexes, dest_folder)

    ds_list_train_path = os.path.join(dest_folder, 'ds_list_train')
    ds_list_train = load_obj(ds_list_train_path, fast=True)
    ds_list_val_path = os.path.join(dest_folder, 'ds_list_val')
    ds_list_val = load_obj(ds_list_val_path, fast=True)
    ds_list_test_path = os.path.join(dest_folder, 'ds_list_test')
    ds_list_test = load_obj(ds_list_test_path, fast=True)
    stats_dict_clean_path = \
         os.path.join(dest_folder, 'clean/stats_dict_clean')
    stats_dict_clean = load_obj(stats_dict_clean_path, fast=True)
    print('Executing verify_splits...')
    log_name = os.path.join(dest_folder, '05_verify_splits.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    verify_splits(ds_list_train, ds_list_val, ds_list_test,
                  stats_dict_clean, dest_folder, f)
    
    #Generar nuevos probelmas, solo para el datasetPan2022
    stats_dict_clean_path = os.path.join(dest_folder,
                                         'clean/stats_dict_clean')
    stats_dict_clean = load_obj(stats_dict_clean_path, fast=True)
    
    news_t,news_f = generate_new_probelms(ds_list_train,stats_dict_clean)
    ds_list_train_n = joinNewDatasets(ds_list_train,stats_dict_clean,news_t,news_f)
    save_obj(ds_list_train_n, os.path.join(dest_folder, 'ds_list_train_n'))
    
    news_t,news_f = generate_new_probelms(ds_list_val,stats_dict_clean)
    ds_list_val_n = joinNewDatasets(ds_list_val,stats_dict_clean,news_t,news_f)
    save_obj(ds_list_val_n, os.path.join(dest_folder, 'ds_list_val_n'))
    
    news_t,news_f = generate_new_probelms(ds_list_test,stats_dict_clean)
    ds_list_test_n = joinNewDatasets(ds_list_test,stats_dict_clean,news_t,news_f)
    save_obj(ds_list_test_n, os.path.join(dest_folder, 'ds_list_test_n'))
    
    # ========== Print time
    print_time(start_time, 'All process')


def compare_datasets():
    small_folder = os.path.join('../data/PAN20_text_split/20-small-train')
    small_stats_dict_clean_path = \
        os.path.join(small_folder, 'clean/stats_dict_clean')
    small_stats_dict_clean = load_obj(small_stats_dict_clean_path, fast=True)
    small_df_stats = small_stats_dict_clean['df_stats']
#     small_df_texts = small_stats_dict_clean['df_texts']

    large_folder = os.path.join('../data/PAN20_text_split/20-small-bal')
    large_folder = os.path.join('../data/PAN20_text_split/20-large-train')
    large_stats_dict_clean_path = \
        os.path.join(large_folder, 'clean/stats_dict_clean')
    large_stats_dict_clean = load_obj(large_stats_dict_clean_path, fast=True)
    large_df_stats = large_stats_dict_clean['df_stats']

    small_df_stats['prob_id'] = \
        small_df_stats['prob_id'].apply(lambda x: x[:-5])
    large_df_stats['prob_id'] = \
        large_df_stats['prob_id'].apply(lambda x: x[:-5])
#     small_pid = small_df_stats['prob_id'].apply(lambda x: x[:-5])
#     large_pid = large_df_stats['prob_id'].apply(lambda x: x[:-5])
    small_pid = small_df_stats['prob_id']
    large_pid = large_df_stats['prob_id']
    int_pid = set(small_pid) & set(large_pid)
#     mask = small_df_stats['prob_id'].isin(int_pid)
#     int_authors = list(set(small_df_stats['author'][mask]))
    mask = large_df_stats['prob_id'].isin(int_pid)
    int_authors = list(set(large_df_stats['author'][mask]))

    # Cerradura de autor
    small_author_cls = asociated_ids_author(small_df_stats, int_authors)
    large_author_cls = asociated_ids_author(large_df_stats, int_authors)

    print(len(int_pid))
    print(len(small_author_cls))
    print(len(large_author_cls))

    dest_folder = small_folder

    # definir small sin problemas en cerradura
    mask = small_df_stats['prob_id'].isin(small_author_cls)
    small_df = small_df_stats[mask]

    # obtener splits
    small_dict = {'df_stats': small_df}
    label = 'authors'
    print('Executing define_ids_partition (' + label + ')...')
    log_name = os.path.join(dest_folder, '03-2_define_ids_partition-' +
                            label + '.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    partition_authors = \
        define_ids_partition(small_dict, dest_folder, f)

    print('Executing define_splits_indexes...')
    log_name = os.path.join(dest_folder, '04-2_define_splits_indexes.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    ids_splits = define_splits_indexes(partition_authors, dest_folder, f,
                                       split_proportion=5200,
                                       split_num=2, bal_lim=0.4)

    # definir small sin problemas en cerradura
    mask = small_df_stats['prob_id'].isin(small_author_cls)
    small_df = small_df_stats[mask]

    # obtener splits
    small_dict = {'df_stats': small_df}
    dest_folder = small_folder
    label = 'authors'
    print('Executing define_ids_partition (' + label + ')...')
    log_name = os.path.join(dest_folder, '03-2_define_ids_partition-' +
                            label + '.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    partition_authors = \
        define_ids_partition(small_dict, dest_folder, f)

    print('Executing define_splits_indexes...')
    log_name = os.path.join(dest_folder, '04-2_define_splits_indexes.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    ids_splits = define_splits_indexes(partition_authors, dest_folder, f,
                                       split_proportion=5200,
                                       split_num=2, bal_lim=0.4)

# ============================== Define partitions Dataset PAN2022 ============
"""
Agrega los problemas creados a un nuevo dataset junto con los creados 
anteriormente
"""
def joinNewDatasets(ds_list,stats_dict_clean,news_t,news_f):
    
    df_texts = stats_dict_clean['df_texts']
    
    pm_list = ds_list['problem_list']
    tru_list = ds_list['truth_list']
    tru_count = ds_list['truth_count']
    txt_list = ds_list['text_id_list']
    neg = len(tru_list) - tru_count
    print("Antes de agregar nuevos = ",len(tru_list))
    
    for new in tqdm(news_t):
        topic_1 = df_texts['topic'][df_texts['text_id']==new[0]].to_numpy()[0]
        topic_2 = df_texts['topic'][df_texts['text_id']==new[1]].to_numpy()[0]
        author_1 = df_texts['author'][df_texts['text_id']==new[0]].to_numpy()[0]
        author_2 = df_texts['author'][df_texts['text_id']==new[1]].to_numpy()[0]
        
        new_item = {}
        new_item['prob_id'] = str(uuid.uuid4())
        new_item['labels'] = [0,1]
        new_item['topics'] = [topic_1,topic_2] 
        new_item['authors'] = [author_1,author_2]
        new_item['text_ids'] = [new[0],new[1]]
        
        pm_list.append(new_item)
        tru_list.append(1)
        tru_count = tru_count + 1
        txt_list.append(new[0])
        txt_list.append(new[1])
        
        
    for new in tqdm(news_f[0:(tru_count-neg)]):
        topic_1 = df_texts['topic'][df_texts['text_id']==new[0]].to_numpy()[0]
        topic_2 = df_texts['topic'][df_texts['text_id']==new[1]].to_numpy()[0]
        author_1 = df_texts['author'][df_texts['text_id']==new[0]].to_numpy()[0]
        author_2 = df_texts['author'][df_texts['text_id']==new[1]].to_numpy()[0]
        
        new_item = {}
        new_item['prob_id'] = str(uuid.uuid4())
        new_item['labels'] = [0,1]
        new_item['topics'] = [topic_1,topic_2] 
        new_item['authors'] = [author_1,author_2]
        new_item['text_ids'] = [new[0],new[1]]
        
        pm_list.append(new_item)
        tru_list.append(0)
        txt_list.append(new[0])
        txt_list.append(new[1])
        
    new_dict = {}
    new_dict['problem_list'] = pm_list
    new_dict['truth_list'] = tru_list
    new_dict['truth_count'] = tru_count
    new_dict['text_id_list'] = list(set(txt_list))
    
    return new_dict

"""
Esta funcion es especial para el datasetpan 2022 ya que como se eliminaron 
muchos problemas para lograr particiones con autores ajenos, se tendrá
que crear nuevos problemas
"""
def generate_new_probelms(ds_list,stats_dict_clean):
    
    df_stats = stats_dict_clean['df_stats']
    df_texts = stats_dict_clean['df_texts']
    
    df_list_T = pd.DataFrame(ds_list['problem_list'])
    textIds_list = df_list_T['text_ids'].to_numpy()
    
    authors_list = [auth for prob in ds_list['problem_list'] for auth in prob['authors']]
    #Autores unicos en particion
    authors = set(authors_list)
    id_texts_by_author = id_text_by_author(df_stats)
    
    news_t = generate_new_true_problems(authors,id_texts_by_author,df_texts,textIds_list)
    news_f = generate_new_false_problems(authors,id_texts_by_author,df_texts,textIds_list)
    
    return news_t,news_f
    
    
def id_text_by_author(df_stats):
    id_texts_by_author = {}
    author_counts = df_stats['author'].value_counts()
    for author in author_counts.index:
        mask = df_stats['author'].isin([author])
        text_ids = df_stats[mask]['text_id'].unique()
        id_texts_by_author[author] = text_ids
    return id_texts_by_author

def generate_new_true_problems(authors,id_texts_by_author,df_texts,textIds_list):
    news_t = []
    print('Generando nuevos probelmas positivos...')
    for author in tqdm(authors):
        text_ids = id_texts_by_author[author] 
        for p in combinantorial(text_ids):
    
            if(p[0]==p[1]):
                continue
                
            #ver si sigue el patron
            topic_txt1 = df_texts['topic'][df_texts['text_id']==p[0]].to_numpy()
            topic_txt2 = df_texts['topic'][df_texts['text_id']==p[1]].to_numpy()
            
            if(topic_txt1.size == 0 or topic_txt2.size == 0):
                continue
            
            if(topic_txt1[0]=='text_message'):
                continue
            
            add_n = True
            #Se checa si ya existe la pareja
            for ids_txt in textIds_list:
                if(p==ids_txt or [p[1],p[0]]==ids_txt):
                    add_n = False
                    break
                
            if(add_n):
                news_t.append(p)
    
    return news_t


def generate_new_false_problems(authors,id_texts_by_author,df_texts,textIds_list):
    news_f = []
    print('Generando nuevos probelmas falsos...')
    for index,author_s in enumerate(tqdm(authors)):
        
        for jndex,author in enumerate(authors):
            
            if(index<=jndex):
                continue
            
            if(author_s == author):
                continue
                
            a_v = id_texts_by_author[author_s]
            b_v = id_texts_by_author[author]
            mesh = np.array(np.meshgrid(a_v, b_v))
            combinations = mesh.T.reshape(-1, 2)
                    
            for i in range(combinations.shape[0]):
                p = [combinations[i,:][0],combinations[i,:][1]]
    
                if(p[0]==p[1]):
                    continue
    
                #checa si son del mismo autor
                auth_txt1 = df_texts['author'][df_texts['text_id']==p[0]].to_numpy()
                auth_txt2 = df_texts['author'][df_texts['text_id']==p[1]].to_numpy()
    
                if(auth_txt1.size == 0 or auth_txt2.size == 0):
                    continue
    
                if(auth_txt1[0] == auth_txt2[0]):
                    continue
    
                #checa si sigue el patron de topicos
                topic_txt1 = df_texts['topic'][df_texts['text_id']==p[0]].to_numpy()
                topic_txt2 = df_texts['topic'][df_texts['text_id']==p[1]].to_numpy()
    
                if(topic_txt1.size == 0 or topic_txt2.size == 0):
                    continue
    
                if(topic_txt1[0]=='text_message'):
                    continue   
    
                add_n = True
                #Se checa si ya existe la pareja
                for ids_txt in textIds_list:
                    if(p==ids_txt or [p[1],p[0]]==ids_txt):
                        add_n = False
                        break
                        
                if(add_n):
                    news_f.append(p)
    return news_f


def combinantorial(lst):
    count = 0
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            yield [element1, element2]
        index += 1
# ============================== Join dataset from ds_file and doc_dict =======

def fit_dict(doc_dict, list_of_ds):
    list_of_dict = [{key: doc_dict[key] for key in ds_list['text_id_list']}
                    for ds_list in list_of_ds]
    return list_of_dict


# ============================== get pairs.json from test_dataset =======

def get_pairs():
#     dataset_name = '20-small-bal'
#     dataset_name = '20-small-train'
    dataset_name = '20-large-train'

    text_dict_path = \
        os.path.join('../data/PAN20_text_split', dataset_name,
                     'to_texts_dict/texts_dict.pkl')
    ds_path = \
        os.path.join('../data/PAN20_text_split', dataset_name,
                     'ds_list_test.pkl')
    dest_folder = \
        os.path.join('../data/PAN20_text_split', dataset_name,
                     'test')

    text_dict = load_obj(text_dict_path, fast=True)
    ds = load_obj(ds_path, fast=True)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    truth_dict = {0: False,
                  1: True}
    path_pairs = os.path.join(dest_folder, 'pairs.jsonl')
    path_truth = os.path.join(dest_folder, 'truth.jsonl')
    with jsonlines.open(path_pairs, mode='w') as pair_writer:
        with jsonlines.open(path_truth, mode='w') as truth_writer:
            for problem, truth in zip(ds['problem_list'], ds['truth_list']):
                pairs = {"id": problem['prob_id'],
                         "fandoms": problem['topics'],
                         "pair": [text_dict[problem['text_ids'][0]],
                                  text_dict[problem['text_ids'][1]]]}
                truth_format = {"id": problem['prob_id'],
                                "same": truth_dict[truth],
                                "authors": problem['authors']}
#                 truth = truth
                pair_writer.write(pairs)
                truth_writer.write(truth_format)



# ============================== write small balanced PAN 20 dataset ==========

def write_small_balanced_set(jsonl_text, jsonl_truth, file_name, size, lim):
    print('========= write_small_balanced_set')
    print('size: ', size)
    reader_text = jsonlines.open(jsonl_text)
    reader_truth = jsonlines.open(jsonl_truth)
    true_instances = []
    false_instances = []
    true_count = 0
    false_count = 0
    for i in range(lim):
        if (true_count < size or false_count < size):
            print('for loop')
            try:
                obj_text = reader_text.read()
                obj_truth = reader_truth.read()
            except EOFError:
                print('EOF en línea: ', i)
                break
            except InvalidLineError:
                print('línea inválida en: ', i)
                continue

            print(obj_truth['same'])
            if (true_count < size) and (obj_truth['same'] is True):
                true_instances.append((obj_text, obj_truth))
                true_count += 1

            if (false_count < size) and (obj_truth['same'] is False):
                false_instances.append((obj_text, obj_truth))
                false_count += 1

            print('true_count: ', true_count)
            print('false_count: ', false_count)

    print('after for loop')
    print('true_count: ', true_count)
    print('false_count: ', false_count)
    reader_text.close()
    reader_truth.close()

    writer_text = jsonlines.open(file_name + '_text.jsonl', mode='w')
    writer_truth = jsonlines.open(file_name + '_truth.jsonl', mode='w')
    for count in range(size):
        try:
            writer_text.write(true_instances[count][0])
            writer_truth.write(true_instances[count][1])
        except IndexError:
            print('Solo se encontraron ', count, ' instancias true')

        try:
            writer_text.write(false_instances[count][0])
            writer_truth.write(false_instances[count][1])
        except IndexError:
            print('Solo se encontraron ', count, ' instancias false')

    writer_text.close()
    writer_truth.close()


# ============================================================
# ============================== Test functions ==========

def test_wrapper_TextDataset():
    dataset_name = '22-train'
    where_to_use = 'acerL20'
    lim = None
    dataset, param_read = wrapper_TextDataset(dataset_name, lim, where_to_use)
    dataset_to_texts_dict(dataset,sys.stdout,"../PAN_datasets")
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of elements: {len(dataset)}')
    print('Problem_list_names:')
    names = [problem['prob_id'][0] for problem in dataset.problem_list]
    print(names)
    print('Truth_list:')
    print(dataset.truth_list)

    problem, y = dataset[0]
    print(f'Problem: {problem}')
    print(f'Label: {y}')

    print(f'param_read: {param_read}')


def test_fit_dict():
    path = ('../data/PAN20_graphs/20-small-bal_2_no-punct/'
            'edge_dict_no-punct_short')
    doc_dict = load_obj(path, fast=True)
    path = ('../data/PAN20_text_split/20-small-bal_2/ds_list_train')
    ds_train = load_obj(path, fast=True)
    path = ('../data/PAN20_text_split/20-small-bal_2/ds_list_val')
    ds_val = load_obj(path, fast=True)
    list_of_ds = [ds_train, ds_val]
    list_of_dict = fit_dict(doc_dict, list_of_ds)
    print(len(list_of_dict))


def verify_splits_custom():
    time_stamp = time_string(short=True)

    dest_folder_1 = os.path.join('../data/PAN20_text_split',
                                 '20-small-train')
#     ds_list_1_path = os.path.join(dest_folder_1, 'ds_list_test')
    ds_list_1_path = os.path.join(dest_folder_1, 'clean/ds_list_clean')
    ds_list_1 = load_obj(ds_list_1_path, fast=True)
    stats_dict_clean_path = \
        os.path.join(dest_folder_1, 'clean/stats_dict_clean')
    stats_dict_clean_1 = load_obj(stats_dict_clean_path, fast=True)
    df_stats_1 = stats_dict_clean_1['df_stats']

    dest_folder_2 = os.path.join('../data/PAN20_text_split',
                                 '20-large-train')
#     ds_list_2_path = os.path.join(dest_folder_2, 'ds_list_train')
    ds_list_2_path = os.path.join(dest_folder_2, 'clean/ds_list_clean')
    ds_list_2 = load_obj(ds_list_2_path, fast=True)
    stats_dict_clean_path = \
        os.path.join(dest_folder_2, 'clean/stats_dict_clean')
    stats_dict_clean_2 = load_obj(stats_dict_clean_path, fast=True)
    df_stats_2 = stats_dict_clean_2['df_stats']

#     ds_list_test_path = os.path.join(dest_folder, 'ds_list_test')
#     ds_list_test = load_obj(ds_list_test_path, fast=True)

    print('Custom verify_splits...')
    log_name = os.path.join(dest_folder_1,
                            'Custom_verify_splits' + time_stamp + '.txt')
    print('log save in: ' + log_name)
    f = open(log_name, 'w+')
    print('ds_list_1_path:', ds_list_1_path, file=f)
    print('ds_list_2_path:', ds_list_2_path, file=f)
#     print('ds_list_test_path:', ds_list_test_path, file=f)

    print('Train:', file=f)
    print(len(ds_list_1['problem_list']),
          '. Truth ', ds_list_1['truth_count'], file=f)
    print('Val:', file=f)
    print(len(ds_list_2['problem_list']),
          '. Truth ', ds_list_2['truth_count'], file=f)

    authors_1, topics_1 = \
        unique_authors_topics(ds_list_1, df_stats_1)
    authors_2, topics_2 = \
        unique_authors_topics(ds_list_2, df_stats_2)

    print('1 and 2', file=f)
    report_intersections(ds_list_1, ds_list_2,
                         authors_1, authors_2,
                         topics_1, topics_2, f)


# ============================================================

def main():
    dataset_pipeline()
#     verify_splits_custom()
    #compare_datasets()
#     test_fit_dict()
#     test_wrapper_TextDataset()
#    get_pairs()


if __name__ == "__main__":
    main()
