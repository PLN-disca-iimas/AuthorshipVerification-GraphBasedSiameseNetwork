# -*- coding: utf-8 -*-
"""Archivo con funciones útiles"""

from __future__ import print_function

import os
import sys
import subprocess
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
import time
import joblib
import torch

import pickle

_ext_dict = {'gzip': '.gz',
             'zlib': '.z'}

# Función para imprimir tiempo
def print_time(start_time, label=None, f=sys.stdout):
    exec_time = time.time() - start_time
    print(f'\n ===== {label}. Execute time was:', file=f)
    print('Minutes: {:.2f}'.format(exec_time/60), file=f)
    print('Seconds: {:.2f}'.format(exec_time), file=f)

#función para guardar en memoria lo encontrado
def save_obj(obj, name, use_joblib=None):
    if use_joblib == None:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        ext = _ext_dict[use_joblib[0]]
        with open(name + ext, 'wb') as f:
            joblib.dump(obj, f, compress=use_joblib)

# función para obtener objeto, folder y nombre
def load_obj(obj_path_all, fast=False):
    obj_path, extension = os.path.splitext(obj_path_all)
    extension = '.pkl' if extension == '' else extension
    with open(obj_path + extension, 'rb') as f:
        if fast:
            return joblib.load(f)
        else:
            folder_name, obj_name = os.path.split(obj_path)
            folder = folder_name + '/'
            obj = joblib.load(f)
            return obj, folder, obj_name

def time_string(short=False):
    ti = time.localtime()
    if short:
        t='{:02}{:02}{:02}{:02}'.format(ti.tm_mon,
                                        ti.tm_mday,
                                        ti.tm_hour,
                                        ti.tm_min)
    else:
        t='{}{:02}{:02}{:02}{:02}{:02}'.format(ti.tm_year,
                                               ti.tm_mon,
                                               ti.tm_mday,
                                               ti.tm_hour,
                                               ti.tm_min,
                                               ti.tm_sec)

    return t

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its
    contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def my_print(a, file=sys.stdout):
    if (isinstance(a, list) or isinstance(a, set)):
        print(*a, sep='\n', file=file)
        print('\n', file=file)

    if isinstance(a, dict):
        print(*zip(a.keys(), a.values()), sep='\n', file=file)
        print('\n', file=file)

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def get_gpu_memory_device():
    if torch.cuda.is_available():
        gpu_usage = get_gpu_memory_map()[torch.cuda.current_device()]
        return gpu_usage
    else:
        return 0

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(),
                                      model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
