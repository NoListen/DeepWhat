import csv
import pandas as pd
import numpy as np
import tensorflow as tf

# 1. PROCESS 2. SPLIT 3. BATCH
# PROCESS THE INPUT AS FLEXIBLE INFO.
# return a class and deal with the data using a dictionary.


# had better not to modify them.
class Data_Block(object):
    # data is an dictionary. More flexible.
    # TODO store cat_dict, cat_n (ignore them temporally)
    def __init__(self, data, info, data_num):
        self._data = data
        # info can be labels or indexes.
        self._info = info
        self._length = data_num

    def __len__(self):
        return self._length

    @property
    def data(self):
        return self._data

    @property
    def info(self):
        return self._info


id_type = 'id'
label_type = 'price'
num_type = ['nbeds', 'rating', 'ncars', 'nbaths', 'land_size', 'house_size','year']
cat_type = [ 'suburb', 'result','property_type']
month_dict = {'November':11, 'August':8, 'July':7, 'December':12, 'January':1, 'February':2, 'October':10, 'June':6,
     'April':4, 'May':5, 'September':9, 'March':3}

# INFO AND DEFAULT


# by default, the data not listed as one group is categoritized to one group.
data_group = {"loc":["suburb"]}
# complement data_group
type_included = []
for g in data_group.values():
    for t in g:
        type_included.append(t)
g = []
for t in num_type + cat_type:
    if t not in type_included:
        g.append(t)
if len(g) > 0:
    data_group["default"] = g

# f is a function, d is a dictionary
# process one by one and return one dict.
# def function_dict(f, d):
#     res = {}
#     for k, v in d.items():
#         res[k] = f(v)

def read_file(file):
    data = pd.read_csv(file)
    return data

def categorize(column):
    cat = list(set(column))
    cat_dict = {}
    n = len(cat)
    for i in range(len(cat)):
        cat_dict[cat[i]] = i
    return cat_dict, n

def vectorize_with_dict(column, dict):
    n_data = len(column)
    data = np.zeros((n_data, 1), dtype="float32")
    for i in range(n_data):
        data[i] = dict[column[i]]
    return data

def one_hot(column, cat_dict, n):
    cat_count = {}
    n_data = len(column)
    data = np.zeros((n_data, n), dtype="float32")
    # slow operation
    for i in range(n_data):
        if cat_count.get(column[i], "") == "":
            cat_count[column[i]] = 1
        else:
            cat_count[column[i]] += 1
        data[i, cat_dict[column[i]]] = 1
    return data

def vectorize(column, array2d=True):
    data = column.values
    if array2d and len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    return data

def data_mean_var(data):
    print("compute mean and var for data with shape ", data.shape)
    mean = np.mean(data, axis=0)
    var = np.sqrt(np.var(data, axis=0)) + 1e-8
    return mean, var

def data_mean_var_dict(data_dict):
    mean = {}
    var = {}
    for k, v in data_dict.items():
        mean[k], var[k] = data_mean_var(v)
    return mean, var

def normalize(data, mean, var):
    return (data-mean)/var

def normalize_dict(data_dict, mean, var):
    res = {}
    assert(set(mean.keys()) == set(var.keys()) == set(data_dict.keys()))
    for k, v in data_dict.items():
        res[k] = (v - mean[k])/var[k]
    return res

# if it's test data, info_mean = 0, info_var = 1
def normalize_block(data_block, data_mean, data_var, info_mean, info_var):
    return Data_Block(normalize_dict(data_block.data, data_mean, data_var), normalize(data_block.info, info_mean, info_var),
                      len(data_block))

# Operate on dictionary
def group_data(data, dg):
    gd = {}
    for i, g in dg.items():
        d = []
        for t in g:
            # default order
            d.append(data[t])
        gd[i] = np.concatenate(d, axis=1)
        print("data group", i, "has shape", gd[i].shape)
    # dictionary seperated by the data_group
    return gd

# return a dict and then group the data flexibly
def process_data(file_name, train=True, dg=data_group):
    f = read_file(file_name)

    data_type = list(f.columns)
    print(data_type)
    data =  {}
    cat_dict_map = {}
    cat_n_map = {}
    data_label = None
    for t in data_type:
        if t in num_type:
            data[t] = vectorize(f[t])
        elif t in cat_type:
            # cat_dict describes each type's class number.
            # cat_n means how many class.
            cat_dict_map[t], cat_n_map[t] = categorize(f[t])
            data[t] = one_hot(f[t], cat_dict_map[t], cat_n_map[t])
        elif train and t == label_type:
            data_label = vectorize(f[t], array2d=True)
      #  elif t == "month":
      #      vectorize_with_dict(f[t], month_dict)
        else:
            print("IGNORE %s" % t)
    data = group_data(data, dg)
    # info is designed as a list while data is a dicitionary
    if train:
        return Data_Block(data, data_label, len(data_label))
    else:
        id_list = vectorize(f[id_type])
        return Data_Block(data, id_list, len(id_list))


# operate on a dictionary.
def split_dict_data(data, idxes):
    sub_data = {}
    for k, d in data.items():
        sub_data[k] = d[idxes]
    return sub_data

# return two data_block
def split_train_validation_data(valid_ratio, data_block):
    #  ensure the splitting is the same all the time
    np.random.seed(1234567)
    train_data_idxes = []
    valid_data_idxes = []
    # TODO len(data) is not avaliable now!
    for i in range(len(data_block)):
        if np.random.rand() > valid_ratio:
            train_data_idxes.append(i)
        else:
            valid_data_idxes.append(i)
    tidxes = np.array(train_data_idxes)
    vidxes = np.array(valid_data_idxes)

    train_data_block = Data_Block(split_dict_data(data_block.data, tidxes), data_block.info[tidxes], len(tidxes))
    valid_data_block = Data_Block(split_dict_data(data_block.data, vidxes), data_block.info[vidxes], len(vidxes))
    return train_data_block, valid_data_block

def sample_dict_data(data, sample_idxes):
    d = {}
    for k, v in data.items():
        d[k] = v[sample_idxes]
    return d

# also return a dictionary.
def data_generator(data_block, batch_size, idx_return = True, shuffle = True):
    n_data = len(data_block)
    print("generator can produce ", n_data)
    idxes = np.arange(n_data)
    assert(batch_size >= 1)
    #remain_to_generate =
    while True:
        if shuffle:
            np.random.shuffle(idxes)
        finish = False
        for i in range(0, n_data, batch_size):
            if i+batch_size >= n_data:
                finish = True
            sample_idxes = idxes[i:i+batch_size]
            if idx_return:
                yield sample_dict_data(data_block.data, sample_idxes), data_block.info[sample_idxes], finish, sample_idxes
            else:
                yield sample_dict_data(data_block.data, sample_idxes), data_block.info[sample_idxes], finish

def build_data_tensor(data):
    dt = {}
    for k, v in data.items():
        s = v.shape[1:]
        dt[k] = tf.placeholder(tf.float32, shape=(None,)+s, name=k)
    return dt