import csv
import pandas as pd
import numpy as np

id_type = 'id'
label_type = 'price'
num_type = ['nbeds', 'rating', 'ncars', 'nbaths', 'land_size', 'house_size','year']#, 'day']
#num_type = ['nbeds', 'nvisits', 'rating', 'ncars', 'nbaths', 'land_size', 'house_size','year', 'day']
#cat_type = ['suburb', 'result', 'agent', 'property_type']
cat_type = [ 'suburb', 'result','property_type']
#np.random.seed(1234567)
#file_name = "HousePriceData_TrainingSet.csv"
month_dict = {'November':11, 'August':8, 'July':7, 'December':12, 'January':1, 'February':2, 'October':10, 'June':6,
     'April':4, 'May':5, 'September':9, 'March':3}

def read_file(file):
    data = pd.read_csv(file)
    return data


# TODO add a clas others????
def categorize(column):
    cat = list(set(column))
    cat_dict = {}
    n = len(cat)
    for i in range(len(cat)):
        cat_dict[cat[i]] = i
    #print(cat_dict)
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
    print(cat_count)
    #v = list(cat_count.values())
    #print( cat_dict.keys(), np.max(v), np.min(v), np.mean(v))
    return data

def vectorize(column, array2d=True):
    data = column.values
    if array2d and len(data.shape) == 1:
        data = np.expand_dims(data, axis=1)
    return data

def data_mean_scale(data):
    mean = np.mean(data, axis=0)
    scale = (np.max(data, axis=0) - np.min(data, axis=0) + 1e-8)
    return mean, scale

def data_mean_var(data):
    print("compute mean and var for data with shape ", data.shape)
    mean = np.mean(data, axis=0)
    # TODO var can be zero because of agent (0)
    var = np.sqrt(np.var(data, axis=0)) + 1e-8
    #print(np.argsort(var)[:10], np.sort(var)[:10])
    #print("get mean and var with shape ", mean.shape, var.shape)
    return mean, var

def normalize(data, mean, var):
    return (data-mean)/var
    

def process_data(file_name, train=True):
    f = read_file(file_name)
    data_type = list(f.columns)
    print(data_type)
    data_shape = {}
#suburb = f["suburb"]
#print (suburb.values.shape)
    data_list = []
    cat_dict_map = {}
    cat_n_map = {}
    data_label = None
    for t in data_type:
        if t in num_type:
            data_list.append(vectorize(f[t]))
        elif t in cat_type:
            cat_dict_map[t], cat_n_map[t] = categorize(f[t])
            data_list.append(one_hot(f[t], cat_dict_map[t], cat_n_map[t]))
        elif train and t == label_type:
            # no need to concatenate
            data_label = vectorize(f[t], array2d=True)
            #data_label = vectorize(f[t], array2d=False)
      #  elif t == "month":
      #      vectorize_with_dict(f[t], month_dict)
        else:
            # TODOD conside the date sold
            print("IGNORE %s" % t)
    print("data processed completed")
    #for d in data_list:
    #    print d.shape
    data = np.concatenate(data_list, axis=1)
    print("the data's shape is", data.shape)
    if train:
        return data, data_label, cat_dict_map, cat_n_map
    else:
        return data, vectorize(f[id_type]),cat_dict_map, cat_n_map



def split_train_validation_data(valid_ratio, data, label):
    #  ensure the splitting is the same all the time
    np.random.seed(1234567)
    train_data_idxes = []
    valid_data_idxes = []
    for i in range(len(data)):
        if np.random.rand() > valid_ratio:
            train_data_idxes.append(i)
        else:
            valid_data_idxes.append(i)
    tidxes = np.array(train_data_idxes)
    vidxes = np.array(valid_data_idxes)
    return data[tidxes], label[tidxes],data[vidxes],label[vidxes]

def data_generator(data, label, batch_size, idx_return = False, train =True, shuffle = True):
    n_data = len(data)
    print("generator can produce ", n_data)
    assert(batch_size >= 1)
    idxes = np.arange(n_data)
    #remain_to_generate = 
    while True:
        if shuffle:
            np.random.shuffle(idxes)
        finish = False
        for i in range(0, n_data, batch_size):
            if i+batch_size >= n_data:
                finish = True
            sample_idxes = idxes[i:i+batch_size]
            if train:
                if idx_return:
                    yield data[sample_idxes], label[sample_idxes], finish, sample_idxes
                else:
                    yield data[sample_idxes], label[sample_idxes], finish
            else:
                if idx_return:
                    yield data[sample_idxes],  finish, sample_idxes
                else:
                    yield data[sample_idxes],  finish
