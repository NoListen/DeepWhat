# coding: utf-8
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import split_train_validation_data, process_data, data_mean_var, normalize, data_generator, data_mean_scale
from model import housing_model
import tensorflow as tf
from tqdm import tqdm

train_file_name =  "HousePriceData_TrainingSet.csv"
valid_ratio = 0.1
dr = 1. # dropout rate
l2_reg = 0.015
lr = 0.001
test_file_name = "test_noprice.csv"
num_epoch = 100
batch_size = 256
train_data, train_label, t_cat_dict_map, t_cat_n_map = process_data(train_file_name)
test_data,  test_id, e_cat_dict_map, e_cat_n_map = process_data(test_file_name, False)

data_shape = train_data.shape[1:]
train_data_mean, train_data_var = data_mean_var(train_data)
#train_label_mean, train_label_var = data_mean_scale(train_label)
train_label_mean, train_label_var = data_mean_var(train_label)
print("#%f %f#" %( train_label_mean, train_label_var))


print("normalizing data")
norm_train_data = normalize(train_data, train_data_mean, train_data_var)
norm_test_data = normalize(test_data, train_data_mean, train_data_var)
print("normalizing label")
norm_train_label = normalize(train_label, train_label_mean, train_label_var)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True


model = housing_model("housing", True, True)
data = tf.placeholder(tf.float32, shape=(None,)+data_shape, name="data")
label = tf.placeholder(tf.float32, shape=(None, 1), name="label")
dropout_ratio = tf.placeholder(tf.float32, shape=None, name="dr")
# call before initialize
y, model_loss, model_opt = model(data, label, lr, dropout_ratio = dropout_ratio,l2_reg=l2_reg)

t_gen = data_generator(norm_train_data, norm_train_label, batch_size, True)
e_gen = data_generator(norm_test_data, None, batch_size, True, train=False, shuffle = False)

train_loss_list = []
train_real_loss_list = []
test_prices = []
with tf.Session(config=config) as sess:
    model.initialize(sess)
    for i in tqdm(range(num_epoch), ncols=50):
        done = False
        while not done:
            sdata, slabel, done, sidx = next(t_gen)
            price, loss, _ = sess.run([y, model_loss, model_opt], feed_dict={data:sdata, label:slabel, dropout_ratio:dr})
            train_loss_list.append(loss)
            train_real_loss_list += list(np.abs((price*train_label_var+train_label_mean)-train_label[sidx]).flatten())
            #print(np.concatenate(price))
        print("train loss", np.mean(train_loss_list))
        print("train real loss", np.mean(train_real_loss_list))
        print(len(train_real_loss_list))
        
        train_loss_list = []
        train_real_loss_list = []
    done = False
    while not done:
        sdata,  done, sidx = next(e_gen)
        price = sess.run(y, feed_dict={data:sdata, dropout_ratio:1.})
        print(price)
        test_prices += list((price*train_label_var+train_label_mean).flatten())
    print(len(test_prices), len(test_id))
    test_result = zip(list(test_id.flatten()), list(test_prices))
    import csv
    csvfile = open('result.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['id',  'price'])
    writer.writerows(test_result)
    csvfile.close()
