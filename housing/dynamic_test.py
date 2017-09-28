# coding: utf-8
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from add_utils import split_train_validation_data, process_data, data_mean_var, data_mean_var_dict,\
    data_generator, normalize_block, build_data_tensor
from model import housing_model
import tensorflow as tf
import json

config = json.load(open('config.json', 'r'))


train_file_name =  config["train_file_name"]
valid_ratio = config["valid_ratio"]
dr = config["dr"] # dropout rate
l2_reg = config["l2_reg"]
lr = config["lr"]
test_file_name = config["test_file_name"]
num_epoch = config["num_epoch"]
batch_size = config["batch_size"]

train_data_block = process_data(train_file_name)
test_data_block = process_data(test_file_name, False)

# data_shape = train_data.shape[1:]
train_data_mean, train_data_var = data_mean_var_dict(train_data_block.data)
#train_label_mean, train_label_var = data_mean_scale(train_label)
train_label_mean, train_label_var = data_mean_var(train_data_block.info)
print("train label mean %f and var %f#" %( train_label_mean, train_label_var))


print("normalizing data block")
norm_train_data_block = normalize_block(train_data_block, train_data_mean, train_data_var, train_label_mean, train_label_var)
norm_test_data_block = normalize_block(test_data_block, train_data_mean, train_data_var, 0, 1)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

model = housing_model("housing", True, True)
data = build_data_tensor(train_data_block.data)
label = tf.placeholder(tf.float32, shape=(None, 1), name="label")
dropout_ratio = tf.placeholder(tf.float32, shape=None, name="dr")
# call before initialize
y, model_loss, model_opt = model(data, label, lr, dropout_ratio = dropout_ratio,l2_reg=l2_reg)

t_gen = data_generator(norm_train_data_block, batch_size, True)
e_gen = data_generator(norm_test_data_block, batch_size, idx_return=False, shuffle = False)

train_loss_list = []
train_real_loss_list = []
test_prices = []
test_id = []
with tf.Session(config=config) as sess:
    model.initialize(sess)
    for i in range(num_epoch):
        done = False
        while not done:
            sdata, slabel, done, sidx = next(t_gen)
            feed_dict = {data[k]: sdata[k] for k in sdata.keys()}
            feed_dict.update({
                label: slabel,
                dropout_ratio: dr
            })
            price, loss, _ = sess.run([y, model_loss, model_opt], feed_dict=feed_dict)
            train_loss_list.append(loss)
            train_real_loss_list += list(np.abs((price*train_label_var+train_label_mean)-train_data_block.info[sidx]).flatten())
        print("train loss", np.mean(train_loss_list))
        print("train real loss", np.mean(train_real_loss_list))

        train_loss_list = []
        train_real_loss_list = []
    done = False
    while not done:
        sdata,  sid, done = next(e_gen)
        feed_dict = {data[k]: sdata[k] for k in sdata.keys()}
        feed_dict.update({
            dropout_ratio: 1.
        })
        price = sess.run(y, feed_dict=feed_dict)
        #print(price)
        #test_prices += list(np.abs((price*train_label_var+train_label_mean)-test_label[sidx]).flatten())
        test_id += list(sid.flatten())
        test_prices += list((price*train_label_var+train_label_mean).flatten())
    #print(np.mean(test_prices), len(test_prices))
    #print(len(test_prices), len(test_id))
    test_result = zip(list(test_id), list(test_prices))
    import csv
    csvfile = open('result.csv', 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['id',  'price'])
    writer.writerows(test_result)
    csvfile.close()
