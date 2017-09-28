import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from add_utils import split_train_validation_data, process_data, data_mean_var, data_mean_var_dict,\
    data_generator, normalize_block, build_data_tensor
from model import housing_model
import tensorflow as tf
# from tqdm import tqdm
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

train_validation_data_block = process_data(train_file_name)
train_data_block, valid_data_block = split_train_validation_data(valid_ratio, train_validation_data_block)
print("train_data's number %i and valid_data's number %i" % (len(train_data_block), len(valid_data_block)))

train_data_mean, train_data_var = data_mean_var_dict(train_data_block.data)

# print("train data mean and var", train_data_mean, train_data_var)
train_label_mean, train_label_var = data_mean_var(train_data_block.info)
print("train label mean and var %f %f#" %( train_label_mean, train_label_var))


print("normalizing data")
norm_train_data_block = normalize_block(train_data_block, train_data_mean, train_data_var, train_label_mean, train_label_var)
norm_valid_data_block = normalize_block(valid_data_block, train_data_mean, train_data_var, train_label_mean, train_label_var)
print("normalizing label")

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

# TODO model needs to be changed either.
model = housing_model("housing", True, True)
data = build_data_tensor(train_validation_data_block.data)
# data_block is a dictionary, too.
label = tf.placeholder(tf.float32, shape=(None, 1), name="label")
dropout_ratio = tf.placeholder(tf.float32, shape=None, name="dr")
# call before initialize
y, model_loss, model_opt = model(data, label, lr, dropout_ratio = dropout_ratio,l2_reg=l2_reg)

t_gen = data_generator(norm_train_data_block, batch_size, True)
v_gen = data_generator(norm_valid_data_block,  batch_size, True)

train_loss_list = []
train_real_loss_list = []
valid_loss_list = []
valid_real_loss_list = []
with tf.Session(config=config) as sess:
    model.initialize(sess)
    # for i in tqdm(range(num_epoch), ncols=50):
    for i in range(num_epoch):
        done = False
        while not done:
            sdata, slabel, done, sidx = next(t_gen)
            feed_dict = {data[k]:sdata[k] for k in sdata.keys()}
            feed_dict.update({
                label: slabel,
                dropout_ratio: dr
            })
            price, loss, _ = sess.run([y, model_loss, model_opt], feed_dict=feed_dict)
            train_loss_list.append(loss)
            train_real_loss_list += list(np.abs((price*train_label_var+train_label_mean)-train_data_block.info[sidx]).flatten())
            #print(np.concatenate(price))
        print("train loss", np.mean(train_loss_list))
        print("train real loss", np.mean(train_real_loss_list))
        print(len(train_real_loss_list))
        done = False
        while not done:
            sdata, slabel, done, sidx = next(v_gen)
            feed_dict = {data[k]: sdata[k] for k in sdata.keys()}
            feed_dict.update({
                label: slabel,
                dropout_ratio: 1.
            })
            price, loss = sess.run([y, model_loss], feed_dict=feed_dict)
            valid_loss_list.append(loss)
            valid_real_loss_list += list(np.abs((price*train_label_var+train_label_mean)-valid_data_block.info[sidx]).flatten())
            #print(price) 
        print("valid loss", np.mean(valid_loss_list))
        print("valid real loss", np.mean(valid_real_loss_list))
        
        train_loss_list = []
        valid_loss_list = []
        train_real_loss_list = []
        valid_real_loss_list = []

    
