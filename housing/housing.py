import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from utils import split_train_validation_data, process_data, data_mean_var, normalize, data_generator, data_mean_scale
from model import housing_model
import tensorflow as tf
from tqdm import tqdm
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
train_validation_data, train_validation_label, tv_cat_dict_map, tv_cat_n_map = process_data(train_file_name)
data_shape = train_validation_data.shape[1:]
train_data, train_label, valid_data, valid_label = split_train_validation_data(valid_ratio, train_validation_data, train_validation_label)
print("train_data's number %i and valid_data's number %i" % (len(train_data), len(valid_data)))
train_data_mean, train_data_var = data_mean_var(train_data)
#train_label_mean, train_label_var = data_mean_scale(train_label)
train_label_mean, train_label_var = data_mean_var(train_label)
print("#%f %f#" %( train_label_mean, train_label_var))


print(np.max(train_label), np.min(train_label))
print("normalizing data")
norm_train_data = normalize(train_data, train_data_mean, train_data_var)
norm_valid_data = normalize(valid_data, train_data_mean, train_data_var)
print("normalizing label")
norm_train_label = normalize(train_label, train_label_mean, train_label_var)
norm_valid_label = normalize(valid_label, train_label_mean, train_label_var)
#norm_train_data = normalize(_data, train_data_mean. train_data_var)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True


model = housing_model("housing", True, True)
data = tf.placeholder(tf.float32, shape=(None,)+data_shape, name="data")
label = tf.placeholder(tf.float32, shape=(None, 1), name="label")
dropout_ratio = tf.placeholder(tf.float32, shape=None, name="dr")
# call before initialize
y, model_loss, model_opt = model(data, label, lr, dropout_ratio = dropout_ratio,l2_reg=l2_reg)

t_gen = data_generator(norm_train_data, norm_train_label, batch_size, True)
v_gen = data_generator(norm_valid_data, norm_valid_label, batch_size, True)

train_loss_list = []
train_real_loss_list = []
valid_loss_list = []
valid_real_loss_list = []
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
        done = False
        while not done:
            sdata, slabel, done, sidx = next(v_gen)
            price, loss = sess.run([y, model_loss], feed_dict={data:sdata, label:slabel, dropout_ratio:1.})
            valid_loss_list.append(loss)
            valid_real_loss_list += list(np.abs((price*train_label_var+train_label_mean)-valid_label[sidx]).flatten())
            #print(price) 
        print("valid loss", np.mean(valid_loss_list))
        print("valid real loss", np.mean(valid_real_loss_list))
        
        train_loss_list = []
        valid_loss_list = []
        train_real_loss_list = []
        valid_real_loss_list = []

    
