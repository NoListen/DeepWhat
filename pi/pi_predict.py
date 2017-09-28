from policy import LSTMPolicy
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
seq_length = 10
state_space = action_space = 10 # 10 numbers
lr = 0.001

str_pi_label = open("pi_15000", "r").read()
pi_label = [int(c) for c in str_pi_label]

one_hot_id_array = np.arange(seq_length+1)

def seq_genenerator(idx, length):
	res = np.zeros((length+1, state_space))
	res[one_hot_id_array, pi_label[idx:idx+length+1]] = 1
	return res.astype(np.float32)

# print seq_genenerator(0, 5)
# supervised setting. rl setting need entropy.
# even a3c maybe helpful

# the network has been setup.
x = tf.placeholder(tf.float32, [None, state_space], name="x")
label = tf.placeholder(tf.float32, [None, action_space], name="label")
policy = LSTMPolicy([state_space], action_space)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = policy.logits, labels = label))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)



with tf.Session() as sess:
	tf.initialize_all_variables().run()
	# num = 0
	max_num = 0
	idx = 0
	last_state = state_in = policy.get_initial_features()
	for step in tqdm(range(10000000), ncols = 70):
		data = seq_genenerator(idx, seq_length)
		# print data.dtype
		# print data[:seq_length]
		feed_dict = {
			policy.x: data[:seq_length],
			label: data[-seq_length:],
			policy.state_in[0]: state_in[0],
			policy.state_in[1]: state_in[1]
		}
		_, state_out, sample = sess.run([optimizer, policy.state_out, policy.sample], feed_dict=feed_dict)
		# print sample
		# print sample
		if step % 10001 == 0:
			test_idx = 0
			test_state_in = policy.get_initial_features()
			while True:
				test_data = seq_genenerator(test_idx, seq_length)
				test_feed_dict = {
					policy.x: test_data[:seq_length],
					policy.state_in[0]: test_state_in[0],
					policy.state_in[1]: test_state_in[1]
				}
				test_state_in, test_sample = sess.run([policy.state_out, policy.sample], feed_dict=test_feed_dict)

				if np.sum(np.abs(test_sample - test_data[-seq_length:]))>0:
					print "I can recite the pi up to %i number in test and %i number in training" % (test_idx, max_num)
					break
				else:
					test_idx += seq_length
			
		if np.sum(np.abs(sample - data[-seq_length:])) > 0:
			max_num = max(max_num, idx)
			if random.random() < 0.1:
				# print "reset"
				idx = 0
				state_in = policy.get_initial_features()
			continue

		idx += seq_length
		state_in = state_out


		# idx += seq_length