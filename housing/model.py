import tensorflow as tf
import tensorflow.contrib as tc

class housing_model(object):
    def __init__(self, name, layer_norm = False, dropout=False):
        self.name = name
        self.layer_norm = layer_norm
        self.dropout = dropout

    def __call__(self, data, label, lr, dropout_ratio, l2_reg=0.):
        self.y = self.build_model(dropout_ratio, **data)
        self.build_optimizer(self.y, label, lr, l2_reg)
        return self.y, self.loss, self.opt

    def build_model(self, dropout_ratio, loc, default):
        with tf.variable_scope(self.name) as scope:
            x2 = loc
            if self.layer_norm:
                x2 = tc.layers.layer_norm(x2, center = True, scale=True)
            #x2 = tf.nn.relu(x2)
            x2 = tf.layers.dense(x2, 64)

            x = tf.concat([default, x2], axis=1)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center = True, scale=True)
            # TODO eliminate this line to see the effect.
            x = tf.nn.relu(x)
            # 1
            #x = tf.layers.dense(x, 640)
            #if self.layer_norm:
            #    x = tc.layers.layer_norm(x, center = True, scale=True)
            #x = tf.nn.relu(x)
            # 2
            x = tf.layers.dense(x,128)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center = True, scale=True)
            x = tf.nn.relu(x)
            # 3
            # x = tf.layers.dense(x, 192)
            # if self.layer_norm:
                # x = tc.layers.layer_norm(x, center = True, scale=True)
            # x = tf.nn.relu(x)

            x = tf.layers.dense(x, 96)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center = True, scale=True)
            x = tf.nn.relu(x)
            if self.dropout:
                x = tf.nn.dropout(x, dropout_ratio,seed=514229)
            y = tf.layers.dense(x, 1)
            return y

    def build_optimizer(self, y, label, lr, l2_reg):
        with tf.variable_scope("%s_optimizer" % self.name):
            #self.loss = tf.reduce_mean(tf.square(y - label))
            self.loss = tf.reduce_mean(tf.abs(y - label))

            if l2_reg:
                reg_vars = [var for var in self.trainable_vars if
                     'kernel' in var.name and 'output' not in var.name]
                for var in reg_vars:
                    print('regularizing: {}'.format(var.name))
                print('  applying l2 regularization with {}'.format(l2_reg))
                reg = tc.layers.apply_regularization(
                    tc.layers.l2_regularizer(l2_reg),
                    weights_list=reg_vars)
                self.loss += reg

            # TODO specify beta1 beta2 epsilon          reg_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def initialize(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
 
