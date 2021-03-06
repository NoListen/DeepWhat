# reinforce is one of the most simple PG algorithm
from agent import Agent
from utils import traj_segment_generator, discount
from collections import deque
import numpy as np
import tensorflow as tf

def reinforce(env, sess, obs_processor, lr, gamma, action_dict, **kargs):
    pi = Agent(sess, lr, action_dict, **kargs)
    sess.run(tf.global_variables_initializer())
    ep_gen = traj_segment_generator(pi, env, obs_processor, stochastic=True)
    ep_test_gen = traj_segment_generator(pi, env, obs_processor, stochastic=False)

    ep_rets = deque(maxlen=100)
    ep_steps = deque(maxlen=100)
    episodes = 0

    test_time = 20
    t = 0

    # TODO set terminal state.
    while True:
        # generate one episode
        ep = ep_gen.__next__()
        ep_rets.append(ep["ep_ret"])
        ep_steps.append(ep["ep_steps"])
        ep["ret"] = discount(ep["rew"], gamma)
        #ep["ret"] -= np.mean(ep["ret"])
        #ep["ret"] /= np.std(ep["ret"])
        episodes += 1

        pi.train(ep)

        t += 1
        if t % test_time == 0:
            ep_test = ep_test_gen.__next__()
            print("Test result %.2f" % np.sum(ep_test["rew"]))

        print("ep%i ret:%.2f steps:%i average_ret:%.1f average_steps:%.1f" %
              (episodes, ep["ep_ret"], ep["ep_steps"], np.mean(ep_rets), np.mean(ep_steps)))
