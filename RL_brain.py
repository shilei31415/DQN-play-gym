import numpy as np
import tensorflow as tf
import time

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 shape_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=16,
                 e_greedy_increment=0.01,
                 output_graph=False
                 ):
        self.n_actions = n_actions
        self.shape_features = shape_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        print(e_greedy_increment)
        self.epsilon_increment = e_greedy_increment
        if e_greedy_increment is not None:
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.step_store = {"s": None, "a": None, "r": None, "s_": None}
        self.memory=[]

        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.allow_soft_placement = True

        self.sess = tf.Session(config=config)

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [None, self.shape_features[0], self.shape_features[1], self.shape_features[2]], name="s")
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name = "Q_target")
        with tf.variable_scope("eval_net"):
            # c_names(collections_names) are the collections to store variables
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [5, 5, 3, 32], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [32], initializer=b_initializer, collections=c_names)
                conv1 = tf.nn.conv2d(self.s, w1, strides=[1, 1, 1, 1], padding="SAME")
                conv1 = tf.nn.leaky_relu(conv1+b1, alpha=0.1)
                l1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [5, 5, 32, 16], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [16], initializer=b_initializer, collections=c_names)
                conv2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding="SAME")
                conv2 = tf.nn.leaky_relu(conv2 + b2, alpha=0.1)
                l2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [5, 5, 16, 4], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [4], initializer=b_initializer, collections=c_names)
                conv3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding="SAME")
                conv3 = tf.nn.leaky_relu(conv3 + b3, alpha=0.1)
                l3 = tf.nn.max_pool(conv3, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l4'):
                shape = l3.get_shape().as_list()
                node = shape[1]*shape[2]*shape[3]
                reshaped = tf.reshape(l3, [-1, node])
                w4 = tf.get_variable("w4", [node, node/4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable("b4", [node/4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.leaky_relu(tf.matmul(reshaped, w4)+b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable("w5", [node/4, node / 16], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable("b5", [node / 16], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.leaky_relu(tf.matmul(l4, w5) + b5)

            with tf.variable_scope('l6'):
                w6 = tf.get_variable("w6", [node/16, self.n_actions], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable("b6", [self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.nn.leaky_relu(tf.matmul(l5, w6) + b6)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.shape_features[0], self.shape_features[1], self.shape_features[2]], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [5, 5, 3, 32], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [32], initializer=b_initializer, collections=c_names)
                conv1 = tf.nn.conv2d(self.s_, w1, strides=[1, 1, 1, 1], padding="SAME")
                conv1 = tf.nn.leaky_relu(conv1 + b1, alpha=0.1)
                l1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [5, 5, 32, 16], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [16], initializer=b_initializer, collections=c_names)
                conv2 = tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding="SAME")
                conv2 = tf.nn.leaky_relu(conv2 + b2, alpha=0.1)
                l2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [5, 5, 16, 4], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [4], initializer=b_initializer, collections=c_names)
                conv3 = tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding="SAME")
                conv3 = tf.nn.leaky_relu(conv3 + b3, alpha=0.1)
                l3 = tf.nn.max_pool(conv3, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

            with tf.variable_scope('l4'):
                shape = l3.get_shape().as_list()
                node = shape[1] * shape[2] * shape[3]
                reshaped = tf.reshape(l3, [-1, node])
                w4 = tf.get_variable("w4", [node, node / 4], initializer=w_initializer, collections=c_names)
                b4 = tf.get_variable("b4", [node / 4], initializer=b_initializer, collections=c_names)
                l4 = tf.nn.leaky_relu(tf.matmul(reshaped, w4) + b4)

            with tf.variable_scope('l5'):
                w5 = tf.get_variable("w5", [node / 4, node / 16], initializer=w_initializer, collections=c_names)
                b5 = tf.get_variable("b5", [node / 16], initializer=b_initializer, collections=c_names)
                l5 = tf.nn.leaky_relu(tf.matmul(l4, w5) + b5)

            with tf.variable_scope('l6'):
                w6 = tf.get_variable("w6", [node / 16, self.n_actions], initializer=w_initializer, collections=c_names)
                b6 = tf.get_variable("b6", [self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.nn.leaky_relu(tf.matmul(l5, w6) + b6)

    def store_transition(self, s, a, r, s_):
        # 如果不存在memory_counter,则定义一个
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        self.step_store["s"] = s
        self.step_store["a"] = a
        self.step_store["r"] = r
        self.step_store["s_"] = s_

        if len(self.memory)<self.memory_size:
            # print(len(self.memory))
            self.memory.append(self.step_store)
        else:
            index = self.memory_counter % self.memory_size
            self.memory[index] = self.step_store

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            print(actions_value)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    # 利用新网络玩
    def choose_action_eval(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        print("eval:", actions_value)
        action = np.argmax(actions_value)
        return action

    #利用旧网络玩
    def choose_action_target(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_next, feed_dict={self.s_: observation})
        print("next:", actions_value)
        action = np.argmax(actions_value)
        return action

    def learn(self):
        # if self.learn_step_counter % self.replace_target_iter ==0:
        #     self.sess.run(self.replace_target_op)
        #     print("\ntarget_params_replaced\n")
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # print(sample_index)
        # print(len(self.memory))
        batch_memory_s = []
        batch_memory_s_ = []
        eval_act_index = []
        reward = []
        for index in sample_index:
            batch_memory_s.append(self.memory[index]["s"])
            batch_memory_s_.append(self.memory[index]["s_"])
            eval_act_index.append(self.memory[index]["a"])
            reward.append(self.memory[index]["r"])

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory_s_,  # fixed params
                self.s: batch_memory_s     # newest params
            })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory_s,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        if self.epsilon < self.epsilon_max:
            self.epsilon = self.epsilon + self.epsilon_increment
        else:
            self.epsilon = self.epsilon_max

        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
        # plt.draw()
        # plt.pause(0.01)
