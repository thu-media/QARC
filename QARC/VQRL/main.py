import os
import logging
import numpy as np
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import innovation_env
import a3c
import time

INPUT_W = 64
INPUT_H = 36
INPUT_D = 3
INPUT_SEQ = 5

S_INFO = 7
S_LEN = 5  # take how many frames in the past
BITRATE_MIN = 0.3
BITRATE_MAX = 1.4
DELAY_GRADIENT_MAX = 0.1
VIDEO_BIT_RATE = [0.3, 0.5, 0.8, 1.1, 1.4]
A_DIM = len(VIDEO_BIT_RATE)
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3

NUM_AGENTS = 8
TRAIN_SEQ_LEN = 10  # take as a train batch
MODEL_SAVE_INTERVAL = 1000
MODEL_TEST_INTERVAL = 100
BUFFER_NORM_FACTOR = 5.0

BITRATE_PENALTY = 1
RTT_PENALTY = 0.5  # 1 sec rebuffering -> 3 Mbps
LOSS_PENALTY = 0.5
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent

RANDOM_SEED = int(time.time() % 1000)
RAND_RANGE = 1000

SUMMARY_DIR = './results'
LOG_FILE = './results/log'
#TEST_LOG_FOLDER = './test_results/'
SAVE_FOLDER = './fig_results/'
TEST_TRACES = './test_sim_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
#NN_MODEL = sys.argv[1]
#LOG_FILE = sys.argv[2]
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    logging.basicConfig(filename=LOG_FILE + '_central',
                        filemode='w',
                        level=logging.INFO)

    with tf.Session() as sess, open(LOG_FILE + '_test', 'wb') as test_log_file:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = a3c.build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(
            SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = 0

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_entropy = 0.0
            total_agents = 0.0

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, info = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0
                total_entropy += np.sum(info['entropy'])

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward / total_agents
            avg_td_loss = total_td_loss / total_batch_len
            avg_entropy = total_entropy / total_batch_len

            logging.info('Epoch: ' + str(epoch) +
                         ' TD_loss: ' + str(avg_td_loss) +
                         ' Avg_reward: ' + str(avg_reward) +
                         ' Avg_entropy: ' + str(avg_entropy))

            summary_str = sess.run(summary_ops, feed_dict={
                summary_vars[0]: avg_td_loss,
                summary_vars[1]: avg_reward,
                summary_vars[2]: avg_entropy
            })

            writer.add_summary(summary_str, epoch)
            writer.flush()
            # print epoch
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                os.system('mkdir log_tests')
                os.system('mkdir log_tests/' + str(epoch))
                os.system('python test.py ' + SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.ckpt log_tests/' + str(epoch))
                #test(sess)
                # os.system("python test.py " + str(epoch) + " " +
                #          SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt&")
                # testing(epoch,
                #        SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt",
                #        test_log_file)


def core_function(src, threshold):
    if src <= threshold:
        return src
    else:
        return 1.0


def test(actor, index):
    os.system('mkdir test_log/')
    cooked_files = os.listdir(TEST_TRACES)
    for cooked_file in cooked_files:
        net_env = innovation_env.Environment(RANDOM_SEED, cooked_file)
        log_path = 'test_log/' + cooked_file + '_log_' + str(index) + '.txt'
        log_file = open(log_path, 'wb')

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        #entropy_record = []
        last_rtt = -1
        last_vmaf = -1

        while True:
            _norm_bitrate = VIDEO_BIT_RATE[bit_rate]
            delay, loss, recv_bitrate, rtt, throughput, limbo_bytes_len = \
                net_env.get_video_chunk(bit_rate)

            if delay is None:
                log_file.write('\n')
                print 'Test done', cooked_file
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                #entropy_record = []
                break

            rtt = float(rtt) / float(1000)
            if last_rtt < 0:
                last_rtt = rtt
            _norm_send_bitrate = bit_rate / A_DIM
            _queuing_delay = abs(rtt - last_rtt)
            _norm_recv_bitrate = min(
                float(recv_bitrate) / delay / BUFFER_NORM_FACTOR, 1.0)

            time_stamp += delay  # in ms
            vmaf = net_env.get_vmaf(bit_rate)
            if last_vmaf < 0:
                last_vmaf = vmaf

            #_normalized_bitrate = (_norm_bitrate - BITRATE_MIN) / (BITRATE_MAX - BITRATE_MIN)
            _vmaf_reward = (vmaf / _norm_bitrate) * BITRATE_MIN
            reward = \
                1.0 * vmaf - \
                0.2 * _norm_send_bitrate - \
                1.0 / DELAY_GRADIENT_MAX * min(_queuing_delay, DELAY_GRADIENT_MAX) - \
                1.0 * abs(last_vmaf - vmaf)
            r_batch.append(reward)

            last_vmaf = vmaf
            last_rtt = rtt
            log_file.write(str(time_stamp) + '\t' +
                           str(_norm_bitrate) + '\t' +
                           str(recv_bitrate) + '\t' +
                           str(limbo_bytes_len) + '\t' +
                           str(rtt) + '\t' +
                           str(vmaf) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)
            state[0, -1] = _norm_send_bitrate  # last quality
            state[1, -1] = _norm_recv_bitrate  # kilo byte / ms
            state[2, -1] = _queuing_delay  # max:500ms
            state[3, -1] = float(loss)  # changed loss
            # test:add fft feature
            _fft = np.fft.fft(state[1])
            state[4] = _fft.real
            state[5] = _fft.imag
            state[6] = net_env.predict_vmaf()
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            #print 'state',state[6]
            #print 'action',action_prob[0]
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(
                1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            # entropy_record.append(a3c.compute_entropy(action_prob[0]))


def agent(agent_id, net_params_queue, exp_queue):
    net_env = innovation_env.Environment(random_seed=agent_id)
    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_vmaf = -1
        bit_rate = DEFAULT_QUALITY
        last_rtt = -1
        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        time_stamp = 0
        index = 1
        while True:  # experience video streaming forever
            _norm_bitrate = VIDEO_BIT_RATE[bit_rate]
            delay, loss, recv_bitrate, rtt, throughput, limbo_bytes_len = \
                net_env.get_video_chunk(bit_rate)

            rtt = float(rtt) / float(1000)
            if last_rtt < 0:
                last_rtt = rtt
            _norm_send_bitrate = bit_rate / A_DIM
            _queuing_delay = abs(rtt - last_rtt)
            _norm_recv_bitrate = min(
                float(recv_bitrate) / delay / BUFFER_NORM_FACTOR, 1.0)

            time_stamp += delay  # in ms
            vmaf = net_env.get_vmaf(bit_rate)
            if last_vmaf < 0:
                last_vmaf = vmaf

            #_normalized_bitrate = (_norm_bitrate - BITRATE_MIN) / (BITRATE_MAX - BITRATE_MIN)
            _vmaf_reward = (vmaf / _norm_bitrate) * BITRATE_MIN
            reward = vmaf - 0.2 * _norm_send_bitrate - 1.0 / DELAY_GRADIENT_MAX * \
                min(_queuing_delay, DELAY_GRADIENT_MAX) - \
                1.0 * abs(last_vmaf - vmaf)
            r_batch.append(reward)

            last_vmaf = vmaf
            last_rtt = rtt
            log_file.write(str(time_stamp) + '\t' +
                           str(_norm_bitrate) + '\t' +
                           str(recv_bitrate) + '\t' +
                           str(limbo_bytes_len) + '\t' +
                           str(rtt) + '\t' +
                           str(vmaf) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)
            state[0, -1] = _norm_send_bitrate  # last quality
            state[1, -1] = _norm_recv_bitrate  # kilo byte / ms
            state[2, -1] = _queuing_delay  # max:500ms
            state[3, -1] = float(loss)  # changed loss
            # test:add fft feature
            _fft = np.fft.fft(state[1])
            state[4] = _fft.real
            state[5] = _fft.imag
            state[6] = net_env.predict_vmaf()
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
            #print 'state',state[6]
            #print 'action',action_prob[0]
            action_cumsum = np.cumsum(action_prob)
            bit_rate = (action_cumsum > np.random.randint(
                1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN:
                exp_queue.put([s_batch[:],  # ignore the first chuck
                               a_batch[:],  # since we don't have the
                               r_batch[:],  # control over it
                               # end_of_video,
                               {'entropy': entropy_record}])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del entropy_record[:]

                #if index % MODEL_TEST_INTERVAL == 0 and agent_id == 0:
                #    print 'start test'
                    #test(actor,index)
                index += 1
                # so that in the log we know where video ends
                log_file.write('\n')

            s_batch.append(state)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)


def main():

    np.random.seed(RANDOM_SEED)
    assert len(VIDEO_BIT_RATE) == A_DIM

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, net_params_queues[i], exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
