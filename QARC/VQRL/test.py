import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import tensorflow as tf
#import load_trace
import a3c
import test_env as env
import time
import warnings
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
ACTOR_LR_RATE = 6.25e-4
CRITIC_LR_RATE = 1e-3

MODEL_SAVE_INTERVAL = 1000
BUFFER_NORM_FACTOR = 5.0
DEFAULT_QUALITY = 0

RANDOM_SEED = int(time.time() % 1000)
RAND_RANGE = 1000
#LOG_FILE = './test_results/log_sim_rl'
TEST_TRACES = './test_sim_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]
LOG_FILE = sys.argv[2]

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


def core_function(src, threshold):
    if src <= threshold:
        return src / threshold
    else:
        return 1
    #_depth = 1.0 / threshold
    # return np.tanh(_depth * src)


def main():
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    #all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    cooked_files = os.listdir(TEST_TRACES)
    g1 = tf.Graph()
    with tf.Session(graph=g1) as sess:
        #sess = tf.session()
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters
        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")

        if not os.path.exists(LOG_FILE):
            os.makedirs(LOG_FILE)

        cooked_files = os.listdir(TEST_TRACES)
        for cooked_file in cooked_files:
            net_env = env.Environment(RANDOM_SEED,cooked_file)
            log_path = LOG_FILE + 'log_sim_rl_' + cooked_file + '_log.txt'
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
                print delay
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
                    0.2 * _norm_send_bitrate  - \
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
                #entropy_record.append(a3c.compute_entropy(action_prob[0]))

if __name__ == '__main__':
    main()
