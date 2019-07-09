import tensorflow as tf
import numpy as np
from fasttext_utils import parse_txt, make_train_vocab, make_label_vocab, next_batch, construct_label, get_all
from utils import validate, freeze_save_graph, percent
import os
import shutil
from tqdm import tqdm
import json
import time
import argparse
from sys import stdout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cache = {}


def batch_generator(descs, labels, batch_size, train_vocab, labels_lookup, word_ngrams, shuffle=False):
    global cache
    inds = np.arange(len(descs))
    rem_inds, batch_inds = next_batch(inds, batch_size, shuffle)

    while len(batch_inds) > 0:
        batch_descs = [descs[i] for i in batch_inds]
        desc_hashes = [hash(str(desc)) for desc in batch_descs]
        batch = [[0] + [train_vocab[phrase]["id"] for phrase in get_all(desc, word_ngrams) if
                        phrase in train_vocab] if h not in cache else cache[h] for
                 desc, h in zip(batch_descs, desc_hashes)]

        for h, inds in zip(desc_hashes, batch):
            if h not in cache:
                cache[h] = inds
        batch_weights = [[1 / len(i) for _ in range(len(i))] for i in batch]
        batch_labels = [labels[i] for i in batch_inds]
        batch_labels = [labels_lookup[label] for label in batch_labels]

        cur_lens = np.array([len(i) for i in batch])
        mx_len = max(cur_lens)
        to_pad = mx_len - cur_lens

        batch = [i + [0 for _ in range(pad)] for i, pad in zip(batch, to_pad)]
        batch_weights = [i + [0 for _ in range(pad)] for i, pad in zip(batch_weights, to_pad)]

        rem_inds, batch_inds = next_batch(rem_inds, batch_size, shuffle)
        yield batch, np.expand_dims(batch_weights, axis=2), batch_labels


def main():
    main_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", type=str, help="path to train file", default="./train.txt")
    parser.add_argument("--label_prefix", type=str, help="label prefix", default="__label__")
    parser.add_argument("--min_word_count", type=int, default=1,
                        help="discard words which appear less than this number")
    parser.add_argument("--min_label_count", type=int, default=1,
                        help="discard labels which appear less than this number")
    parser.add_argument("--dim", type=int, default=100, help="length of embedding vector")
    parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--word_ngrams", type=int, default=1, help="word ngrams")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for train")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
    parser.add_argument("--learning_rate_multiplier", type=float, default=0.8,
                        help="learning rate multiplier after each epoch")
    parser.add_argument("--data_fraction", type=float, default=1,
                        help="data fraction, if < 1, train (and validation) data will be randomly sampled")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu for training")
    parser.add_argument("--gpu_fraction", type=float, default=0.5, help="what fraction of gpu to allocate")
    parser.add_argument("--result_dir", type=str, help="result dir", default="./results/")

    args = parser.parse_args()
    assert args.use_gpu in [0, 1]
    train_path = args.train_path
    label_prefix = args.label_prefix
    min_word_count = args.min_word_count
    min_label_count = args.min_label_count
    emb_dim = args.dim
    n_epochs = args.n_epochs
    word_ngrams = args.word_ngrams
    batch_size = args.batch_size
    initial_learning_rate = args.learning_rate
    learning_rate_multiplier = args.learning_rate_multiplier
    seed = args.seed
    data_fraction = args.data_fraction
    use_gpu = bool(args.use_gpu)
    gpu_fraction = args.gpu_fraction
    result_dir = validate(args.result_dir)

    print('training with arguments:')
    print(args)
    print('\n')

    np.random.seed(seed)

    train_descs, train_labels, max_words = parse_txt(train_path, return_max_len=True, debug_till_row=-1,
                                                     fraction=data_fraction, seed=seed, label_prefix=label_prefix)

    model_params = {
        "word_ngrams": word_ngrams,
        "word_id_path": os.path.abspath(os.path.join(result_dir, "word_id.json")),
        "label_dict_path": os.path.abspath(os.path.join(result_dir, "label_dict.json"))
    }

    for child_dir in os.listdir(result_dir):
        dir_tmp = os.path.join(result_dir, child_dir)
        if os.path.isdir(dir_tmp):
            shutil.rmtree(dir_tmp)
        if dir_tmp.endswith(".pb"):
            os.remove(dir_tmp)

    max_words_with_ng = 1
    for ng in range(word_ngrams):
        max_words_with_ng += max_words - ng

    print("preparing dataset")
    print("total number of datapoints: {}".format(len(train_descs)))
    print("max number of words in description: {}".format(max_words))
    print("max number of words with n-grams in description: {}".format(max_words_with_ng))

    label_dict_path = os.path.join(result_dir, "label_dict.json")
    word_id_path = os.path.join(result_dir, "word_id.json")

    train_vocab = make_train_vocab(train_descs, word_ngrams)
    label_vocab = make_label_vocab(train_labels)

    if min_word_count > 1:
        tmp_cnt = 1
        train_vocab_thresholded = {}
        for k, v in sorted(train_vocab.items(), key=lambda t: t[0]):
            if v["cnt"] >= min_word_count:
                v["id"] = tmp_cnt
                train_vocab_thresholded[k] = v
                tmp_cnt += 1

        train_vocab = train_vocab_thresholded.copy()
        del train_vocab_thresholded

        print("number of unique words and phrases after thresholding: {}".format(len(train_vocab)))

    print("\nnumber of labels in train: {}".format(len(set(label_vocab.keys()))))
    if min_label_count > 1:
        label_vocab_thresholded = {}
        tmp_cnt = 0
        for k, v in sorted(label_vocab.items(), key=lambda t: t[0]):
            if v["cnt"] >= min_label_count:
                v["id"] = tmp_cnt
                label_vocab_thresholded[k] = v
                tmp_cnt += 1

        label_vocab = label_vocab_thresholded.copy()
        del label_vocab_thresholded

        print("number of unique labels after thresholding: {}".format(len(label_vocab)))

    final_train_labels = set(label_vocab.keys())

    with open(label_dict_path, "w+") as outfile:
        json.dump(label_vocab, outfile)
    with open(word_id_path, "w+") as outfile:
        json.dump(train_vocab, outfile)
    with open(os.path.join(result_dir, "model_params.json"), "w+") as outfile:
        json.dump(model_params, outfile)

    num_words_in_train = len(train_vocab)
    num_labels = len(label_vocab)

    train_descs2, train_labels2 = [], []
    labels_lookup = {}

    labels_thrown, descs_thrown = 0, 0
    for train_desc, train_label in zip(tqdm(train_descs), train_labels):
        final_train_inds = [0] + [train_vocab[phrase]["id"] for phrase in
                                  get_all(train_desc, word_ngrams) if
                                  phrase in train_vocab]
        if len(final_train_inds) == 1:
            descs_thrown += 1
            continue

        if train_label not in labels_lookup:
            if train_label in final_train_labels:
                labels_lookup[train_label] = construct_label(label_vocab[train_label]["id"], num_labels)
            else:
                labels_thrown += 1
                continue

        train_labels2.append(train_label)
        train_descs2.append(train_desc)
    del train_descs, train_labels

    print("\n{} datapoints thrown because of empty description".format(descs_thrown))
    if min_label_count > 1:
        print("{} datapoints thrown because of label".format(labels_thrown))

    if use_gpu:
        device = "/gpu:0"
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                                          allow_growth=True))
    else:
        device = "/cpu:0"
        config = tf.ConfigProto(allow_soft_placement=True)

    with tf.device(device):
        with tf.Session(config=config) as sess:
            input_ph = tf.placeholder(tf.int32, shape=[None, None], name="input")
            weights_ph = tf.placeholder(tf.float32, shape=[None, None, 1], name="input_weights")
            labels_ph = tf.placeholder(tf.float32, shape=[None, num_labels], name="label")
            learning_rate_ph = tf.placeholder_with_default(initial_learning_rate, shape=[], name="learning_rate")

            tf.set_random_seed(seed)

            with tf.name_scope("embeddings"):
                look_up_table = tf.Variable(tf.random_uniform([num_words_in_train + 1, emb_dim]),
                                            name="embedding_matrix")

            with tf.name_scope("mean_sentece_vector"):
                gath_vecs = tf.gather(look_up_table, input_ph)
                weights_broadcasted = tf.tile(weights_ph, tf.stack([1, 1, emb_dim]))
                mean_emb = tf.reduce_sum(tf.multiply(weights_broadcasted, gath_vecs), axis=1, name="sentence_embedding")

            logits = tf.layers.dense(mean_emb, num_labels, use_bias=False,
                                     kernel_initializer=tf.truncated_normal_initializer(), name="logits")
            output = tf.nn.softmax(logits, name="prediction")
            # this is not used in the training, but will be used for inference

            correctly_predicted = tf.nn.in_top_k(logits, tf.argmax(labels_ph, axis=1), 1, name="top_1")

            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_ph,
                                                                                logits=logits), name="ce_loss")

            train_op = tf.train.AdamOptimizer(learning_rate_ph).minimize(ce_loss)
            sess.run(tf.global_variables_initializer())

            train_start = time.time()

            for epoch in range(n_epochs):
                print("\nepoch {} started".format(epoch + 1))

                end_epoch_accuracy, end_epoch_accuracy_k = [], []
                moving_loss = []
                for batch, batch_weights, batch_labels in \
                        batch_generator(train_descs2, train_labels2, batch_size, train_vocab, labels_lookup,
                                        word_ngrams, shuffle=True):
                    _, correct, batch_loss = sess.run([train_op, correctly_predicted, ce_loss],
                                                      feed_dict={input_ph: batch,
                                                                 weights_ph: batch_weights,
                                                                 labels_ph: batch_labels})

                    end_epoch_accuracy.extend(correct)
                    moving_loss.append(batch_loss)

                print('\ncurrent learning rate: {}'.format(round(initial_learning_rate, 7)))
                print("epoch {} ended".format(epoch + 1))
                print("epoch moving mean loss: {}".format(np.round(np.mean(moving_loss), 3)))
                print("train moving average accuracy: {}".format(percent(end_epoch_accuracy)))

                initial_learning_rate *= learning_rate_multiplier

            freeze_save_graph(sess, result_dir, "model_ep{}.pb".format(epoch + 1), "prediction")
            print("the model is stored at {}".format(result_dir))
            print("the training took {} seconds".format(round(time.time() - train_start, 0)))
    print("all process took {} seconds".format(round(time.time() - main_start, 0)))


if __name__ == "__main__":
    main()
