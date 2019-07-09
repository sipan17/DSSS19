import time
import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from fasttext_utils import parse_txt, next_batch, get_all
from utils import load_graph

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cache = {}


def batch_generator(descs, batch_size, train_vocab, word_ngrams, sort_ngrams, shuffle=False,
                    show_progress=True):
    global cache
    inds = np.arange(len(descs))
    rem_inds, batch_inds = next_batch(inds, batch_size, shuffle)

    if show_progress:
        progress_bar = tqdm(total=int(np.ceil(len(descs) / batch_size)))
    while len(batch_inds) > 0:
        batch_descs = [descs[i] for i in batch_inds]
        desc_hashes = [hash(str(desc)) for desc in batch_descs]
        batch = [[0] + [train_vocab[phrase]["id"] for phrase in get_all(desc, word_ngrams, sort_ngrams) if
                        phrase in train_vocab] if h not in cache else cache[h] for
                 desc, h in zip(batch_descs, desc_hashes)]

        for h, inds in zip(desc_hashes, batch):
            if h not in cache:
                cache[h] = inds
        batch_weights = [[1 / len(i) for _ in range(len(i))] for i in batch]

        cur_lens = np.array([len(i) for i in batch])
        mx_len = max(cur_lens)
        to_pad = mx_len - cur_lens

        batch = [i + [0 for _ in range(pad)] for i, pad in zip(batch, to_pad)]
        batch_weights = [i + [0 for _ in range(pad)] for i, pad in zip(batch_weights, to_pad)]

        rem_inds, batch_inds = next_batch(rem_inds, batch_size, shuffle)
        if show_progress:
            progress_bar.update()
        yield batch, np.expand_dims(batch_weights, axis=2)

    if show_progress:
        progress_bar.close()


def main():
    main_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="path to pb file")
    parser.add_argument("--model_params_path", type=str, help="path to model_params.json")
    parser.add_argument("--test_data_path", type=str, help="path to txt file (in fasttext format)")
    parser.add_argument("--label_prefix", type=str, help="label prefix", default="__label__")
    parser.add_argument("--batch_size", type=int, default=4096, help="batch size for inference")
    parser.add_argument("--use_gpu", type=int, default=0, help="use gpu for training")
    parser.add_argument("--gpu_fraction", type=float, default=0.5, help="what fraction of gpu to allocate")

    args = parser.parse_args()
    model_path = args.model_path
    model_params_path = args.model_params_path
    test_data_path = args.test_data_path
    use_gpu = args.use_gpu
    assert use_gpu in [0, 1], "use_gpu parameter should be either 0 or 1"
    use_gpu = bool(use_gpu)
    gpu_fraction = args.gpu_fraction
    assert 0 <= gpu_fraction <= 1, "gpu_fraction parameter should be in [0, 1]"
    for path in [model_path, model_params_path, test_data_path]:
        assert os.path.isfile(path), "no such file: {}".format(path)
    assert model_path.endswith(".pb"), "model_path should be .pb file"

    test_descs, test_labels, max_words = parse_txt(test_data_path, return_max_len=True, debug_till_row=-1,
                                                   fraction=1, seed=None, label_prefix=args.label_prefix)

    with open(model_params_path) as infile:
        model_params = json.load(infile)

    with open(model_params["word_id_path"]) as infile:
        train_vocab = json.load(infile)
    with open(model_params["label_dict_path"]) as infile:
        label_dict = json.load(infile)

    test_labels_num = [label_dict[test_label]["id"] for test_label in test_labels]

    word_ngrams = model_params["word_ngrams"]
    sort_ngrams = False
    if "sort_ngrams" in model_params:
        sort_ngrams = model_params["sort_ngrams"]

    get_list = ["input", "input_weights", "embeddings/embedding_matrix/read",
                "mean_sentece_vector/sentence_embedding", "logits/kernel/read", "prediction"]
    get_list = [i + ":0" for i in get_list]

    device = "/cpu:0"
    if use_gpu:
        device = "/gpu:0"
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                                          allow_growth=True))
    else:
        config = tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)

    with tf.device(device):
        sess = tf.Session(config=config)
        input_ph, weights_ph, input_mat, sent_vec, output_mat, output = load_graph(model_path, get_list)

    preds = []

    for batch, batch_weights in batch_generator(test_descs, args.batch_size, train_vocab, word_ngrams, sort_ngrams):
        batch_probs = sess.run(output, feed_dict={input_ph: batch, weights_ph: batch_weights})
        preds.extend(np.argmax(batch_probs, axis=1))

    accuracy = np.round(np.mean(np.array(preds) == test_labels_num), 5)

    print("number of datapoint: {}\naccuracy: {}\ntesting took {} seconds".
          format(len(preds), accuracy, round(time.time() - main_start, 0)))


if __name__ == "__main__":
    main()
