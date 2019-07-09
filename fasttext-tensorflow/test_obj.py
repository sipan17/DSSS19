from fasttext_model import FastTextModel
import time
import argparse
import os


def main():
    main_start = time.time()
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, help="path to pb file")
    parser.add_argument("--model_params_path", type=str, help="path to model_params.json")
    parser.add_argument("--test_data_path", type=str, help="path to txt file (in fasttext format)")
    parser.add_argument("--label_prefix", type=str, help="label prefix", default="__label__")
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

    model = FastTextModel(model_path, model_params_path, args.label_prefix, None, use_gpu, gpu_fraction)
    results = model.test_file(test_data_path)
    print("number of datapoint: {}\naccuracy: {}\ntesting took {} seconds".
          format(results[0], results[-1], round(time.time() - main_start, 0)))


if __name__ == "__main__":
    main()
