from DataOps import _read_tfrecords_for_lm
from Models import convert_to_strings


if __name__ == '__main__':
    ds_train = _read_tfrecords_for_lm("b:/!temp/y_pred_y_true_pairs/train.tfrecord")

    for y_pred, y_true in ds_train:
        y_pred = [[y_pred]]
        y_true = [[y_true]]
        print(f"prediction: {convert_to_strings(y_pred)[-1]}")
        print(f"truth: {convert_to_strings(y_true)[-1]}")
