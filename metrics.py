import keras


def single_class_precision(interesting_class_id):
    def prec(y_true, y_pred):
        class_id_true = keras.argmax(y_true, axis=-1)
        class_id_pred = keras.argmax(y_pred, axis=-1)
        precision_mask = keras.cast(keras.equal(
            class_id_pred, interesting_class_id), 'int32')
        class_prec_tensor = keras.cast(keras.equal(
            class_id_true, class_id_pred), 'int32') * precision_mask
        class_prec = keras.cast(keras.sum(class_prec_tensor), 'float32') / \
            keras.cast(keras.maximum(keras.sum(precision_mask), 1), 'float32')
        return class_prec
    return prec


def single_class_recall(interesting_class_id):
    def recall(y_true, y_pred):
        class_id_true = keras.argmax(y_true, axis=-1)
        class_id_pred = keras.argmax(y_pred, axis=-1)
        recall_mask = keras.cast(keras.equal(
            class_id_true, interesting_class_id), 'int32')
        class_recall_tensor = keras.cast(keras.equal(
            class_id_true, class_id_pred), 'int32') * recall_mask
        class_recall = keras.cast(keras.sum(class_recall_tensor), 'float32') / \
            keras.cast(keras.maximum(keras.sum(recall_mask), 1), 'float32')
        return class_recall
    return recall
