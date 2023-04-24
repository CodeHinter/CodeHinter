import tensorflow as tf
import pickle as pkl

if __name__ == "__main__":
    model = tf.keras.models.load_model("./categorical_crossentropy_onehot_15")

    with open('./dataset_test_onehot.pkl', "rb") as f:
        [X_test, y_test] = pkl.load(f)
    print(X_test[0].shape)
    loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
    print(loss, top1, top3, top5, top10)

    # loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
