import tensorflow as tf
import pickle as pkl

if __name__ == "__main__":
    model = tf.keras.models.load_model("./mymodel_25")

    with open('./dataset_test.pkl', "rb") as f:
        [X_train, y_train] = pkl.load(f)

    with open('./dataset_train.pkl', "rb") as f:
        [X_test, y_test] = pkl.load(f)

    loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', top1, top3, top5, top10)
