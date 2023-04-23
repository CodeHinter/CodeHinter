import tensorflow as tf
import pickle as pkl

if __name__ == "__main__":
    model = tf.keras.models.load_model("./mymodel_25")

    with open('./dataset_train.pkl', "rb") as f:
        [X_test, y_test] = pkl.load(f)
    print(X_test[0].shape)
    output = model.predict(X_test[0].reshape(512,77))
    print(output)

    # loss, top1, top3, top5, top10 = model.evaluate(X_test, y_test)
