import os
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

def mnist_dataset():
    (x, y), _ = datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds


def train(epoch, model, optimizer):

  train_ds = mnist_dataset()
  loss = 0.0
  accuracy = 0.0
  for step, (x, y) in enumerate(train_ds):
    loss, accuracy = train_one_step(model, optimizer, x, y)

    if step % 500 == 0:
      print('epoch', epoch, ': loss', loss.numpy(), '; accuracy', accuracy.numpy())

  return loss, accuracy


@tf.function
def prepare_mnist_features_and_labels(x, y):
    return tf.cast(x, tf.float32) / 255.0, tf.cast(y, tf.int64)


@tf.function
def compute_loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )


@tf.function
def compute_acc(logits, labels):
    pred = tf.argmax(logits, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(pred, labels), tf.float32))


@tf.function
def train_one_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = compute_loss(logits, y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    acc = compute_acc(logits, y)
    return loss, acc


if __name__ == '__main__':
    model = keras.Sequential([
        layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        layers.Dense(100, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(10)])

    optimizer = optimizers.Adam()

    for epoch in range(20):
        loss, accuracy = train(epoch, model, optimizer)
        print(f'loss: {loss}, accuracy: {accuracy}')

    pass
