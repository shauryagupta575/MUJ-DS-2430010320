"""Corrections for three issues found in notebooks/wildfire_detection_vit-2.ipynb.

These functions are drop-in replacements for the notebook versions. They have NOT
been run against the recorded results; rerun the affected notebook cells after
swapping them in and update the reports with the new numbers.

1. make_gradcam_heatmap  - the notebook version fails with AttributeError under Keras 3.
2. balanced_focal_loss   - the notebook's focal loss puts alpha = 0.75 on the majority
                           (fire) class and sums instead of averaging.
3. make_eval_ds          - ResNet50 / MobileNetV2 training accuracy was computed on
                           unpreprocessed images; this builds a correct evaluation set.
"""
import tensorflow as tf
from tensorflow import keras


def make_gradcam_heatmap(img_batch, model, last_conv_layer_name="conv4"):
    """Grad-CAM for a Sequential CNN with a single sigmoid output.

    Walks the layers manually instead of building a sub-model, which avoids the
    Keras 3 AttributeError raised by keras.Model(model.inputs, ...).
    """
    with tf.GradientTape() as tape:
        x, conv_out = img_batch, None
        for layer in model.layers:
            x = layer(x, training=False)
            if layer.name == last_conv_layer_name:
                conv_out = x
        prediction = x[:, 0]
    grads = tape.gradient(prediction, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ weights[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def balanced_focal_loss(gamma=2.0, alpha_fire=0.25):
    """Focal loss that up-weights the minority class (no-fire).

    alpha_fire is the weight on label 1 (fire); no-fire gets 1 - alpha_fire.
    Uses Keras' built-in implementation with mean reduction.
    """
    return keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, alpha=alpha_fire, gamma=gamma)


def make_eval_ds(paths, labels, preprocess_fn, load_fn, batch_size=32):
    """Un-augmented, correctly preprocessed dataset for evaluating a backbone.

    Pass it as eval_train=... to evaluate_and_record() so the reported training
    accuracy uses the same preprocessing as the test set.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
