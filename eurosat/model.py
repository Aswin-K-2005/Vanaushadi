"""Model construction and Grad-CAM, shared by training, evaluation and the app.

The model takes RAW uint8-range 64x64x3 patches and does resizing + the
backbone's own preprocessing internally. So anything that calls the model
(app, change detection, evaluation) just feeds pixels -- no chance of the
"forgot to divide by 255" bug between training and serving.

The backbone is wired in flat (not nested as a single layer), so Grad-CAM can
grab the last conv layer with a plain model.get_layer(name).
"""

import numpy as np
import tensorflow as tf

import keras

BACKBONES = {
    # name: (constructor, last conv layer name used for Grad-CAM)
    "mobilenetv2": (keras.applications.MobileNetV2, "out_relu"),
    "efficientnetb0": (keras.applications.EfficientNetB0, "top_activation"),
    "resnet50": (keras.applications.ResNet50, "conv5_block3_out"),
}


@keras.saving.register_keras_serializable(package="eurosat")
class BackbonePreprocess(keras.layers.Layer):
    """Each backbone's ImageNet preprocessing, as a real (serialisable) layer.

    A Lambda layer wrapping preprocess_input would not reload cleanly, so the
    three formulas are written out here:
      mobilenetv2    : scale 0..255 -> -1..1
      efficientnetb0 : none (the Keras EfficientNet has rescaling built in)
      resnet50       : RGB -> BGR, subtract ImageNet channel means ("caffe" mode)
    """

    def __init__(self, backbone, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone

    def call(self, x):
        if self.backbone == "mobilenetv2":
            return x / 127.5 - 1.0
        if self.backbone == "resnet50":
            return x[..., ::-1] - tf.constant([103.939, 116.779, 123.68])
        return x

    def get_config(self):
        return {**super().get_config(), "backbone": self.backbone}


def build_model(backbone="mobilenetv2", img_size=224, n_classes=10, weights="imagenet", dropout=0.3):
    ctor, _ = BACKBONES[backbone]
    inputs = keras.Input(shape=(64, 64, 3), name="patch")
    x = keras.layers.Resizing(img_size, img_size, interpolation="bilinear", name="upsample")(inputs)
    x = BackbonePreprocess(backbone, name="preprocess")(x)
    base = ctor(include_top=False, weights=weights, input_tensor=x)
    base.trainable = False
    y = keras.layers.GlobalAveragePooling2D(name="gap")(base.output)
    y = keras.layers.Dropout(dropout, name="head_dropout")(y)
    outputs = keras.layers.Dense(n_classes, activation="softmax", name="classifier")(y)
    model = keras.Model(inputs, outputs, name=f"eurosat_{backbone}")
    return model, base


def last_conv_layer_name(backbone):
    return BACKBONES[backbone][1]


def set_finetune(model, base, n_unfrozen):
    """Unfreeze the last n layers of the backbone, keeping BatchNorm frozen.

    Keeping BN in inference mode is the standard trick for fine-tuning on a
    small batch size: otherwise the running statistics learned on ImageNet get
    overwritten by noisy per-batch statistics and accuracy drops.
    """
    base.trainable = True
    layers = base.layers
    cutoff = max(0, len(layers) - n_unfrozen)
    for i, layer in enumerate(layers):
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = i >= cutoff


def load(path):
    return keras.models.load_model(path, compile=False)


# ---------------------------------------------------------------- Grad-CAM
def make_gradcam_model(model, conv_layer_name):
    conv = model.get_layer(conv_layer_name)
    return keras.Model(model.inputs, [conv.output, model.output])


def gradcam(grad_model, patches, class_idx=None):
    """Grad-CAM heatmaps for a batch of uint8 patches.

    Returns (heatmaps [N,h,w] in 0..1, probs [N,C]). If class_idx is None the
    predicted class is explained; pass an int or array to explain another
    class -- e.g. "why did it NOT say River?" for a Highway/River error.
    """
    x = tf.convert_to_tensor(np.asarray(patches, dtype=np.float32))
    with tf.GradientTape() as tape:
        conv_out, probs = grad_model(x, training=False)
        if class_idx is None:
            idx = tf.argmax(probs, axis=1)
        else:
            idx = tf.broadcast_to(tf.convert_to_tensor(class_idx, dtype=tf.int64), [tf.shape(probs)[0]])
        score = tf.gather(probs, idx, axis=1, batch_dims=1)
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy(), probs.numpy()


def overlay(patch, heatmap, alpha=0.45, out_size=256):
    """Upscale a 64x64 patch and its heatmap and blend them (returns uint8 RGB)."""
    import cv2

    img = cv2.resize(np.asarray(patch, dtype=np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    hm = cv2.resize(heatmap.astype(np.float32), (out_size, out_size), interpolation=cv2.INTER_CUBIC)
    hm = np.uint8(255 * np.clip(hm, 0, 1))
    color = cv2.cvtColor(cv2.applyColorMap(hm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (img * (1 - alpha) + color * alpha).astype(np.uint8)
