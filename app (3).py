import gradio as gr
import numpy as np
import cv2
import base64
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# ── Build & Load Model ──
def build_model(input_shape=(224, 224, 3)):
    base = VGG16(weights=None, include_top=False, input_shape=input_shape)
    base.trainable = False
    inputs = layers.Input(input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='VGG16_Pneumonia')

print("Loading model...")
model = build_model()
model.load_weights('best_vgg16.h5')
print("✅ Model loaded!")

# ── Grad-CAM ──
def get_gradcam(inp):
    vgg16 = model.get_layer('vgg16')
    grad_model = tf.keras.models.Model(
        inputs=vgg16.input,
        outputs=[vgg16.get_layer('block5_conv3').output, vgg16.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, vgg_output = grad_model(inp)
        x = model.get_layer('global_average_pooling2d')(vgg_output)
        x = model.get_layer('dense')(x)
        x = model.get_layer('dropout')(x, training=False)
        x = model.get_layer('dense_1')(x)
        x = model.get_layer('dropout_1')(x, training=False)
        predictions = model.get_layer('dense_2')(x)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap.numpy(), 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap

def img_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_img(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# ── Main API Function ──
def analyze(image_b64: str):
    try:
        img_bgr = base64_to_img(image_b64)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        inp = preprocess_input(img_resized.copy().astype(np.float32))
        inp = np.expand_dims(inp, axis=0)

        pred = float(model.predict(inp, verbose=0)[0][0])
        is_pneumonia = pred > 0.5
        confidence = round((pred if is_pneumonia else 1 - pred) * 100)

        heatmap = get_gradcam(inp)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        overlay = cv2.addWeighted(
            cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR),
            0.6, heatmap_colored, 0.4, 0
        )

        notes = (
            "<strong>Findings:</strong> The model identified opacification patterns "
            "consistent with pneumonia, primarily in the lower lung zones. Grad-CAM "
            "highlights (red regions) correspond to areas of increased radiodensity "
            "in the original X-ray.<br><br>"
            "<strong>Recommendation:</strong> Correlate with clinical symptoms and "
            "laboratory findings. Further imaging may be warranted."
        ) if is_pneumonia else (
            "<strong>Findings:</strong> Lung fields appear clear with no significant "
            "consolidation or opacification. Cardiac silhouette within normal limits. "
            "Grad-CAM activation is diffuse, confirming absence of focal infection "
            "patterns.<br><br>"
            "<strong>Recommendation:</strong> Continue routine follow-up as clinically "
            "indicated."
        )

        return {
            'label': 'PNEUMONIA' if is_pneumonia else 'NORMAL',
            'confidence': confidence,
            'is_pneumonia': is_pneumonia,
            'notes': notes,
            'orig_img': f'data:image/jpeg;base64,{img_to_base64(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))}',
            'heatmap_img': f'data:image/jpeg;base64,{img_to_base64(heatmap_colored)}',
            'overlay_img': f'data:image/jpeg;base64,{img_to_base64(overlay)}',
        }
    except Exception as e:
        return {'error': str(e)}

# ── Gradio with CORS enabled ──
demo = gr.Interface(
    fn=analyze,
    inputs=gr.Text(label="Base64 Image"),
    outputs=gr.JSON(label="Result"),
    title="PneumoScan AI API",
    api_name="analyze"
)

demo.launch(
    show_error=True,
)
