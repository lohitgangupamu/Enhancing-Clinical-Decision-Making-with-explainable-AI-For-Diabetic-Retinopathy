from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model('inception_resnet_v2_model.keras')  # Load your trained model

# Set folder for saving outputs
OUTPUT_FOLDER = 'static'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Define Grad-CAM++ Function
def grad_cam_plus_plus(input_model, image, layer_name):
    # Reference the input layer explicitly
    grad_model = tf.keras.models.Model(
        inputs=[input_model.input],  # Use the model's input layer directly
        outputs=[
            input_model.get_layer(layer_name).output,  # Final convolutional layer output
            input_model.output  # Model prediction output
        ]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)  # Pass the preprocessed image to the grad_model
        loss = predictions[:, tf.argmax(predictions[0])]  # Use the predicted class to calculate loss

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Process gradients and outputs
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads

    # Perform weighted sum of convolutional outputs
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    # Resize and normalize the heatmap
    cam = cv2.resize(cam.numpy(), (299, 299))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    return heatmap, predictions


# Define Preprocessing Function
def preprocess_image(image):
    green_channel = image[:, :, 1]
    green_channel = green_channel.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green_channel = clahe.apply(green_channel)
    resized = cv2.resize(green_channel, (299, 299), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    processed_image = np.stack((normalized, normalized, normalized), axis=-1)
    tensor_input = tf.convert_to_tensor(np.expand_dims(processed_image, axis=0), dtype=tf.float32)
    return tensor_input

# Define Function to Draw Bounding Boxes
# Define Function to Draw Bounding Boxes (Improved Version)
def draw_bounding_boxes(original_image, heatmap):
    # Resize heatmap to match the original image dimensions
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert heatmap to 8-bit grayscale
    heatmap_8bit = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Threshold the heatmap to detect high-activation areas
    _, binary_heatmap = cv2.threshold(heatmap_8bit, 150, 255, cv2.THRESH_BINARY)

    # Perform morphological operations to enhance regions of interest
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_heatmap = cv2.morphologyEx(binary_heatmap, cv2.MORPH_CLOSE, kernel)

    # Find contours in the binary heatmap
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes directly based on heatmap regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 200:  # Filter for significant regions
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original_image





def add_text_to_image(image, text, position, font_scale=1, color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image




@app.route('/')
def index():
    return render_template('index.html') 
     # Render the upload form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files['image']  # Retrieve uploaded image
        if not image_file:
            return jsonify({'error': 'No file uploaded'})

        # Decode the uploaded image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Save the original image as it is uploaded
        

        # Preprocess the image for model prediction
        preprocessed_image = preprocess_image(image)
        layer_name = 'conv_7b_ac'  # Replace with your model's final convolutional layer
        heatmap, predictions = grad_cam_plus_plus(model, preprocessed_image, layer_name)

        # Extract prediction details
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class]) * 100

        # Save the heatmap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_filename = f"heatmap_{predicted_class}.png"
        heatmap_path = os.path.join(OUTPUT_FOLDER, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_colored)

        # Generate image with bounding boxes
        original_with_boxes = draw_bounding_boxes(image.copy(), heatmap)
        boxes_filename = f"bounding_boxes_{predicted_class}.png"
        boxes_path = os.path.join(OUTPUT_FOLDER, boxes_filename)
        cv2.imwrite(boxes_path, original_with_boxes)

        return jsonify({
            'class': predicted_class,
            'confidence': confidence,
            
            'heatmap_url': heatmap_filename,
            'bounding_boxes_url': boxes_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)})










@app.route('/static/<filename>')
def serve_static(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)
