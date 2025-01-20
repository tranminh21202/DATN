from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import base64
import os
import subprocess
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import numpy as np
import cv2
import align.detect_face
import facenet
from flask_cors import CORS
from flask_cors import cross_origin


app = Flask(__name__)
app.secret_key = 'your_secret_key'
CORS(app)

# Configuration constants
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
DATASET_RAW_PATH = './dataset/facedata/raw'
DATASET_PROCESSED_PATH = './dataset/facedata/processed'
MODEL_PATH = './Models/20180402-114759.pb'
CLASSIFIER_MODEL_PATH = './Models/facemodel.pkl'

# Load The Custom Classifier
with open(CLASSIFIER_MODEL_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier loaded successfully")

# Initialize TensorFlow session and load FaceNet model
tf.Graph().as_default()
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
facenet.load_model(MODEL_PATH)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

@app.route('/')
def index():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    save_path = os.path.join(DATASET_RAW_PATH, username)

    os.makedirs(save_path, exist_ok=True)  # Tạo thư mục nếu chưa có
    process_images(username)
    train_model()
    flash('Registration successful!')
    return render_template('authen.html')

@app.route('/save_image', methods=['POST'])
def save_image():
    image_data = request.form['image']
    username = request.form['username']
    count = request.form['count']

    image_data = base64.b64decode(image_data.split(",")[1])  # Decode ảnh từ base64
    save_dir = os.path.join(DATASET_RAW_PATH, username)
    save_path = os.path.join(save_dir, f"{username}_{count}.jpg")

    os.makedirs(save_dir, exist_ok=True)  # Tạo thư mục nếu chưa có
    with open(save_path, 'wb') as f:
        f.write(image_data)

    if count == '49':
        flash('Photos captured successfully!')  # Hiển thị thông báo khi đủ 10 ảnh

    return jsonify(success=True)

def process_images(username):
    command = f"python src/align_dataset_mtcnn.py {DATASET_RAW_PATH} {DATASET_PROCESSED_PATH} --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25"
    subprocess.run(command, shell=True)

def train_model():
    command = f"python src/classifier.py TRAIN {DATASET_PROCESSED_PATH} {MODEL_PATH} {CLASSIFIER_MODEL_PATH} --batch_size 1000"
    subprocess.run(command, shell=True)

@app.route('/authen', methods=['GET', 'POST'])
@cross_origin()
def authen():
    global model
    if request.method == 'POST':
        name = "Unknown"
        uploaded_file = request.files.get('image')  # Kiểm tra file được tải lên

        # Lấy dữ liệu ảnh từ file tải lên hoặc ảnh chụp (base64)
        if uploaded_file:
            img_data = uploaded_file.read()
        elif 'capturedImageData' in request.form:
            img_data = base64.b64decode(request.form['capturedImageData'].split(",")[1])
        else:
            return "Không có dữ liệu ảnh để xác thực", 400

        # Đọc ảnh từ buffer và xử lý để nhận diện khuôn mặt
        np_img = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]

        if faces_found > 0:
            for bb in bounding_boxes:
                bb = bb.astype(int)
                cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)

                # Dự đoán tên người dùng
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                # Xác thực người dùng nếu xác suất đạt yêu cầu
                if best_class_probabilities[0] > 0.7:
                    name = class_names[best_class_indices[0]]
                    return f"Xác thực khuôn mặt thành công. Hello {name}!"
                else:
                    return "Xác thực khuôn mặt không thành công."
        else:
            return "Không tìm thấy khuôn mặt trong ảnh."

    # Render trang xác thực nếu phương thức là GET
    return render_template("authen.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)