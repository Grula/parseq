import onnx
import onnxruntime as ort

import cv2
import numpy as np
import math

# load onnx model
onnx_model = onnx.load("parseq.onnx")
onnx.checker.check_model(onnx_model)


sess = ort.InferenceSession('parseq.onnx')

image = cv2.imread('demo_images/01.jpg')

# 32x128
image_resized = cv2.resize(image, (128, 32))
# Convert to float32
image_resized = image_resized.astype(np.float32)
# Normlize image [0.5, 0.5]
image_resized = (image_resized - 0.5) / 0.5
# Transpose to (1, 3, 32, 128)
image_resized = np.transpose(image_resized, (2, 0, 1))
# Add batch dimension
image_resized = np.expand_dims(image_resized, axis=0)
# Run inference
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
logits = sess.run([output_name], {input_name: image_resized})[0]

print(logits)
