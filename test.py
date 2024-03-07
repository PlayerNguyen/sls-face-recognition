import numpy as np
import tensorflow as tf
from deepface import DeepFace

tf.debugging.set_log_device_placement(True)

# print(DeepFace.verify(img1_path="img1.jpg", img2_path="img3.jpg"))
table = DeepFace.find("./data/Nguyen/Nguyen1.jpg", db_path="./data", detector_backend="mtcnn")
print(table[0])

