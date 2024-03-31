import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class FeatureExtractionService:
    def __init__(self, model, model_type):
        self.model_type = model_type
        self.model = model
        self.feature_extractor = None
        self.dataset_embedding = None

    def create_feature_extractor(self):
        if self.model_type == 'H5':
            self.feature_extractor = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
        elif self.model_type == 'TFLITE':
            self.feature_extractor = self.model
        else:
            raise ValueError("Unsupported model type. Please use 'H5' or 'TFLITE'.")

    def preprocess_image_for_tflite(self, img):
        img_resized = tf.image.resize(img, [128, 128])
        img_normalized = img_resized / 255.0
        return img_normalized

    def calculate_dataset_embedding(self, data_generator):
        features = []
        if self.model_type == 'H5':
            for _, (imgs, _) in enumerate(data_generator):
                features.append(self.feature_extractor.predict(imgs))
        elif self.model_type == 'TFLITE':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            for imgs, _ in data_generator:
                # Preprocess images to match TFLite input requirements
                for img in imgs:
                    img = self.preprocess_image_for_tflite(img)  # Implement this based on your model needs
                    img = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])
                    self.model.set_tensor(input_details[0]['index'], img)
                    self.model.invoke()
                    features.append(self.model.get_tensor(output_details[0]['index'])[0])
        else:
            raise ValueError("Unsupported model type. Please use 'H5' or 'TFLITE'.")

        features = np.concatenate(features, axis=0)
        self.dataset_embedding = np.mean(features, axis=0)

    def predict_image(self, image):
        if self.model_type == 'H5':
            return self.feature_extractor.predict(image[np.newaxis, ...])
        elif self.model_type == 'TFLITE':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()
            self.model.set_tensor(input_details[0]['index'], image[np.newaxis, ...].astype(np.float32))
            self.model.invoke()
            return self.model.get_tensor(output_details[0]['index'])

    def is_image_similar(self, image, threshold=0.8):
        image_embedding = self.predict_image(image)
        if image_embedding is not None and self.dataset_embedding is not None:
            similarity = np.dot(image_embedding, self.dataset_embedding) / (
                    np.linalg.norm(image_embedding) * np.linalg.norm(self.dataset_embedding))
            return similarity >= threshold
        return False
