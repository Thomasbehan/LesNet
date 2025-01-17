import logging

import numpy as np
from pyramid.httpexceptions import HTTPBadRequest
from tensorflow.lite.python.interpreter import Interpreter

from lesnet.config.model import ModelConfig
from lesnet.services.model import SVModel
from lesnet.services.data import Data

log = logging.getLogger(__name__)


class Inference:
    def __init__(self):
        self.model_service = SVModel()
        self.data_service = Data()
        self.model, self.class_labels = self.model_service.load_model()
        self.dataset_embedding = None

    def calculate_dataset_embedding(self, data_generator):
        features = []
        if ModelConfig.MODEL_TYPE == 'KERAS':
            for _, (imgs, _) in enumerate(data_generator):
                features.append(self.model.predict(imgs))
        elif ModelConfig.MODEL_TYPE == 'TFLITE':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            for imgs, _ in data_generator:
                for img in imgs:
                    img = self.model_service.preprocess_image_for_tflite(img)
                    img = np.expand_dims(img, axis=0).astype(input_details[0]['dtype'])
                    self.model.set_tensor(input_details[0]['index'], img)
                    self.model.invoke()
                    features.append(self.model.get_tensor(output_details[0]['index'])[0])
        else:
            raise ValueError("Unsupported model type. Please use 'H5' or 'TFLITE'.")

        features = np.concatenate(features, axis=0)
        self.dataset_embedding = np.mean(features, axis=0)

    def is_image_similar(self, image, threshold=0.8):
        image_embedding = self._predict_similar(image)
        if image_embedding is not None and self.dataset_embedding is not None:
            similarity = np.dot(image_embedding, self.dataset_embedding) / (
                    np.linalg.norm(image_embedding) * np.linalg.norm(self.dataset_embedding))
            return similarity >= threshold
        return False

    def _predict_similar(self, image):
        if ModelConfig.MODEL_TYPE == 'KERAS':
            return self.model.predict(image[np.newaxis, ...])
        elif ModelConfig.MODEL_TYPE == 'TFLITE':
            input_details = self.model.get_input_details()
            output_details = self.model.get_output_details()

            image_preprocessed = self.model_service.preprocess_image_for_tflite(image)
            image_preprocessed = np.expand_dims(image_preprocessed, axis=0).astype(input_details[0]['dtype'])

            self.model.set_tensor(input_details[0]['index'], image_preprocessed)
            self.model.invoke()
            return self.model.get_tensor(output_details[0]['index'])
        else:
            raise ValueError("Unsupported model type")

    def predict(self, image_file):
        try:
            image = self.data_service.load_image_for_prediction(image_file)
            image_array = np.expand_dims(image, axis=0)

            # Make a prediction
            if isinstance(self.model, Interpreter):
                self.model.allocate_tensors()
                input_details = self.model.get_input_details()
                self.model.set_tensor(input_details[0]['index'], image_array)
                self.model.invoke()
                output_details = self.model.get_output_details()
                predictions = self.model.get_tensor(output_details[0]['index'])
            else:
                predictions = self.model.predict(image_array)

            # Get the class probabilities and corresponding labels
            predicted_probabilities = predictions[0]
            predicted_classes = np.argsort(predicted_probabilities)[::-1]  # Sort classes by probability
            top_n = 5  # Number of top predictions to return
            results = []

            for i in range(top_n):
                class_index = predicted_classes[i]
                results.append({
                    'label': self.class_labels[class_index],
                    'probability': float(predicted_probabilities[class_index]) * 100
                })

            return {
                'predictions': results
            }
        except Exception as e:
            log.exception(e)
            return HTTPBadRequest(detail=str(e))
