import json
import logging

import numpy as np
from PIL import Image
from pyramid.httpexceptions import HTTPBadRequest
from pyramid.response import Response
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.lite.python.interpreter import Interpreter

from skinvestigatorai.config.model import ModelConfig
from skinvestigatorai.services.model import SVModel

log = logging.getLogger(__name__)


class Inference:
    def __init__(self):
        self.model_service = SVModel()
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
            image = Image.open(image_file).convert('RGB')
            image = image.resize(ModelConfig.IMG_SIZE)
            image_array = img_to_array(image)
            image_array = image_array / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            threshold = 0.50

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

            max_confidence = np.max(predictions)

            # Check if the image is similar based on the highest confidence score
            if max_confidence < threshold:
                error_message = """
                                I'm not too sure about this one?
                                Please make sure the image is of a skin lesion, is clear, focused, and occupies most of 
                                the frame while leaving sufficient space around the edges.
                                """
                return Response(status=400, body=json.dumps({"error": error_message.strip()}),
                                content_type='application/json; charset=UTF-8')
            else:

                print("Predictions: ", predictions)
                print("Predictions MAXARG: ", np.argmax(predictions))
                print("Class Labels: ", self.class_labels)

                predicted_class = self.class_labels[np.argmax(predictions)]

                # Return the prediction result
                return {
                    'prediction': predicted_class,
                    'confidence': float(predictions[0][np.argmax(predictions)]) * 100
                }
        except Exception as e:
            log.exception(e)
            return HTTPBadRequest(detail=str(e))
