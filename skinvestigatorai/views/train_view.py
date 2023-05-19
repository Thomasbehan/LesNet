from pyramid.view import view_config
from skinvestigatorai.core.ai.detector import SkinCancerDetector
from skinvestigatorai.core.ai.config import train_dir, val_dir, test_dir

@view_config(route_name='train', renderer='json')
def train_model(request):
    detector = SkinCancerDetector(train_dir, val_dir, test_dir)
    train_generator, val_generator, test_datagen = detector.preprocess_data()
    detector.build_model(num_classes=len(train_generator.class_indices))
    history = detector.train_model(train_generator, val_generator)
    return {'status': 'success', 'message': 'Model trained successfully \n\n ' + str(history)}
