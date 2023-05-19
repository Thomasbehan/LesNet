import os
from skinvestigatorai.core.route_generator import RouteGenerator

train_route_name = RouteGenerator.generate_route_name()
os.environ['TRAIN_ROUTE_NAME'] = train_route_name
print("routes.py: " + os.environ['TRAIN_ROUTE_NAME'])


def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('train', '/' + os.environ['TRAIN_ROUTE_NAME'])
    config.add_route('tensorboard', '/tensorboard')
    config.add_route('predict', '/predict')
    config.add_route('dashboard', '/dashboard')
