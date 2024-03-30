def generate_route_name():
    import uuid
    return str(uuid.uuid4()) + '/train'


def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('train', generate_route_name())
    config.add_route('tensorboard', '/tensorboard')
    config.add_route('predict', '/predict')
    config.add_route('dashboard', '/dashboard')
