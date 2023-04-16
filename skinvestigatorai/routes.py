def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('train', '/train')
    config.add_route('tensorboard', '/tensorboard')
    config.add_route('predict', '/predict')
    config.add_route('dashboard', '/dashboard')
