def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    config.add_route('home', '/')
    config.add_route('supported-diagnoses', '/supported-diagnoses')
    config.add_route('predict', '/predict')
    config.add_route('labels', '/labels')
