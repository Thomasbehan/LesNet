def main(global_config, **settings):
    """Return a Pyramid WSGI application.

    Pyramid is imported lazily so the ML subpackage (lesnet.ml) can be used in
    training/inference environments that don't install the web stack.
    """
    from pyramid.config import Configurator

    with Configurator(settings=settings) as config:
        config.include('pyramid_jinja2')
        config.include('.routes')
        config.scan()
    return config.make_wsgi_app()
