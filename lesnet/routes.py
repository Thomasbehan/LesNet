import time

from pyramid.static import QueryStringConstantCacheBuster


def includeme(config):
    config.add_static_view('static', 'static', cache_max_age=3600)
    # Every process start (i.e. every deploy) gets a new ?x= token on static URLs, so browsers never keep a
    # stale stylesheet or script for an hour after a release.
    config.add_cache_buster('lesnet:static/', QueryStringConstantCacheBuster(str(int(time.time()))))
    config.add_route('home', '/')
    config.add_route('supported-diagnoses', '/supported-diagnoses')
    config.add_route('predict', '/predict')
    config.add_route('labels', '/labels')
