from pyramid.view import view_config


@view_config(route_name='home', renderer='lesnet:templates/evaluate.jinja2')
def home_view(request):
    return {'project': 'LesNet'}

@view_config(route_name='supported-diagnoses', renderer='lesnet:templates/supported-diagnoses.jinja2')
def supported_diagnoses_view(request):
    return {'project': 'LesNet'}
