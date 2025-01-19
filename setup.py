import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt')) as f:
    CHANGES = f.read()

requires = [
    'plaster_pastedeploy==1.0.1',
    'pyserve==0.2.8',
    'pyramid==2.0.2',
    'pyramid_jinja2==2.10.1',
    'pyramid_debugtoolbar==4.12.1',
    'waitress==3.0.2',
    'Pillow==10.3.0',
    'requests==2.32.2',
    'scipy==1.10.1',
    'scikit-learn==1.6.1',
    'tensorboard==2.16.2',
    'termcolor==2.3.0',
    'tensorflow==2.16.1',
    'torch==2.5.1',
    'torchvision==0.20.1',
    'tqdm==4.66.3',
    'vit-keras==0.1.2',
    'tensorflow_addons==0.20.0',
    'albumentations==2.0.0',
    'keras-tuner==1.4.7',
    'pandas==2.2.3',
    'sympy==1.13.1',
]

tests_require = [
    'WebTest==3.0.2',
    'pytest==8.3.4',
    'pytest-cov==6.0.0',
    'pytest-mock==3.14.0',
    'flake8==7.1.1',
    'ruff==0.9.1',
]

setup(
    name='lesnet',
    version='4.0.1',
    description='LesNet',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='Thomas Behan',
    author_email='https://github.com/Thomasbehan',
    url='https://lesnet.onrender.com/',
    keywords='web pyramid pylons',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = lesnet:main',
        ],
    },
)
