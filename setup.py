import os
import platform
from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md')) as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt')) as f:
    CHANGES = f.read()

# Check if the system is MacOS with ARM processor
is_macos_arm = platform.system() == 'Darwin' and platform.processor() == 'arm'

# Specify the packages based on the system
if is_macos_arm:
    requires = [
        'tensorflow-macos==2.12.0',
    ]
else:
    requires = [
        'tensorflow==2.12.0',
    ]

requires.extend([
    'plaster_pastedeploy',
    'pyramid',
    'pyramid_jinja2',
    'pyramid_debugtoolbar',
    'waitress',
    'Pillow==9.5.0',
    'requests==2.28.2',
    'scipy==1.10.1',
    'sympy==1.11.1',
    'tensorboard==2.12.2',
    'termcolor==2.2.0',
    'torch==2.0.0',
    'torchvision==0.15.1',
    'tqdm==4.65.0',
    'albumentations',

])

tests_require = [
    'WebTest',
    'pytest',
    'pytest-cov',
]

setup(
    name='skinvestigatorai',
    version='0.0.4',
    description='SkinVestigatorAI',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    author='',
    author_email='',
    url='',
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
            'main = skinvestigatorai:main',
        ],
    },
)
