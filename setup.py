import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.md'), 'r', encoding='utf-8', newline='') as f:
    README = f.read()
with open(os.path.join(here, 'CHANGES.txt'), 'r', encoding='utf-8', newline='') as f:
    CHANGES = f.read()

requires = [
    'pyramid==2.1',
    'pyramid_jinja2==2.10.1',
    'pyramid_debugtoolbar==4.12.1',
    'waitress==3.0.2',
    'tensorflow==2.21.0',
    'tensorboard==2.20.0',
    'numpy==2.4.6',
    'Pillow==12.3.0',
    'requests==2.34.2',
    'scipy==1.17.1',
    'scikit-learn==1.9.0',
    'tqdm==4.68.2',
    'opencv-python-headless==4.13.0.92',
]

tests_require = [
    'WebTest==3.0.7',
    'pytest==9.1.0',
    'pytest-cov==7.1.0',
    'pytest-mock==3.15.1',
    'ruff==0.15.17',
]

# JEPA world-model pretraining (lesnet/jepa/) — isolated PyTorch subsystem.
# Optional so TF-only users don't pull torch: pip install -e ".[jepa]"
jepa_require = [
    'torch>=2.4',
    'torchvision>=0.19',
    'tensorboard>=2.16',
    'h5py>=3.11',            # full-archive HDF5 image bundle (~500k ISIC images in one file)
    'onnx>=1.16',            # ONNX export of the context encoder
    'onnxscript>=0.1',       # required by torch.onnx.export (dynamo exporter)
    'onnxconverter-common>=1.14',  # fp16 quantisation tier (optional; export skips it if absent)
    'onnxruntime>=1.18',     # the 512 MB deployment runtime (not torch)
    'psutil>=5.9',           # measured inference-RSS budget gate
]

setup(
    name='lesnet',
    version='5.0.0',
    description='LesNet',
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Framework :: Pyramid',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Application',
    ],
    python_requires='>=3.11,<3.13',
    author='Thomas Behan',
    author_email='https://github.com/Thomasbehan',
    url='https://lesnet.onrender.com/',
    keywords='web pyramid pylons',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=False,
    extras_require={
        'testing': tests_require,
        'jepa': jepa_require,
    },
    install_requires=requires,
    entry_points={
        'paste.app_factory': [
            'main = lesnet:main',
        ],
    },
)
