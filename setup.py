import sys
from os import path
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='rct',
    version='0.0.1',
    description='MTL Random Control Trial',
    license='MIT',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    install_requires=['sh', 'numpy', 'torch', 'torchvision', 'scipy'],
    extras_require={
        'dev': ['ipython', 'ipdb']
    },
)