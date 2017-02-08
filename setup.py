
from distutils.core import setup

setup(
    name='CGraph',
    version=open('cgraph/__init__.py').readlines()[-1].split()[-1].strip('\''),
    description='Symbolic computation in Python',
    author='Christoph Heindl',
    url='https://github.com/cheind/py-cgraph',
    packages=['cgraph', 'cgraph.test', 'cgraph.app'],
)