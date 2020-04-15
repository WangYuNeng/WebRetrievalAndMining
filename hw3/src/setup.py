from setuptools import setup
from Cython.Build import cythonize

setup(
    name='model',
    ext_modules=cythonize("model.pyx", annotate=True),
    zip_safe=False,
)