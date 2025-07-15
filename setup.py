from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name='house_price_cleaner',
    ext_modules=cythonize([
        Extension("house_price_cleaner.cleaner", ["house_price_cleaner/cleaner.pyx"])
    ]),
    packages=['house_price_cleaner'],
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'category_encoders',
        'xgboost'
    ],
    zip_safe=False,
)

