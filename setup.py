from setuptools import setup
from Cython.Build import cythonize

setup(
    name='house_price_cleaner',
    ext_modules=cythonize("house_price_cleaner/cleaner.py"),
    packages=['house_price_cleaner'],
    install_requires=[
	'pandas',
	'numpy',
	'scikit-learn',
	'category_encoders',
	'xgboost' ],
    zip_safe=False,
)

