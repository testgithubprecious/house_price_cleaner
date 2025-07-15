from setuptools import setup, find_packages

setup(
    name='house_price_cleaner',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'category_encoders',
        'xgboost'
    ],
    zip_safe=False,
)

