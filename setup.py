from setuptools import setup, find_packages


setup(name='ml_tseries',
      version='0.1.0',
      description='Machine Learning utils for time series',
      packages=find_packages(include=['ml_tseries']),
      install_requires=[
          'numpy',
          'pandas',
          'sklearn',
          'scipy',
          'joblib',
          'hyperopt',
          'sklearn',
          'Sphinx==3.0.0',
          'sphinx_theme',
          'sphinx-autodoc-typehints'
      ],
      python_requires='>=3.6',
      )
