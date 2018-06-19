from setuptools import setup, find_packages


setup(
    name='hierarchical_attention',
    version='0',
    description='Hierarchical Attention Network',
    url='https://www.github.com/rakreek/hierarchical_attention',
    author='Andy Kreek, Kourtney Traina, Paul Koester',
    packages=find_packages(),
    install_requires=[
        'commonregex',
        'gensim',
        'stemming',
        'keras>=2.1',
        'tensorflow',
        'wget',
        'nltk',
        'scipy',
        'lxml',
        'numpy',
        'matplotlib',
        'pandas',
        'seaborn',
        'nose-exclude',
        'coverage',
    ],
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True)
