from __future__ import print_function
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="caner",
    version="0.1.0",
    author="",
    author_email="",
    description="Cluster Adversarial Based Named Entity Recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anonymous4submit/caner",
    packages=find_packages(),
    data_files=[('doc', ['README.md'])],
    include_package_data=True,
    classifiers=[
        "Environment :: Web Environment",
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese',
        'Operating System :: MacOS',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Topic :: NLP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
            'gensim==3.4.0',
            'tensorflow-gpu==2.3.1',
            'scikit-learn>=0.20.0',
            'pandas>=0.23.3',
            'numpy>=1.14.3',
            'fasttext>=0.8.3',
            'Cython>=0.28.5',
            'flair==0.4.0'
        ],
    zip_safe=True,
)