from setuptools import setup, find_packages

setup(
    name='cmim',
    version='0.1.0',
    description='CMIM Feature Selector -  A fast sklearn friendly modern python reimplementation of the Conditional Mutual Information Maximization feature selection algorithm',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Dan Ofer',
    url='https://github.com/ddofer/CMIM',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add any other dependencies
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
