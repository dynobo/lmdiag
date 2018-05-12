from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='lmdiag',
    version='0.1.0',
    description='Diagnostic Plots for Lineare Regression Models.',
    long_description='Plots a matrix of 4 charts to diagnose linear \
                      regression models. Or plot the charts individually. \
                      Similar to lm.plot in R.',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='lm lineare regression diagnostics plot chart r matplotlib',
    url='http://github.com/dynobo/lmdiag',
    author='dynobo',
    author_email='dynobo@mailbox.org',
    license='MIT',
    packages=['lmdiag'],
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'matplotlib'
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False
)
