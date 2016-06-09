#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

long_description = "Pupil tracking library for mouse pupil"

setup(
    name='pupil_tracking',
    version='0.1.0.dev1',
    description="Pupil tracker library.",
    long_description=long_description,
    author='Jugnu Agrawal Fabian Sinz',
    author_email='jugnu.ag.jsr@gmail.com sinz@bcm.edu',
    license="Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License",
    url='https://github.com/cajal/pupil-tracking',
    keywords='eyetracker',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License',
        'Topic :: Database :: Front-Ends',
    ],
)
