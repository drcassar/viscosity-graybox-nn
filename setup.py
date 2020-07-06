import setuptools

setuptools.setup(
    version='1.0',
    author='Daniel Roberto Cassar',
    author_email='contact@danielcassar.com.br',
    description='Gray-box Neural Network to predict the viscosity of liquids',
    url="https://github.com/drcassar/viscosity-graybox-nn",
    packages=setuptools.find_packages(),
    install_requires=[
        'sklearn',
        'matplotlib',
        'python-ternary',
        'mendeleev==0.6.0',
        'glasspy==0.3',
        'torch==1.5.0',
        'pytorch-lightning==0.7.6',
    ],
    keywords='viscosity, liquids',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Environment :: Console",
    ],
    license='GPL',
    python_requires='>=3.6',
)
