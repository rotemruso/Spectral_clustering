from platform import version
from setuptools import Extension,setup,find_packages

setup(
    name='spkmeansmodule',
    version='0.1.0',
    author='Rotem Ruso & Ketty Vaisbrot',
    install_requires=['invoke'],
    packages=find_packages(),
    ext_modules=[
        Extension(
            'spkmeansmodule',
            ['spkmeansmodule.c','spkmeans.c']
        )
    ]
)