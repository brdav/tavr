from setuptools import setup, find_packages


setup(
    name='tavi',
    version='1.0',
    description='',
    author='brdav',
    url='https://github.com/brdav/tavi',
    packages=find_packages(),
    install_requires=['pytorch-lightning'],
    python_requires=">=3.8.0"
)
