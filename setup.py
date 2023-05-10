from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'torch==1.12.1',
    'monai',
    'numpy',
    'scikit-learn',
    'google-cloud-storage'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    setup_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='PyTorch Python Package'
)