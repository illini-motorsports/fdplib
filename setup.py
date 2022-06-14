from setuptools import setup, find_packages

setup(
    name = 'fdplib',
    url = 'https://github.com/illini-motorsports/fdplib',
    version = 0.7,
    author = 'cmmeyer1800',
    author_email = 'collinmmeyer@gmail.com',
    python_requires = ">=3.6",
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    packages = (find_packages()),
    install_requires=[
        "tqdm>=4.64.0"
    ]
)