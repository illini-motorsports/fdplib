from setuptools import setup, find_packages


setup(
    name='fdplib',
    url='https://github.com/illini-motorsports/fdplib',
    version = 0.2,
    author='cmmeyer1800',
    author_email='collinmmeyer@gmail.com',
    python_requires = ">=3.6",
    long_description=open('README').read(),
    packages= find_packages(
        where='src'
    ),
    package_dir={'': 'src'},
    install_requires=[
        "tqdm>=4.62.3"
    ]
)