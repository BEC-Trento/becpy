from setuptools import setup

VERSION = 'becpy.__version__'
BASE_CVS_URL = 'https://github.com/carmelom/becpy'

setup(
    name='becpy',
    packages=['becpy', ],
    version=VERSION,
    author='Carmelo Mordini',
    author_email='carmelo.mordini@unitn.it',
    install_requires=[x.strip() for x in open('requirements.txt').readlines()],
    url=BASE_CVS_URL,
    download_url='{}/tarball/{}'.format(BASE_CVS_URL, VERSION),
    test_suite='tests',
    tests_require=[x.strip() for x in open('requirements_test.txt').readlines()],
    keywords=[],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License (GPL)",
    ],
)
