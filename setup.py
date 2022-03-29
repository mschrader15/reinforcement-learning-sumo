"""
This was taken almost verbatim (besides package name and info) from FLOW

https://github.com/flow-project/flow/blob/master/setup.py
"""

from os.path import dirname, realpath
from setuptools import find_packages, setup, Distribution


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


class BinaryDistribution(Distribution):
    """See parent class."""

    def has_ext_modules(self):
        """Return True for external modules."""
        return True


setup(
    name='rl_sumo',
    version="1.0.3",
    distclass=BinaryDistribution,
    # cmdclass={"build_ext": build_ext},
    packages=find_packages(),
    description=("Univ. of Alabama ME691 Reinforcement Learning Project"),
    # long_description=open("README.md").read(),
    url="https://github.com/mschrader15/reinforcement-learning-sumo",
    keywords=("autonomous vehicles intelligent-traffic-control"
              "reinforcement-learning deep-learning python"),
    install_requires=_read_requirements_file(),
    zip_safe=False,
)
