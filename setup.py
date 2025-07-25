from setuptools import setup

__copyright__ = "Kilian Helfenbein"
__license__ = "MIT License"
__author__ = "Kilian Helfenbein"

setup(
    name="MWE DB Access",
    author=__author__,
    author_email="kilian.helfenbein@rl-institut.de",
    description="MWE DB Access",
    long_description="MWE DB Access",
    version="0.1.0",
    url="TODO",
    license=__license__,
    install_requires=[
        "saio",
    ],
    extras_require={
        "dev": ["black", "isort", "jupyterlab", "pre-commit", "pyupgrade"],
    },
)
