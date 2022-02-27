import subprocess
import setuptools


def get_version():
    git_command = subprocess.Popen(
        "git rev-list  `git rev-list --tags --no-walk --max-count=1`..HEAD --count",
        stdout=subprocess.PIPE,
        shell=True,
    )

    git_latest_tag_command = subprocess.Popen(
        "git describe --tags $(git rev-list --tags --max-count=1)",
        stdout=subprocess.PIPE,
        shell=True,
    )

    version = git_command.stdout.read().decode("utf-8").strip()
    latest_tag = git_latest_tag_command.stdout.read().decode("utf-8").strip()

    return ".".join([latest_tag, version])


def get_description():
    with open("README.md") as f:
        long_description = f.read()
        return long_description


requirements = [
    "pandas ~= 0.24.2",
    # "pyspark ~= 2.4.3",
]

setuptools.setup(
    name="kolang",
    version=get_version(),
    scripts=[],
    packages=setuptools.find_packages(exclude="tests"),
    include_package_data=True,
    zip_safe=True,
    install_requires=requirements,
)
