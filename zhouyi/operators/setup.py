import datetime
from setuptools import setup
from distutils.core import setup
from setuptools import find_packages, find_namespace_packages
import sys


def parse_arg_remove_string(argv, arg_name_equal):
    arg_value = None
    for arg in sys.argv[1:]:
        if arg.startswith(arg_name_equal):
            arg_value = arg[len(arg_name_equal) :]
            sys.argv.remove(arg)
            break

    return arg_value

arch = parse_arg_remove_string(sys.argv, "--arch=")
package_data = {"ZhouyiOperators": ["operators/*"]}
__version__ = str(datetime.datetime.now().date().strftime("%y.%#m.%#d"))
data_files = []

setup(
    name=f"ZhouyiOperators",
    author="ArmChina",
    version=__version__,
    description="Zhouyi NPU operators",
    packages=find_namespace_packages(),
    package_dir={"ZhouyiOperators": "ZhouyiOperators"},
    include_package_data=True,
    package_data=package_data,
    data_files=data_files,
)
