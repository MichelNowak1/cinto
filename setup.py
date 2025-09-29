from setuptools_rust import Binding, RustExtension
from distutils.core import setup


def make_rust_extension(module_name):
    return RustExtension(
        module_name, "Cargo.toml", debug=False, binding=Binding.PyO3
    )

setup(
    name="cinto",
    version="0.1.0",
    rust_extensions=[ make_rust_extension("cinto._cinto") ],
    packages=["cinto"],
    include_package_data=True,
    setup_requires=["setuptools", "setuptools-rust"],
    zip_safe=False,
)
