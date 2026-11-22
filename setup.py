# setup.py: only the wheel tag; everything else is in pyproject.toml.
#
# symjit's Rust library (symjit/_lib*) is a plain cdylib that python/symjit/engine.py loads with
# ctypes: it links to no libpython and uses no Python API. A wheel built with one Python therefore
# works with every Python that pyproject.toml's requires-python allows, and is tagged
# py3-none-<platform> instead of cpXY-cpXY-<platform>. The platform part (linux_x86_64, win_amd64,
# macosx_11_0_arm64, ...) stays as setuptools computes it; on Linux, `auditwheel repair` then turns
# linux_<arch> into the matching manylinux tag (see .github/workflows/wheels.yml).
from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel


class bdist_wheel_py3(bdist_wheel):
    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(cmdclass={"bdist_wheel": bdist_wheel_py3})
