# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# Sphinx makefile with api docs generation
# (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
# (2) make.py generate Python api docs, one ".rst" file per class / function
# (3) make.py calls the actual `sphinx-build`

from __future__ import print_function
import argparse
import subprocess
import sys
import multiprocessing
import importlib
import os
from inspect import getmembers, isbuiltin, isclass, ismodule
import shutil
import warnings
import weakref
from tempfile import mkdtemp


def _create_or_clear_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Removed directory %s" % dir_path)
    os.makedirs(dir_path)
    print("Created directory %s" % dir_path)


class TemporaryDirectory(object):
    """
    Porting Python 3's TemporaryDirectory context manager to Python 2
    https://github.com/python/cpython/blob/master/Lib/tempfile.py
    """

    def __init__(self, suffix=None, prefix=None, dir=None):
        self.name = mkdtemp(suffix, prefix, dir)
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self.name,
            warn_message="Implicitly cleaning up {!r}".format(self))

    @classmethod
    def _rmtree(cls, name):

        def onerror(func, path, exc_info):
            if issubclass(exc_info[0], PermissionError):

                def resetperms(path):
                    try:
                        os.chflags(path, 0)
                    except AttributeError:
                        pass
                    os.chmod(path, 0o700)

                try:
                    if path != name:
                        resetperms(os.path.dirname(path))
                    resetperms(path)

                    try:
                        os.unlink(path)
                    # PermissionError is raised on FreeBSD for directories
                    except (IsADirectoryError, PermissionError):
                        cls._rmtree(path)
                except FileNotFoundError:
                    pass
            elif issubclass(exc_info[0], FileNotFoundError):
                pass
            else:
                raise ()

        shutil.rmtree(name, onerror=onerror)

    @classmethod
    def _cleanup(cls, name, warn_message):
        cls._rmtree(name)
        warnings.warn(warn_message, ResourceWarning)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def cleanup(self):
        if self._finalizer.detach():
            self._rmtree(self.name)


class PyAPIDocsBuilder:
    """
    Generate Python API *.rst files, per (sub) module, per class, per function.
    The file name is the full module name.

    E.g. If output_dir == "python_api", the following files are generated:
    python_api/open3d.camera.rst
    python_api/open3d.camera.PinholeCameraIntrinsic.rst
    ...
    """

    def __init__(self, output_dir, c_module, c_module_relative):
        self.output_dir = output_dir
        self.c_module = c_module
        self.c_module_relative = c_module_relative
        print("Generating *.rst Python API docs in directory: %s" %
              self.output_dir)

    def generate_rst(self):
        _create_or_clear_dir(self.output_dir)

        main_c_module = importlib.import_module(self.c_module)
        sub_module_names = sorted(
            [obj[0] for obj in getmembers(main_c_module) if ismodule(obj[1])])
        for sub_module_name in sub_module_names:
            PyAPIDocsBuilder._generate_sub_module_class_function_docs(
                sub_module_name, self.output_dir)

    @staticmethod
    def _generate_function_doc(sub_module_full_name, function_name,
                               output_path):
        # print("Generating docs: %s" % (output_path,))
        out_string = ""
        out_string += "%s.%s" % (sub_module_full_name, function_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name
        out_string += "\n\n" + ".. autofunction:: %s" % function_name
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_class_doc(sub_module_full_name, class_name, output_path):
        # print("Generating docs: %s" % (output_path,))
        out_string = ""
        out_string += "%s.%s" % (sub_module_full_name, class_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name
        out_string += "\n\n" + ".. autoclass:: %s" % class_name
        out_string += "\n    :members:"
        out_string += "\n    :undoc-members:"
        out_string += "\n    :inherited-members:"
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_sub_module_doc(sub_module_name, class_names, function_names,
                                 sub_module_doc_path):
        # print("Generating docs: %s" % (sub_module_doc_path,))
        class_names = sorted(class_names)
        function_names = sorted(function_names)
        sub_module_full_name = "open3d.%s" % (sub_module_name,)
        out_string = ""
        out_string += sub_module_full_name
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % sub_module_full_name

        if len(class_names) > 0:
            out_string += "\n\n**Classes**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for class_name in class_names:
                out_string += "\n    " + "%s" % (class_name,)
            out_string += "\n"

        if len(function_names) > 0:
            out_string += "\n\n**Functions**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for function_name in function_names:
                out_string += "\n    " + "%s" % (function_name,)
            out_string += "\n"

        obj_names = class_names + function_names
        if len(obj_names) > 0:
            out_string += "\n\n.. toctree::"
            out_string += "\n    :hidden:"
            out_string += "\n"
            for obj_name in obj_names:
                out_string += "\n    %s <%s.%s>" % (
                    obj_name,
                    sub_module_full_name,
                    obj_name,
                )
            out_string += "\n"

        with open(sub_module_doc_path, "w") as f:
            f.write(out_string)

    @staticmethod
    def _generate_sub_module_class_function_docs(sub_module_name, output_dir):
        sub_module = importlib.import_module("open3d.open3d.%s" %
                                             (sub_module_name,))
        sub_module_full_name = "open3d.%s" % (sub_module_name,)
        print("Generating docs for submodule: %s" % sub_module_full_name)

        # Class docs
        class_names = [
            obj[0] for obj in getmembers(sub_module) if isclass(obj[1])
        ]
        for class_name in class_names:
            file_name = "%s.%s.rst" % (sub_module_full_name, class_name)
            output_path = os.path.join(output_dir, file_name)
            PyAPIDocsBuilder._generate_class_doc(sub_module_full_name,
                                                 class_name, output_path)

        # Function docs
        function_names = [
            obj[0] for obj in getmembers(sub_module) if isbuiltin(obj[1])
        ]
        for function_name in function_names:
            file_name = "%s.%s.rst" % (sub_module_full_name, function_name)
            output_path = os.path.join(output_dir, file_name)
            PyAPIDocsBuilder._generate_function_doc(sub_module_full_name,
                                                    function_name, output_path)

        # Submodule docs
        sub_module_doc_path = os.path.join(output_dir,
                                           sub_module_full_name + ".rst")
        PyAPIDocsBuilder._generate_sub_module_doc(sub_module_name, class_names,
                                                  function_names,
                                                  sub_module_doc_path)


class SphinxDocsBuilder:
    """
    SphinxDocsBuilder calls Python api docs generation and then calls
    sphinx-build:

    (1) The user call `make *` (e.g. `make html`) gets forwarded to make.py
    (2) Calls PyAPIDocsBuilder to generate Python api docs rst files
    (3) Calls `sphinx-build` with the user argument
    """

    def __init__(self, html_output_dir):
        # Hard-coded parameters for Python API docs generation for now
        # Directory structure for the Open3D Python package:
        # open3d
        # - __init__.py
        # - open3d.so  # Actual name depends on OS and Python version
        self.c_module = "open3d.open3d"  # Points to the open3d.so
        self.c_module_relative = "open3d"  # The relative module reference to open3d.so
        self.python_api_output_dir = "python_api"
        self.html_output_dir = html_output_dir

    def run(self):
        self._gen_python_api_docs()
        self._run_sphinx()

    def _gen_python_api_docs(self):
        """
        Generate Python docs.
        Each module, class and function gets one .rst file.
        """
        # self.python_api_output_dir cannot be a temp dir, since other
        # "*.rst" files reference it
        pd = PyAPIDocsBuilder(self.python_api_output_dir, self.c_module,
                              self.c_module_relative)
        pd.generate_rst()

    def _run_sphinx(self):
        """
        Call Sphinx command with hard-coded "html" target
        """
        with TemporaryDirectory() as build_dir:
            cmd = [
                "sphinx-build",
                "-M",
                "html",
                ".",
                build_dir,
                "-j",
                str(multiprocessing.cpu_count()),
            ]
            print('Calling: "%s"' % " ".join(cmd))
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
            shutil.copytree(os.path.join(build_dir, "html"),
                            os.path.join(self.html_output_dir, "html"))


class DoxygenDocsBuilder:

    def __init__(self, html_output_dir):
        self.html_output_dir = html_output_dir

    def run(self):
        doxygen_temp_dir = "doxygen"
        _create_or_clear_dir(doxygen_temp_dir)

        cmd = ["doxygen", "Doxyfile"]
        print('Calling: "%s"' % " ".join(cmd))
        subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
        shutil.copytree(os.path.join("doxygen", "html"),
                        os.path.join(self.html_output_dir, "html", "cpp_api"))

        if os.path.exists(doxygen_temp_dir):
            shutil.rmtree(doxygen_temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sphinx",
                        dest="build_sphinx",
                        action="store_true",
                        default=False,
                        help="Build Sphinx for main docs and Python API docs.")
    parser.add_argument("--doxygen",
                        dest="build_doxygen",
                        action="store_true",
                        default=False,
                        help="Build Doxygen for C++ API docs.")
    args = parser.parse_args()

    # Clear output dir if new docs are to be built
    html_output_dir = "_out"
    _create_or_clear_dir(html_output_dir)

    # Sphinx is hard-coded to build with the "html" option
    # To customize build, run sphinx-build manually
    if args.build_sphinx:
        print("Sphinx build enabled")
        sdb = SphinxDocsBuilder(html_output_dir)
        sdb.run()
    else:
        print("Sphinx build disabled, use --sphinx to enable")

    # Doxygen is hard-coded to build with default option
    # To customize build, customize Doxyfile or run doxygen manually
    if args.build_doxygen:
        print("Doxygen build enabled")
        ddb = DoxygenDocsBuilder(html_output_dir)
        ddb.run()
    else:
        print("Doxygen build disabled, use --doxygen to enable")
