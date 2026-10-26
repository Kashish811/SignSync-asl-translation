"""Runs Uni-Sign's demo/online_inference.py with placeholder modules for training-only packages.

Usage (from the Uni-Sign folder):  python run_demo.py <video> <checkpoint>
The demo imports deepspeed / decord at module level, but inference does not use them.
"""
import importlib.machinery
import os
import runpy
import sys
import types
from unittest import mock


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return mock.MagicMock()


for _m in ("deepspeed", "deepspeed.comm", "deepspeed.accelerator", "decord"):
    try:
        __import__(_m)                                    # use the real package if it is installed
    except Exception:
        stub = _Stub(_m)
        stub.__spec__ = importlib.machinery.ModuleSpec(_m, None)
        stub.__path__ = []
        sys.modules[_m] = stub
        if "." in _m:
            parent, child = _m.rsplit(".", 1)
            setattr(sys.modules[parent], child, stub)

video, ckpt = sys.argv[1], sys.argv[2]
root = os.getcwd()                                         # the Uni-Sign folder
sys.path[:0] = [root, os.path.join(root, "demo")]          # same import paths as `python demo/online_inference.py`
sys.argv = ["online_inference.py", "--online_video", video, "--finetune", ckpt]
runpy.run_path(os.path.join(root, "demo", "online_inference.py"), run_name="__main__")
