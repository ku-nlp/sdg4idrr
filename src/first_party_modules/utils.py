import sys
from pdb import set_trace


class ObjectHook(dict):
    # return None if key is not found
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

    # define __(get|set)state__ for multiprocessing/DistributedDataParallel (pickling the object)
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict_):
        self.__dict__ = dict_


def debug() -> None:
    _ = sys.stdin.readlines()
    sys.stdin = open("/dev/tty")
    set_trace()
