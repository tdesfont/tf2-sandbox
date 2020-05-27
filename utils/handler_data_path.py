"""
Automate cross-platform local access to data
"""

import os
import platform
import pdb


def identify_platform():
    signature = {"os": os.name, "system": platform.system(), "release": platform.release()}
    return signature


def get_data_path():
    signature = identify_platform()
    if signature == {'os': 'posix', 'system': 'Darwin', 'release': '19.4.0'}:
        return "/Users/thibaultdesfontaines/data/"


if __name__ == "__main__":
    print(identify_platform())
    print(get_data_path())