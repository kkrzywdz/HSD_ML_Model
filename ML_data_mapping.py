# ML_dictionary.py

"""
Dictionary Data Module

This module contains a lists with data used in the application.
"""

# No imports needed in this file

__version__ = '0.8.0'

# OS mapping
os_mapping = {
    'linux.redhat': ['redhat', 'centos', 'fedora'],
    'linux.ubuntu': ['ubuntu', 'solaris', 'open_source', 'sles'],
    'linux.windriver': ['windriver'],
    'linux.yocto': ['yocto'],
    'linux.rtos': ['rtos'],
    'windows': ['windows', 'azure stack', 'vibranium', 'chromium'],
    'freebsd': ['freebsd', 'FreeBSD'],
    'esxi': ['esx', 'vmware'],
    'freebsd': ['freebsd'],
    'linux': ['linux'],
    'preboot': ['preos', 'preboot', 'uefi'],
    'android': ['android'],
    'macos': ['macos', 'ios', 'chrome'],
    'agnostic': ['none', 'agnostic', 'bare_metal'],
    'other': ['other', 'N/A']
}