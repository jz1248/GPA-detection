#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilities that related to file access.

Created on 2021/11/27

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
from typing import List, Any
import os
import pickle


def get_file_path_with_ext(path: str, file_ext: str, recursive: bool = False, exclude_ext: bool = True) -> List[str]:
    """get path of all files which extension is `file_ext` in `path` """
    file_list = []
    for root, dirs, files in os.walk(path, topdown=True):
        if not recursive and root != path:
            continue
        for file in sorted(files):
            if os.extsep in file and file.split(os.extsep)[-1] == file_ext:
                if exclude_ext:
                    file_list.append(file[:-(len(file_ext)+1)])
                else:
                    file_list.append(file)

    return file_list


def read_txt_file_by_lines(path: str) -> List[str]:
    lines = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if len(line) > 0 and line[-1] == "\n":
                lines.append(line[:-1])
            line = f.readline()
    return lines


def read_txt_file(path: str) -> str:
    with open(path, 'r') as f:
        content = f.read()
    return content


def pickle_load(path: str) -> Any:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def pickle_dump(obj: Any, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)
