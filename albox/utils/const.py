#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/26

@author: Jonathan Zhang <jz1248it@gmail.com>
"""


class _const:
    """
    Constant support
    """

    class ConstError(TypeError):
        pass

    class ConstNameError(TypeError):
        pass

    def setattr(self, key, value):
        # self.dict
        if key in self.__dict__:
            raise self.ConstError(f"Cannot assign constant `{key}` to another value")
        if not key.isupper():
            raise self.ConstNameError(f"constant name `{key}` is not all upper case!")
        self.__dict__[key] = value

import sys

sys.modules[__name__] = _const()
