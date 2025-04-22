#!/usr/bin/env python
# -*- coding: utf-8 -*-

class classproperty:
    def __init__(self, func):
        self.func = func
    def __get__(self, instance, owner=None):
        return self.func(owner if owner is not None else type(instance))

if __name__ == '__main__':
    pass
