#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : websocket.py
# @Software: PyCharm


class G:
    def __init__(self, max_cache: int = 100):
        self.cache = []
        self.max_cache = max_cache 