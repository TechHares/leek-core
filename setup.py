#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Leek Core 量化交易框架安装配置
"""

import os
from setuptools import setup, find_packages

# 读取README文件
with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

# 读取依赖
def parse_requirements(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

requirements = parse_requirements("requirements.txt")

# 测试依赖
test_requirements = [
    'pytest>=6.0.0',
    'pytest-cov>=2.12.0',
]

# 可选依赖
extras_require = {
    'dev': [
        'black>=22.1.0',
        'isort>=5.10.0',
        'flake8>=4.0.0',
        'mypy>=0.931',
    ],
    'docs': [
        'sphinx>=4.4.0',
        'sphinx-rtd-theme>=1.0.0',
    ],
    'backtest': [
        'vectorbt>=0.24.0',
        'pyfolio>=0.9.0',
    ],
}

setup(
    name='leek-core',
    version='0.1.0',
    description='Leek Core 量化交易引擎/组件化事件驱动框架',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='shenglin.li',
    author_email='shenglin.li@example.com',
    url='https://github.com/shenglin-li/leek-core',
    packages=find_packages(exclude=['tests', 'tests.*', 'examples', 'docs']),
    include_package_data=True,
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require=extras_require,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: Chinese (Simplified)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Framework :: AsyncIO',
        'Framework :: Sphinx',
    ],
    keywords='quantitative trading, algorithmic trading, finance, investment, event-driven, component, okx, backtest',
    entry_points={
        'console_scripts': [
            'leek=leek_core.cli:main',
        ],
    },
)
