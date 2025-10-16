#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
leek-core
"""

def get_version_from_pyproject():
    """从pyproject.toml文件中读取版本信息"""
    try:
        # 尝试使用Python 3.11+内置的tomllib
        import tomllib
    except ImportError:
        # 对于较老的Python版本，使用tomli
        try:
            import tomli as tomllib
        except ImportError:
            # 如果都没有，返回默认版本
            return "0.1.0"
    
    try:
        # 获取项目根目录的pyproject.toml文件路径
        from pathlib import Path
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent  # leek_core/ -> src/ -> leek-core/
        pyproject_path = project_root / "pyproject.toml"
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                # 优先读取 PEP 621 的 [project].version，如不存在则回退到 Poetry 的 [tool.poetry].version
                version = (
                    data.get("project", {}).get("version")
                    or data.get("tool", {}).get("poetry", {}).get("version")
                    or "0.1.0"
                )
                return version
        else:
            return "0.1.0"
    except Exception:
        # 如果读取失败，返回默认版本
        return "0.1.0"

__version__ = get_version_from_pyproject()
