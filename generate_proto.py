#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成 protobuf 代码的脚本
"""

import os
import subprocess
import sys

def generate_proto():
    """生成 protobuf 代码"""
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    proto_dir = os.path.join(current_dir, "src", "leek_core", "engine", "grpc")
    
    # 检查 protobuf 文件是否存在
    proto_file = os.path.join(proto_dir, "engine.proto")
    if not os.path.exists(proto_file):
        print(f"protobuf 文件不存在: {proto_file}")
        return False
    
    try:
        # 生成 Python 代码
        cmd = [
            "python", "-m", "grpc_tools.protoc",
            f"--python_out={proto_dir}",
            f"--grpc_python_out={proto_dir}",
            f"--proto_path={proto_dir}",
            "engine.proto"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=proto_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("protobuf 代码生成成功")
            print(f"生成的文件:")
            print(f"  - {proto_dir}/engine_pb2.py")
            print(f"  - {proto_dir}/engine_pb2_grpc.py")
            return True
        else:
            print(f"生成失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"生成过程中出错: {e}")
        return False

if __name__ == "__main__":
    success = generate_proto()
    sys.exit(0 if success else 1) 