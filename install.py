"""安装所有依赖，自动检测缺失的包并安装。

用法: python install.py
"""

import importlib
import subprocess
import sys

# (import_name, pip_name) — import 名和 pip 包名不同的要分开写
DEPS = [
    ("yaml", "pyyaml"),
    ("fastapi", "fastapi>=0.115.0"),
    ("uvicorn", "uvicorn[standard]>=0.30.0"),
    ("chromadb", "chromadb>=0.5.0"),
    ("jieba", "jieba>=0.42.1"),
    ("httpx", "httpx>=0.27.0"),
    ("pydantic", "pydantic>=2.0"),
    ("tiktoken", "tiktoken>=0.7.0"),
    ("numpy", "numpy"),
    ("torch", "torch>=2.0.0"),
    ("sentence_transformers", "sentence-transformers>=3.0.0"),
    # FlagEmbedding 和 transformers 5.x 有冲突，reranker 可选
    # ("FlagEmbedding", "FlagEmbedding>=1.2.0"),
    ("pytest", "pytest>=8.0"),
]


def check_and_install():
    missing = []
    for import_name, pip_name in DEPS:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))

    if not missing:
        print("所有依赖已安装。")
        return

    print(f"缺失 {len(missing)} 个包:")
    for import_name, pip_name in missing:
        print(f"  {import_name} → pip install {pip_name}")

    print("\n开始安装...\n")
    pip_names = [pip_name for _, pip_name in missing]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install"] + pip_names,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n安装失败 (exit code {result.returncode})")
        sys.exit(1)

    # 验证
    print("\n验证安装...")
    still_missing = []
    for import_name, pip_name in missing:
        try:
            importlib.import_module(import_name)
            print(f"  OK {import_name}")
        except ImportError:
            print(f"  FAIL {import_name}")
            still_missing.append(import_name)

    if still_missing:
        print(f"\n仍然缺失: {still_missing}")
        sys.exit(1)
    else:
        print("\n全部安装成功。")


if __name__ == "__main__":
    check_and_install()
