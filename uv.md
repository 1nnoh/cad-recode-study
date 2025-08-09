我已经创建了requirements.txt文件。现在你可以使用以下命令来
  用uv安装相同的环境：

  # 创建虚拟环境
  uv venv

  # 激活虚拟环境
  source .venv/bin/activate

  # 安装依赖 (使用--no-deps参数，就像Dockerfile中一样)
  uv pip install --no-deps -r requirements.txt

  注意：你还需要手动安装Open3D，因为Dockerfile中是从源码编译
  的。你可以尝试：

  # 简化方式安装Open3D
  uv pip install open3d

  或者如果需要headless渲染，可以参考Dockerfile中的编译方式手
  动构建Open3D。