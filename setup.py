from setuptools import setup, find_packages

# requirements.txt を読み込む関数
def load_requirements(file_name):
    with open(file_name, 'r') as f:
        return f.read().splitlines()

# パッケージのメタデータと依存関係を定義
setup(
    name='abinitioTools',  # パッケージ名
    version='0.1',             # バージョン
    packages=find_packages(),   # パッケージの自動検出
    install_requires=load_requirements('requirements.txt'),  # 依存関係を指定
    author='Naoki Kitamura',         # 作者情報
    url='https://example.com/your_package',  # プロジェクトのURL
)
