# ai-study

AI 学习项目，使用 uv 管理 Python 环境和依赖。目前处于初始化阶段，脚本入口为 `src/main.py`，运行后输出 `Hello World`，已添加 Transformers 5.19.0 稳定版依赖，尚未添加测试。

## 学习文档

从 [AI 学习引导](docs/学习引导.md)开始，按 12 个阶段阅读 32 篇课文和 12 篇阶段串讲，并通过练习与阶段验收检查掌握情况。课程文档集中在 `docs/lesson/`；[专用名词](docs/专用名词.md)保留在 `docs/`，供随时查阅。

## 环境要求

- 已安装 uv，且可以通过 `uv --version` 查看版本。
- Python 3.14.4：项目通过 `.python-version` 固定开发环境版本，`pyproject.toml` 声明最低支持版本为 3.14.4。

## 快速开始

以下命令均在项目根目录执行。

### 1. 准备 Python

若本机尚未安装 Python 3.14.4，可使用 uv 安装：

```bash
uv python install 3.14.4
```

### 2. 创建环境并同步依赖

```bash
uv sync --locked
```

该命令创建 `.venv` 虚拟环境，并按照 `uv.lock` 同步依赖。`--locked` 会检查锁文件是否与项目配置一致；不一致时命令报错，需要先更新锁文件。

项目按脚本方式运行，uv 管理环境和依赖，无需构建或安装项目自身。

### 3. 运行脚本

```bash
uv run python src/main.py
```

预期输出：

```text
Hello World
```

### 4. 核对 Python 版本

```bash
uv run python --version
```

预期输出为 `Python 3.14.4`。

## Transformers 版本

项目同时固定使用 PyTorch **2.14.1**，提供模型运行所需的计算后端。

项目固定使用 PyPI 发布的 Transformers **5.19.0**，与当前[官方文档](https://huggingface.co/docs/transformers/installation)的版本一致。`pyproject.toml` 固定包版本，`uv.lock` 锁定完整依赖。按照上述同步步骤即可安装。

查看已安装版本：

```bash
uv run python -c 'from importlib.metadata import version; print(version("transformers"))'
```

## 项目结构

```text
ai-study/
├── docs/
│   ├── lesson/        # 32 篇课文与12篇阶段串讲
│   ├── 学习引导.md    # 学习路线与课程入口
│   └── 专用名词.md    # 通用术语资料
├── src/
│   └── main.py        # 脚本入口
├── .gitignore         # Git 忽略规则
├── .python-version    # 开发环境的 Python 版本
├── pyproject.toml     # 项目元数据和依赖配置
├── uv.lock            # 依赖锁文件
├── README.md          # 项目说明
└── LICENSE            # 许可证
```

## 版本控制约定

- 提交源码、`pyproject.toml`、`uv.lock` 和 `.python-version`，便于复现开发环境。
- `.venv`、Python 缓存、`.env`、`.idea/` 和 `.DS_Store` 已配置忽略规则。
- Git 忽略规则不影响已进入索引的文件；这类文件需要单独从索引移除，才能停止跟踪。

## 许可证

项目采用 [MIT 许可证](LICENSE)。
