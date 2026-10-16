# 本地安装与使用

无需公网部署或租用服务器。每位使用者在本机运行，用自己的 API 凭据访问外部模型与路线服务。

已验证环境：Windows、Python 3.12、NVIDIA CUDA。Embedding 使用 CPU，默认预训练 reranker 使用 GPU。最低显存未系统测量，尚未验收纯 CPU 及其他平台配置。

## 1. 获取项目与创建环境

从 GitHub 下载 ZIP 并解压，或克隆仓库。在项目目录打开终端，确认 `python --version` 是 3.12，再执行：

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-runtime.lock.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation
.venv\Scripts\python.exe -m pip check
```

PowerShell、CMD、Anaconda Prompt 均可执行以上命令，无需激活环境或修改 PowerShell 执行策略。若使用 Anaconda，可先 `conda create -n hk-taxi-bootstrap python=3.12 -y`、`conda activate hk-taxi-bootstrap` 提供 Python 3.12，再执行以上步骤；应用实际运行在项目 `.venv`。

## 2. 填写自己的服务凭据

PowerShell：

```powershell
Copy-Item .env.example .env
```

CMD / Anaconda Prompt：

```bat
copy .env.example .env
```

用文本编辑器填写 `.env`：

- `DASHSCOPE_API_KEY`：自己的阿里云百炼 Key。
- `DASHSCOPE_BASE_URL`：与 Key 所属地域及业务空间对应的兼容接口地址。
- `GENERATION_MODEL`：账号实际可用的模型 ID。
- `ORS_API_KEY`：自己的 openrouteservice Key，用于实时路线请求。

不提交 `.env`。外部服务的权限、免费额度和计费取决于使用者自己的账号；初始化不发送 Qwen 或 ORS 请求。

## 3. 一次性初始化

```bat
.venv\Scripts\python.exe scripts/setup.py
```

仓库已附带全部当前运行输入。该命令校验输入、下载固定版本 BGE-M3 和 BGE Reranker v2 M3、重建 SQL 数据库并在本地为 64 块语料生成向量与索引。无需下载原始文档、OCR、重新切分、标注或提供额外输入包。

两套必要模型文件约 4.59 GB；Python、CUDA 依赖及缓存另占磁盘空间。模型从 Hugging Face 下载，需要相应网络连通性。若已有完全相同版本模型，可显式使用缓存：

```bat
.venv\Scripts\python.exe scripts/setup.py --model-cache <缓存根目录>
```

缓存目录须含 `models/<模型>/<revision>`，各文件仍校验 hash。重复初始化会验证已有产物并跳过不必要的重建；没有文档下载或源码 OCR 恢复分支。

## 4. 启动聊天页面

PowerShell：

```powershell
.\start_web.cmd
```

CMD / Anaconda Prompt：

```bat
start_web.cmd
```

打开 `http://127.0.0.1:8501/`。修改端口：`start_web.cmd --server.port 8502`。停止：终端 Ctrl+C。

也可直接启动：

```bat
.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.port 8501
```

终端对话：

```bat
.venv\Scripts\python.exe scripts/chat.py --session demo --interactive
```

页面与 CLI 默认启用预训练重排。个人会话与导出保存在被忽略的 `artifacts/`。

## 5. 检查与开发

```bat
.venv\Scripts\python.exe scripts/doctor.py --models --credentials
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts/evaluate.py
```

doctor 仅检查本地配置和文件，不测试真实 API 连接。测试阻断外部网络，允许本机回环；检索评测在本地生成查询向量并执行 GPU 重排。需要重建生成物时运行 `scripts/prepare_data.py --force`，不改动语料或检索设置。
