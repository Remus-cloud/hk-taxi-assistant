# 随仓库提供的数据与本地索引

## 已附带的运行输入

| 位置 | 内容 |
|---|---|
| `data/curated/chunks.jsonl` | 64 块已审阅、清理及切分的文档文本 |
| `data/curated/content_types.json` | 内容类型标签 |
| `data/curated/topics.json` | 主题保护标签 |
| `data/curated/source_scope.annotations.json` | 来源适用范围说明 |
| `data/curated/taxi_stands.json` | 518 条站点记录 |
| `data/reference/passenger_facilities_v2.json` | 20 条候车设施 |
| `data/reference/structured_rules.json` | 结构化事实规则与证据映射 |
| `data/reference/evidence/` | 规则校验所需的提取证据 |
| `data/reference/harbour_geometry_v1.json` | 隧道线形及来源信息 |
| `data/manifests/sources.json` | 文档来源、定位及 chunk ID |
| `data/manifests/runtime_inputs.json` | 当前输入版本与逐文件 SHA256 |

这些数据随仓库提供，不要求使用者自行收集或处理。原始网页与 PDF、历史语料快照及已舍弃实验不属于运行必需文件，不附带。

## 切分与检索

语料已按标题、问答及完整规则组织为单元，仅对长单元递归字符切分；首轮目标上限为 600 个 Unicode 字符、普通长段重叠 80 字符，收费例外与适用条件尽量保留在一起。启动时直接读取当前已审阅文本，不再次切分，也不调用 LLM 重新生成语料。

初始化在本地用固定版本 BGE-M3 生成 64 块向量，构建 BM25 输入及 SQL 表。用户提问时，系统生成查询向量，结合 BM25、RRF、预训练 reranker 和主题保护选择证据。使用者无需设置语义匹配或自行标注。

## 生成物与更新

- `data/processed/`：本机数据库及生成物校验清单。
- `artifacts/retrieval_index/`：本机语料向量与索引清单。
- `models/`：固定版本的模型文件。

以上由 `python scripts/setup.py` 创建并由 Git 忽略。不同机器无需生成完全相同的 DuckDB 二进制；输入版本和本机产物分别校验。

当前数据库覆盖站点、设施和已实现的收费／隧道／事实规则。等待及其他道路收费仍未开启。后续更新语料时，应同步维护来源、事实标签和输入 manifest，并重建索引、执行回归评测。
