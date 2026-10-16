# 来源与第三方组件

本项目仅允许个人学习及日常使用。第三方资料、模型与依赖的权利归其各自权利人所有；本项目的使用范围说明不替代第三方条款。

- 的士规则、收费及场站资料：香港运输署、香港政府及相关场站运营机构。来源 URL、正文定位及 chunk ID 见 `data/manifests/sources.json`，语料及结构化记录也保留来源字段。
- 的士站开放数据：香港政府、运输署及 DATA.GOV.HK。保留来源署名；参见 [DATA.GOV.HK 使用条款](https://data.gov.hk/en/terms-and-conditions) 及数据集具体条款。
- 隧道线形：© OpenStreetMap contributors，ODbL 1.0。保留 `data/reference/harbour_geometry_LICENSE.txt` 和线形数据中的来源字段。
- Embedding：`BAAI/bge-m3`；reranker：`BAAI/bge-reranker-v2-m3`。版本见 `configs/models.json`，模型另行下载，沿用各自模型许可。
- Python 依赖沿用各自许可；依赖锁文件不改变这些许可。

仓库附带当前运行所需的已处理语料、结构化数据及提取证据。原始网页／PDF、模型权重、API 凭据、个人对话及本机生成物不随仓库提供。
