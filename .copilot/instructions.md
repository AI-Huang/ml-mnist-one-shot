# Project Copilot Instructions

- 项目文档统一放在 `docs/` 目录下。
- 不要把新的说明文档、研究记录或数据集构造文档放到 `datasets/`、`report/` 或代码目录中,除非用户明确要求。
- 涉及数据集构造、实验记录、研究综述等 Markdown 文档时,优先新建或更新 `docs/*.md`。
- 修改 `oneshot` CLI、训练入口、数据集构造或可视化训练样本流程时,同步更新 `docs/USAGE.md` 中的 `uv run oneshot` 命令说明。