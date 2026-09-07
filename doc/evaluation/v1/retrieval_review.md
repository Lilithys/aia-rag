# 全量语料的 Chunking / Embedding / Retrieval 重评

2026-09-06。本轮完成了源数据与派生产物核验、切分与缓存修复、3 种 chunking × 4 种检索配置的完整 dev 对照、BGE-M3/E5 固定输入对照、重排 smoke 筛查及完整 dev 确认，并验证了实际 Chroma Retriever。生成与 RAGAS 尚未运行，付费模型 API 调用为 0。

结论：**继续以 semantic 1024 + BGE-M3 + dense 为基线，保留两个重排候选进入生成验证。** 小块和混合检索没有全面胜出；在新语料上，重排的收益已与旧 199-chunk 实验不同。当前生产默认仍为 dense，未仅凭检索代理指标启用新策略。

## 1. 固定了什么

- 知识库：冻结 inventory 中全部 488 篇文档；源文件、解析和清洗的 hash 均已核验。
- 题目：固定 dev 100 题，其中 70 answer、20 clarify、10 编写的能力拒答题。检索主表分母始终为 70 answer；clarify 单列，refuse 只分析相似度，不据此判断生成是否拒答。
- holdout 未用于模型查询、策略选择或错误分析。验证发布包完整性时会核验其文件 hash。
- 主检索 query：全部带角色的记录历史 + 当前问题；所有主对照共用同一 query。未调用 query rewrite LLM。
- 本地模型采用缓存的固定 revision，主实验在 Mac GPU `mps:0` 运行。环境、模型配置、query/正文指纹及逐题结果已保存。

数据包的 `independent_human_review=false` 仍然成立；本轮实验没有把代理审阅变成人工标签认证。

## 2. 首先核验旧产物，再修复未来的可靠性

重新从指定源目录解析全部文档，488/488 成功，包括 24 篇扫描 PDF。重新解析与清洗的输出，分别与旧 full 文件 **488/488 逐字节一致**。使用原切分实现重建的 1,388 个 chunk，其 ID、顺序和正文也与旧 full chunks 一致。

随后重新编码这 1,388 个 chunk，所得 `.npy` 与旧 full 向量的 SHA-256 完全相同，逐元素差异为 0。因此，**没有发现现有这批 full 向量过期**。之前的 ID-only 缓存机制仍有确定的失效风险：正文变化、ID 不变时会误用旧向量；这与这批旧文件实际是否损坏是两个判断。

修复后，embedding manifest 校验正文 hash、顺序、encoder/revision、前缀、有效窗口、运行库/设备、维度、dtype 和向量文件 hash。有效旧缓存中的相同正文可按内容复用，即使 chunk ID 发生变化；无 manifest 的旧缓存必须重新编码验证。标准目录下的旧 demo 缓存会在首次初始化时自动迁移，也可以提前用 CLI 构建，避免演示时承担冷启动耗时。

Chroma 加载时核对实际文档、metadata 与向量内容；同 ID 内容变化也会刷新，维度变化则重建该派生 collection。测试覆盖损坏向量、模型版本变化、重排 IDs、旧缓存迁移和索引内容更新。

## 3. 切分与证据计分的修复

本次发现并修复：

1. token 窗口可能落在中文字符的 UTF-8 字节中间，重叠片段出现 `�`。重叠尾部和递归硬切分现在选择合法字符边界。
2. 递归切分在跨 chunk 边界时丢掉句末分隔符；现在保留标点。
3. semantic 深层递归可能丢掉从父层继承的引言；现在继续传递。
4. 源注释中的连续列表与 Markdown 项目符号造成对齐假缺失；对齐器只去除已识别的展示标记，保留数字、负号、日期范围和否定词。
5. 标题面包屑是实际送给 encoder/生成器的内容，其中可验证的源标题也计入 provenance。

chunk 的 `source_spans` 保留 `source_block_id`、块内范围与原位置。它来自源正文与独立解析文本的对齐，属于评测/引用 metadata，不拼入 embedding 文本。范围覆盖可在多个返回片段之间合并，重复片段不会补齐缺失部分。

新计分只使用 **answer-side** 注释；question/precondition-side 命中不能替代答案。金额 200/500、肯定/否定不再通过旧模糊匹配被视为同一证据。旧评测函数会拒绝新的 task-labelled 数据，避免误用历史口径。

本文“答案证据命中@k”表示至少一个注释答案块被完整覆盖；“全部块覆盖”要求覆盖该题全部注释答案块。主检索每个配置还报告注释块召回与注释证据精确率。这些是本地代理指标，生成正确率、Faithfulness 和 RAGAS Context Precision 尚无新结果。

严格块对齐也有限制：注释块可能大于真正必要的答案事实。全语料 16,871 个源块中，16,046 个可在清洗文本中精确对齐；263 个在清洗前存在、清洗后不再匹配，另有 562 个存在解析或格式对齐差异。它们不能全部解释成事实丢失。semantic 基线的 70 道回答题中，4 题的完整注释覆盖存在此类限制，且来自同一份中文扫描 PDF；例如 OCR 把“案卷”识别成“案什”。这些题仍保留在分母中，不能以这个小切片推断所有 OCR 文档的表现。

## 4. Chunking 对照：看 k，也看相同 token 预算

修复后，所有配置仍覆盖 488 文档。size 是目标值，实际大小受章节、标题和 overlap 影响，不能假设配置 512 就严格小于 512 个模型 tokens。

| 配置，均使用 BGE-M3 dense | chunks | 命中@5 | 命中@10 | @10 平均返回 tokens | 3,000-token 预算命中 | 6,000-token 预算命中 |
|---|---:|---:|---:|---:|---:|---:|
| semantic 1024 / overlap 0 | 1,385 | 55.7% | **65.7%** | 5,566 | 55.7% | **65.7%** |
| semantic 512 / overlap 0 | 2,482 | 54.3% | 58.6% | **3,104** | **58.6%** | 61.4% |
| recursive 512 / overlap 64 | 1,859 | 51.4% | 60.0% | 3,939 | 55.7% | **65.7%** |

token 预算使用排序前缀，遇到下一块会超出预算即停止，最多取 top20；不会根据 gold 跳过片段。tokens 使用项目的 cl100k_base 统计，模型输入另外由各自 tokenizer 检查。

semantic 1024 在固定 k 下表现更好；semantic 512 在 3,000-token 预算下多命中 2/70 题，差距尚小。6,000-token 预算下 semantic 1024 与 recursive 同为 46/70 命中。当前证据不足以把所有配置切到小块；semantic 1024 是合适的继续比较起点。

修复后的 semantic 1024 复用了 1,375 行向量，仅重算 10 行；semantic 512 复用 2,426 行，仅重算 56 行。内容缓存使小修复不必触发全库重算。

## 5. Dense / BM25 / Hybrid

以 semantic 1024、同一 dev、top10 比较：

| 检索配置 | 答案证据命中@10 |
|---|---:|
| Dense | **46/70，65.7%** |
| BM25 | 22/70，31.4% |
| RRF，dense 权重 0.5 | 32/70，45.7% |
| RRF，dense 权重 0.8 | 43/70，61.4% |

混合对照各取最多 50 个候选，排除 BM25 非正分候选，再按 RRF 融合。两个小块配置下也未发现 hybrid 超过对应 dense。该结论适用于本轮配置和 dev，不代表所有融合方式都无效。

dense top10 的 24 道未完整命中题中：4 道有注释对齐限制，17 道没有检索到答案所在文档，3 道找到了相关文档但未覆盖完整答案证据。因此，下一轮 query/检索优化应优先分析那 17 道文档级失配，而不是继续扩大 chunking 网格。

## 6. 编码器公平对照

从修复后的 semantic 512 出发，两个模型接收完全相同的 passage 文本；超长部分统一限制到前 448 个 E5 tokens，304/2,482 个片段发生变化。两者都使用各自规定的 query/passage 前缀。query 预设上限为最近 384 个 E5 tokens，但实际 100 个 query 均未触及这个限制，完整历史仍被保留。

实测两个模型的 passage/query **截断数都为 0**。这组比较控制了有效可见文本，但其 passage 视图与上一节主实验不同，不能跨表直接解释成模型替换收益。

| 相同输入视图 | 命中@10，70 题 | 同语言，21 题 | 跨语言，26 题 | mixed，23 题 |
|---|---:|---:|---:|---:|
| BGE-M3 | **54.3%** | 66.7% | **42.3%** | **56.5%** |
| multilingual-e5-large | 34.3% | **76.2%** | 3.8% | 30.4% |

当前项目继续使用 BGE-M3 有依据。E5 在同语言切片较好、跨语言切片较弱；不能将其概括成模型完全没有跨语言能力，也不能把全部差异解释为长输入截断。

## 7. 重排：旧负结论在新条件下需要更新

先使用固定 smoke 筛查，窗口化版本在 14 道 answer 中多命中 1 题，因此扩展到完整 dev 确认。候选始终来自 semantic 1024 dense top20。两个重排版本使用相同的最近 96 个 reranker tokens 的 query；一个直接按 512-token pair 上限截断，另一个遍历 passage 的 384-token 窗口、步长 320，并取每个父 chunk 的最高分。窗口选择不使用 gold，窗口化版实测没有输入超限。

| 配置 | 命中@5 | 命中@10 | @10 平均返回 tokens | 相同 6,000-token 预算命中 | 额外重排 p90 |
|---|---:|---:|---:|---:|---:|
| 不重排 | 39/70 | 46/70 | 5,566 | 46/70 | — |
| 直接截断重排 | 41/70 | 52/70 | 6,207 | 48/70 | 0.59 秒 |
| 窗口化重排 | **45/70** | **53/70** | 6,503 | **52/70** | 1.17 秒 |

窗口化 top10 相对 dense top10 新增命中 9 题、丢失 2 题，净增 7/70。题目级 paired bootstrap 的 95% 差值区间约为 +1.4 至 +18.6 个百分点；这是开发集探索结果，未校正多配置选择和同源文档相关性。

也需要保留成本判断：直接截断重排已经获得大部分 @10 增益；窗口化在 @10 仅再多命中 1 题，但额外耗时接近翻倍。窗口化在 @5 和固定 token 预算下的收益更明显，是否采用取决于下一轮生成实验的目标。

一个有价值的省 token 候选是 **窗口化 top5**：45/70 命中、平均 3,568 tokens；dense top10 为 46/70、5,566 tokens。返回正文 tokens 减少约 36%，命中净少 1 题。尚不能把它解释为总费用减少 36% 或延迟一定下降，因为还有 prompt、输出、重排与 judge 开销。

## 8. 实际索引与能力边界

已通过真实 `get_default_retriever` 建立独立 Chroma index，并对全部 dev 逐题执行 warm、并发 1 的 query embedding + Chroma top20 检索。该阶段 p50 约 36 ms、p90 约 64 ms，初始化含模型与索引约 4.08 秒；不包含 rewrite、生成或 judge，不能作为端到端 ≤10 秒的验收结果。

与精确 cosine 排序的 top10 集合平均一致率为 99.5%，主表答案证据命中未变。这个规模下，当前实测没有显示索引近似搜索是主要质量瓶颈。

所有 90 道 answer/clarify 的 top1 相似度都 ≥0.55；10 道能力拒答题中也有 8 道 ≥0.55，仅 2 道明显域外题低于阈值。这只说明前置低相似度 gate 不会拦截那 8 道题，不代表生成模型一定会错误作答。领域相关但缺少个人/实时事实的情况仍需单独处理。

## 9. 下一轮与可复查证据

进入生成验证的三个配置为：dense top10 基线、直接截断重排 top10、窗口化重排 top5。先使用固定 smoke、相同生成模型与显式 judge/预算配置，验证证据命中是否转化为正确率、引用支持、总费用和端到端延迟。之后再决定生产配置；holdout 继续留作锁定后的验收。

本轮已完成检索侧重评；新版生成/评分执行器、Flash judge 校准仍是后续工作。这里没有新的整体正确率或 RAGAS 达标声明。

关键文件位于 `data/experiments/eval_v1/retrieval_fixed/`：

- [主对照结果](../../../data/experiments/eval_v1/retrieval_fixed/retrieval_results.json)、[逐题重排结果](../../../data/experiments/eval_v1/retrieval_fixed/rerank_dev_results.json)、[编码器固定输入对照](../../../data/experiments/eval_v1/retrieval_fixed/encoder_controlled/results.json)。
- [语料来源与 chunk 指纹](../../../data/experiments/eval_v1/retrieval_fixed/corpus_manifest.json)、[旧向量逐元素核验](../../../data/experiments/eval_v1/retrieval_fixed/legacy_vector_audit.json)、[源块对齐诊断](../../../data/experiments/eval_v1/retrieval_fixed/source_alignment_diagnostics.json)。
- [实际 Retriever 检查](../../../data/experiments/eval_v1/retrieval_fixed/semantic_1024/production_retriever_check.json)、[成对比较](../../../data/experiments/eval_v1/retrieval_fixed/paired_comparisons.json)、[后续生成候选](../../../data/experiments/eval_v1/retrieval_fixed/generation_candidates.json)。
- [完成核验记录](../../../data/experiments/eval_v1/retrieval_fixed/completion_manifest.json)：三套主实验 chunks 用最终代码重新生成，正文和 provenance 均与保存产物一致；五组 embedding 缓存及 query 向量通过 hash 校验，28 项回归测试通过。该记录另存最终源码指纹，执行时 manifest 中的历史源码指纹保持原样。
- 当前复现命令统一记录在仓库根 README；历史 runner 已从提交版移除。

原始检查目录 `retrieval_v1/` 保留修复前的排查记录，未完成全部配置；正式结论使用 `retrieval_fixed/`。原始 v1.0 测试题、reference 和 holdout 划分未改动。
