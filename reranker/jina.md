
|   模型   |    特点    |   适用场景    |
|---- | ----| ---- |
|jina-reranker-v1-tiny-en|	轻量级，速度快，参数少	|英文场景，资源有限或低延迟需求|
|jina-reranker-v1-turbo-en|	较大模型，速度与准确度平衡	|英文检索，追求更好效果但仍要快|
|jina-reranker-v1-base-en|	基础版，准确度高于 tiny	|英文场景，效果优先于速度|
|jina-reranker-v2-base-multilingual |	支持多语言，跨语言检索效果好	|中文、英文、德语、西班牙语等多语言场景|
|jina-reranker-m0 |	更小的实验性模型，成本低 | 测试或轻量任务，不推荐生产|
|jina-colbert-v1-en|	多向量匹配（ColBERT架构），适合大规模检索	|英文场景，海量文档库，需高召回率|
|jina-reranker-v3 |	最新一代，0.6B 参数，支持长上下文，性能优于 v2 |	多语言场景，复杂检索任务，追求 SOTA 效果|