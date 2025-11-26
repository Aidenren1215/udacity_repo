import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

import httpx
from leann import LeannBuilder, LeannSearcher


def configure_leann_openai(
    base_url: str,
    api_key: str = "dummy",
    disable_ssl_verify: bool = True,
) -> None:
    """
    配置 LEANN 使用的 OpenAI-兼容内部 API。
    - 会设置 OPENAI_BASE_URL / OPENAI_API_KEY 环境变量
    - 可选：全局 monkey-patch httpx.Client，让 verify=False
    必须在第一次使用 LeannVectorDB / LeannBuilder 之前调用一次。
    """
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key

    if disable_ssl_verify:
        # 全局 patch httpx.Client 默认 verify=False（你那边自签证书必须关验证）
        if not getattr(httpx.Client, "_leann_ocbc_patched", False):
            original_init = httpx.Client.__init__

            def _patched_init(self, *args, **kwargs):
                # 调用方如果显式写了 verify，就尊重；否则默认 False
                kwargs.setdefault("verify", False)
                return original_init(self, *args, **kwargs)

            httpx.Client.__init__ = _patched_init
            httpx.Client._leann_ocbc_patched = True

        print("[configure_leann_openai] httpx.Client patched with verify=False")

    print("[configure_leann_openai] OPENAI_BASE_URL =", os.environ.get("OPENAI_BASE_URL"))


class LeannVectorDB:
    """
    一个简单的 LEANN 封装，把单个 .leann 文件当成一个“collection”。

    设计：
    - LEANN 本身只做索引（.leann）
    - 我们自己在旁边维护一个 store.jsonl 当“文档仓库”，结构：
        {"id": "...", "text": "...", "metadata": {...}, "deleted": false}

    对外提供：
    - add(text, metadata)           # 单条添加到 store（不会立刻更新索引）
    - add_many(texts, metadatas)    # 批量添加
    - delete(doc_id)                # 标记删除
    - rebuild_index()               # 全量读取 store 中未删除的记录，重建 .leann 索引
    - search(query, top_k)          # 调用 LeannSearcher.search
    """

    def __init__(
        self,
        index_path: str,
        backend_name: str = "hnsw",
        embedding_mode: str = "openai",
        embedding_model: str = "bge-large-en-v1.5",
        store_path: Optional[str] = None,
    ):
        """
        :param index_path:   LEANN 索引文件路径，例如 "fd_docs.leann"
        :param backend_name: LEANN backend，一般用 "hnsw" 或 "diskann"
        :param embedding_mode:   LEANN embedding backend，例如 "openai"
        :param embedding_model:  使用的 embedding 模型名（供 LEANN 内部调用）
        :param store_path:   文档存储 JSONL 路径（默认 = index_path + ".store.jsonl"）
        """
        self.index_path = Path(index_path)
        self.dir = self.index_path.parent

        if store_path is None:
            # 单独的文档存储文件，避免和 LEANN 自己的 meta / passages 冲突
            self.store_path = self.index_path.with_suffix(self.index_path.suffix + ".store.jsonl")
        else:
            self.store_path = Path(store_path)

        self.backend_name = backend_name
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model

        self._searcher: Optional[LeannSearcher] = None

        # 确保目录存在
        self.dir.mkdir(parents=True, exist_ok=True)

    # ================ 内部工具 ================

    def _invalidate_searcher(self) -> None:
        """索引重建后，需要重新加载 searcher。"""
        self._searcher = None

    def _iter_store(self) -> Iterator[Dict[str, Any]]:
        """逐行读取 store.jsonl。"""
        if not self.store_path.exists():
            return
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _write_store(self, records: List[Dict[str, Any]]) -> None:
        """全量重写 store.jsonl。"""
        with self.store_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _ensure_searcher(self) -> LeannSearcher:
        """惰性加载 LeannSearcher。"""
        if self._searcher is None:
            if not self.index_path.exists():
                raise RuntimeError(f"Index file does not exist: {self.index_path}")
            self._searcher = LeannSearcher(str(self.index_path))
        return self._searcher

    # ================ 对外 API：add / add_many / delete / rebuild_index / search ================

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, doc_id: Optional[str] = None) -> str:
        """
        单条新增（只写入 store.jsonl，不会立刻更新索引，需要之后调用 rebuild_index）。

        :return: 最终使用的 doc_id
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        record = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "deleted": False,
        }

        # 追加写入
        with self.store_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._invalidate_searcher()
        return doc_id

    def add_many(
        self,
        texts: List[str],
        metadatas: Optional[List[Optional[Dict[str, Any]]]] = None,
        doc_ids: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """
        批量新增：
        - texts: 必填，长度 N
        - metadatas: 可选，长度 N 或 None
        - doc_ids: 可选，长度 N 或 None（None 时自动生成）

        只写 store.jsonl，仍然需要你之后调用 rebuild_index() 才会更新索引。
        """
        n = len(texts)
        metadatas = metadatas or [None] * n
        if doc_ids is None:
            doc_ids = [None] * n

        if not (len(metadatas) == len(doc_ids) == n):
            raise ValueError("texts, metadatas, doc_ids 长度必须一致")

        out_ids: List[str] = []

        with self.store_path.open("a", encoding="utf-8") as f:
            for text, meta, did in zip(texts, metadatas, doc_ids):
                if did is None:
                    did = str(uuid.uuid4())
                out_ids.append(did)

                record = {
                    "id": did,
                    "text": text,
                    "metadata": meta or {},
                    "deleted": False,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._invalidate_searcher()
        return out_ids

    def delete(self, doc_id: str) -> None:
        """
        逻辑删除：
        - 在 store.jsonl 中把对应记录的 deleted 标记为 True
        - 不会立刻更新索引，需要之后调用 rebuild_index()
        """
        records = list(self._iter_store())
        changed = False
        for r in records:
            if r.get("id") == doc_id and not r.get("deleted", False):
                r["deleted"] = True
                changed = True

        if changed:
            self._write_store(records)
            self._invalidate_searcher()

    def rebuild_index(self) -> None:
        """
        全量重建 LEANN 索引：
        - 读取 store.jsonl 中 deleted == False 的记录
        - 逐条 builder.add_text(text)
        - 调 build_index() 覆盖写 index_path
        """
        active_records = [r for r in self._iter_store() if not r.get("deleted", False)]

        if not active_records:
            # 没有有效文档，索引删掉即可
            if self.index_path.exists():
                self.index_path.unlink()
            self._invalidate_searcher()
            return

        builder = LeannBuilder(
            backend_name=self.backend_name,
            embedding_mode=self.embedding_mode,
            embedding_model=self.embedding_model,
        )

        for r in active_records:
            builder.add_text(r["text"])

        # 覆盖写索引文件
        builder.build_index(str(self.index_path))
        self._invalidate_searcher()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        用 LEANN 做语义搜索。

        返回：
        [
            {
                "text":  <匹配文本>,
                "score": <相似度/距离，视 LEANN 版本而定>,
                "raw":   <LEANN 原始返回对象>,
            },
            ...
        ]
        """
        searcher = self._ensure_searcher()
        results = searcher.search(query, top_k=top_k)

        parsed: List[Dict[str, Any]] = []

        for r in results:
            # 尝试从不同字段名里取 text / score（兼容不同版本）
            text = None
            score = None

            if isinstance(r, dict):
                text = r.get("text") or r.get("content") or r.get("chunk") or str(r)
                score = r.get("score") or r.get("distance")
            else:
                text = getattr(r, "text", None) or getattr(r, "content", None) or str(r)
                score = getattr(r, "score", None) or getattr(r, "distance", None)

            parsed.append(
                {
                    "text": text,
                    "score": score,
                    "raw": r,
                }
            )

        return parsed



from leann_vector_db import configure_leann_openai, LeannVectorDB

# 1) 先配置内部 LLM 网关 + 关闭 SSL 校验
configure_leann_openai(
    base_url="https://ocbc-llm-coordinator.xxx.apps.prod.ocbc.com",
    api_key="dummy-or-real",
)

# 2) 创建一个 "collection"（本质是一个 .leann + 一个 .store.jsonl）
db = LeannVectorDB(
    index_path="fd_docs.leann",
    backend_name="hnsw",
    embedding_mode="openai",
    embedding_model="bge-large-en-v1.5",
)

# 3) 批量 add 文本（比如已经完成 chunking 的结果）
texts = [
    "This chunk explains 6M vs 12M fixed deposit optimisation.",
    "This chunk explains FTP curves and internal transfer pricing.",
]
metas = [
    {"source": "fd_policy.pdf", "section": "tenor"},
    {"source": "ftp_guide.pdf", "section": "intro"},
]
doc_ids = db.add_many(texts, metadatas=metas)

# 4) 重建索引（仅在你完成一批 add/delete 后需要跑一次）
db.rebuild_index()

# 5) 搜索
res = db.search("how to optimise fixed deposits", top_k=3)
for r in res:
    print(r["score"], r["text"])

# 6) 删除一条，然后再重建一次索引
db.delete(doc_ids[0])
db.rebuild_index()
