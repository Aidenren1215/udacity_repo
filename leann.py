class LeannVectorDB:
    """
    A simple LEANN wrapper treating a single `.leann` file as a "collection".

    Design:
    - LEANN only handles vector indexing (.leann)
    - We maintain a separate store.jsonl as a lightweight document store:
        {"id": "...", "text": "...", "metadata": {...}, "deleted": false}

    Exposed API:
    - add(text, metadata)         # append to store.jsonl (does NOT update index immediately)
    - add_many(texts, metadatas)  # batch append
    - delete(doc_id)              # logical delete (mark deleted=true)
    - rebuild_index()             # rebuild .leann index from all non-deleted records
    - search(query, top_k)        # call LeannSearcher.search
    """

    def __init__(
        self,
        index_path: str,
        backend_name: str = "hnsw",
        # Default now uses local sentence-transformers (BGE) instead of "openai"
        embedding_mode: str = "sentence-transformers",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        store_path: Optional[str] = None,
    ):
        """
        :param index_path:      Path to LEANN index file, e.g. "fd_docs.leann"
        :param backend_name:    LEANN backend, usually "hnsw" or "diskann"
        :param embedding_mode:  Embedding backend for LEANN:
                                    - "sentence-transformers" = local HF/BGE (GPU-enabled)
                                    - "openai" or others = API-based
        :param embedding_model: Embedding model name passed to LEANN
        :param store_path:      Path to store.jsonl (default = index_path + ".store.jsonl")
        """
        self.index_path = Path(index_path)
        self.dir = self.index_path.parent

        # Store file for original texts + metadata
        if store_path is None:
            self.store_path = self.index_path.with_suffix(
                self.index_path.suffix + ".store.jsonl"
            )
        else:
            self.store_path = Path(store_path)

        self.backend_name = backend_name
        self.embedding_mode = embedding_mode
        self.embedding_model = embedding_model

        self._searcher: Optional[LeannSearcher] = None

        # Ensure directory exists
        self.dir.mkdir(parents=True, exist_ok=True)

    # ======================= Internal Utilities =======================

    def _invalidate_searcher(self) -> None:
        """Reset searcher so it reloads after index rebuild."""
        self._searcher = None

    def _iter_store(self) -> Iterator[Dict[str, Any]]:
        """Iterate through store.jsonl line by line."""
        if not self.store_path.exists():
            return
        with self.store_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _write_store(self, records: List[Dict[str, Any]]) -> None:
        """Rewrite store.jsonl completely."""
        with self.store_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _ensure_searcher(self) -> LeannSearcher:
        """
        Lazily load LeannSearcher.  
        LEANN will automatically read metadata (embedding_mode, model, backend)
        from the `.leann.meta.json` file.
        """
        if self._searcher is None:
            if not self.index_path.exists():
                raise RuntimeError(f"Index file does not exist: {self.index_path}")
            self._searcher = LeannSearcher(str(self.index_path))
        return self._searcher

    # ======================= Public API =======================

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Append a single record to store.jsonl.
        Does NOT update index until rebuild_index() is called.

        :return: The document ID used
        """
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        record = {
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "deleted": False,
        }

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
        Append multiple records to store.jsonl.
        Index is NOT updated until rebuild_index() is called.

        :param texts:      Required list of size N
        :param metadatas:  Optional list of size N or None
        :param doc_ids:    Optional list of size N or None
        :return:           List of doc_ids used
        """
        n = len(texts)
        metadatas = metadatas or [None] * n
        if doc_ids is None:
            doc_ids = [None] * n

        if not (len(metadatas) == len(doc_ids) == n):
            raise ValueError("texts, metadatas, and doc_ids must have equal length")

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
        Logical delete:
        - Mark the record in store.jsonl as deleted=True
        - Does NOT update index until rebuild_index() is called
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
        Rebuild entire LEANN index from all active (non-deleted) records.
        """
        active_records = [
            r for r in self._iter_store() if not r.get("deleted", False)
        ]

        if not active_records:
            # No active documents → remove index file
            if self.index_path.exists():
                self.index_path.unlink()
            self._invalidate_searcher()
            return

        # LEANN automatically uses GPU if torch.cuda.is_available()
        builder = LeannBuilder(
            backend_name=self.backend_name,
            embedding_mode=self.embedding_mode,     # "sentence-transformers"
            embedding_model=self.embedding_model,   # "BAAI/bge-large-en-v1.5"
        )

        for r in active_records:
            builder.add_text(r["text"])

        builder.build_index(str(self.index_path))
        self._invalidate_searcher()

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform semantic search via LEANN.

        Returns a list:
        [
            {
                "text":  <matched text>,
                "score": <similarity or distance>,
                "raw":   <raw LEANN result object>,
            },
            ...
        ]
        """
        searcher = self._ensure_searcher()
        results = searcher.search(query, top_k=top_k)

        parsed: List[Dict[str, Any]] = []

        for r in results:
            # Extract text and score in a version-agnostic way
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