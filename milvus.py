from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    list_collections,
)
from typing import List, Dict, Any, Optional


class MilvusLiteVectorDB:
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        dim: int,
        metadata_fields: Optional[Dict[str, DataType]] = None,
    ):
        """
        Milvus Lite Vector DB Wrapper

        Parameters
        ----------
        db_path : str
            Path to local .db file (Milvus Lite).
        collection_name : str
            Name of the Milvus collection.
        dim : int
            Embedding dimension.
        metadata_fields : dict[str, DataType], optional
            Extra scalar fields, for example:
                {
                    "doc_id": DataType.VARCHAR,
                    "chunk_id": DataType.INT64,
                    "text": DataType.VARCHAR,
                }
        """
        self.db_path = db_path
        self.collection_name = collection_name
        self.dim = dim
        self.metadata_fields = metadata_fields or {}

        # Connect to Milvus Lite
        connections.connect(alias="default", uri=db_path)

        # Create or load collection
        if collection_name not in list_collections():
            self._create_collection()
        else:
            self.collection = Collection(collection_name)

        self.collection.load()

    def _create_collection(self):
        print(f"[MilvusLiteVectorDB] Creating new collection: {self.collection_name}")

        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
        ]

        # Add metadata fields: doc_id / chunk_id / text / etc.
        for field, dtype in self.metadata_fields.items():
            if dtype == DataType.VARCHAR:
                fields.append(
                    FieldSchema(
                        name=field,
                        dtype=DataType.VARCHAR,
                        max_length=5000,
                    )
                )
            else:
                fields.append(FieldSchema(name=field, dtype=dtype))

        schema = CollectionSchema(fields=fields, description="Milvus Lite VectorDB")
        self.collection = Collection(self.collection_name, schema)

        # Create index on vector
        self.collection.create_index(
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
        )

    # ======================================================
    # Basic Operations
    # ======================================================

    def add(
        self,
        ids: List[int],
        vectors: List[List[float]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Insert embeddings + metadata.
        Chunking 自然通过 metadata 实现，比如：
            {"doc_id": "A", "chunk_id": 3, "text": "..."}
        """
        if metadata_list is None:
            metadata_list = [{} for _ in ids]

        if len(ids) != len(vectors) or len(ids) != len(metadata_list):
            raise ValueError("ids, vectors, metadata_list length must match")

        # Base columns: id + vector
        data_columns = [ids, vectors]

        # Append metadata columns in consistent order
        for field in self.metadata_fields.keys():
            col = [meta.get(field, "") for meta in metadata_list]
            data_columns.append(col)

        result = self.collection.insert(data_columns)
        return result.primary_keys

    def search(
        self,
        queries: List[List[float]],
        k: int = 5,
        filter_expr: Optional[str] = None,
    ):
        """
        Pure vector search + optional scalar filter.

        Parameters
        ----------
        queries : List[List[float]]
            Query vectors.
        k : int
            Top K.
        filter_expr : str, optional
            Milvus boolean expression on metadata fields, e.g.:
                'doc_id == "A"'
                'text like "%rate%"'
                'doc_id == "A" and chunk_id >= 3'
        """
        results = self.collection.search(
            data=queries,
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr,
            output_fields=["id"] + list(self.metadata_fields.keys()),
        )
        return results

    def delete(self, ids: List[int]):
        """
        Delete entities by primary key list.
        """
        if not ids:
            return None
        expr = f"id in {ids}"
        return self.collection.delete(expr)

    def upsert(
        self,
        ids: List[int],
        vectors: List[List[float]],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Delete existing ids then insert again.
        """
        if ids:
            self.delete(ids)
        return self.add(ids, vectors, metadata_list)

    def count(self) -> int:
        """
        Total number of entities in the collection.
        """
        return self.collection.num_entities

    def flush(self):
        """
        Flush pending inserts to disk.
        """
        self.collection.flush()

    # ======================================================
    # Extra: delete whole document by doc_id
    # ======================================================

    def delete_by_doc_id(self, doc_id: str):
        """
        Delete all chunks / rows belonging to a given doc_id.

        Requirements:
        - metadata_fields MUST contain "doc_id".
        """
        if "doc_id" not in self.metadata_fields:
            raise ValueError("metadata_fields does not contain 'doc_id', cannot delete by doc_id")

        # 注意：简单写法，没有做复杂转义
        expr = f'doc_id == "{doc_id}"'
        return self.collection.delete(expr)

    # ======================================================
    # Extra: hybrid search (vector + metadata keyword / doc_id)
    # ======================================================

    def hybrid_search(
        self,
        queries: List[List[float]],
        k: int = 5,
        keyword: Optional[str] = None,
        doc_id: Optional[str] = None,
    ):
        """
        Hybrid search: vector similarity + metadata constraints.

        Parameters
        ----------
        queries : List[List[float]]
            Query embeddings.
        k : int
            Top K.
        keyword : str, optional
            Keyword to match against a 'text' field using LIKE.
            Requires metadata_fields 包含 "text": DataType.VARCHAR.
        doc_id : str, optional
            Filter by doc_id == given value.
            Requires metadata_fields 包含 "doc_id": DataType.VARCHAR.

        Returns
        -------
        search_results : List[Hits]
            Milvus search results, each Hit 带有 entity.metadata.
        """
        expr_parts = []

        if doc_id is not None:
            if "doc_id" not in self.metadata_fields:
                raise ValueError("metadata_fields does not contain 'doc_id', cannot filter by doc_id")
            # 简单处理：把 " 换成 ' 避免表达式坏掉
            doc_id_safe = doc_id.replace('"', "'")
            expr_parts.append(f'doc_id == "{doc_id_safe}"')

        if keyword is not None:
            if "text" not in self.metadata_fields:
                raise ValueError("metadata_fields does not contain 'text', cannot filter by keyword")
            kw_safe = keyword.replace('"', "'")
            expr_parts.append(f'text like "%{kw_safe}%"')

        filter_expr = " and ".join(expr_parts) if expr_parts else None

        results = self.collection.search(
            data=queries,
            anns_field="vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=k,
            expr=filter_expr,
            output_fields=["id"] + list(self.metadata_fields.keys()),
        )
        return results
