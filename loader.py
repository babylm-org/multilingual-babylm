import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_from_disk
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load_data(self, source_path: Path) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts with keys: 'text', 'doc_id', 'metadata'
        """
        pass


class TextDirLoader(BaseLoader):
    def load_data(self, source_path: Path) -> List[Dict[str, Any]]:
        docs = []
        for txt_file in sorted(source_path.glob("*.txt")):
            with open(txt_file, "r", encoding="utf-8") as f:
                doc_id = txt_file.stem  # Use filename without .txt
                docs.append({"text": f.read(), "doc_id": doc_id, "metadata": {}})
        return docs


class CSVLoader(BaseLoader):
    def __init__(self, text_field="text"):
        self.text_field = text_field

    def load_data(self, source_path: Path) -> List[Dict[str, Any]]:
        docs = []
        with open(source_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                text = row.get(self.text_field, "")
                if not text:
                    continue
                meta = {k: v for k, v in row.items() if k != self.text_field}
                doc_id = row.get("doc_id") or row.get("id") or None
                docs.append({"text": text, "doc_id": doc_id if doc_id is not None else str(i), "metadata": meta})
        return docs


class JSONLoader(BaseLoader):
    def __init__(self, text_field="text"):
        self.text_field = text_field

    def load_data(self, source_path: Path) -> List[Dict[str, Any]]:
        docs = []
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            items = list(data.values())
        else:
            items = data
        for i, row in enumerate(items):
            if not isinstance(row, dict):
                continue
            text = row.get(self.text_field, "")
            if not text:
                continue
            meta = {k: v for k, v in row.items() if k != self.text_field}
            doc_id = row.get("doc_id") or row.get("id") or None
            docs.append({"text": text, "doc_id": doc_id if doc_id is not None else str(i), "metadata": meta})
        return docs


class HFLoader(BaseLoader):
    def __init__(self, text_field="text", split: Optional[str] = None):
        self.text_field = text_field
        self.split = split

    def load_data(self, source_path: Path) -> List[Dict[str, Any]]:
        docs = []
        ds = load_from_disk(str(source_path))
        if self.split and hasattr(ds, "split"):
            ds = ds[self.split]
        elif isinstance(ds, dict):
            ds = next(iter(ds.values()))
        for i, row in enumerate(ds):
            if isinstance(row, dict):
                text = row.get(self.text_field, "")
                meta = {k: v for k, v in row.items() if k != self.text_field}
                doc_id = row.get("doc_id") or row.get("id") or None
            else:
                text = getattr(row, self.text_field, "")
                meta = {k: v for k, v in row.__dict__.items() if k != self.text_field}
                doc_id = getattr(row, "doc_id", None) or getattr(row, "id", None) or None
            if not text:
                continue
            docs.append({"text": text, "doc_id": doc_id if doc_id is not None else str(i), "metadata": meta})
        return docs


def get_loader(loader_type, **kwargs):
    if loader_type == "text":
        return TextDirLoader()
    elif loader_type == "csv":
        return CSVLoader(text_field=kwargs.get("text_field", "text"))
    elif loader_type == "json":
        return JSONLoader(text_field=kwargs.get("text_field", "text"))
    elif loader_type == "hf":
        return HFLoader(
            text_field=kwargs.get("text_field", "text"), split=kwargs.get("split")
        )
    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")
