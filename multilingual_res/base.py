"""
Abstract base class for resource fetchers in multilingual_res.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseResourceFetcher(ABC):
    @abstractmethod
    def fetch(cls, language_code: str, script_code: Optional[str] = None) -> List[Dict]:
        """
        Fetch data for a given language code and optional script code.
        Returns a list of dicts (see resource fetcher for format).
        """
        pass
