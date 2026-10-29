"""
Fuzzy Full-Text Index
Inverted index with edit-distance search, prefix autocomplete, and ~ operator.
Handles "Mahul" → "Rahul" fuzzy retrieval.
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import re


class TrieNode:
    """Trie node for prefix-based autocomplete."""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.entity_ids: Set[str] = set()  # Entities at this exact prefix


class FuzzyIndex:
    """
    Full-text inverted index with:
    1. Inverted index: token → set of entity IDs (for fast lookup)
    2. Trie: prefix → entity IDs (for autocomplete)
    3. Edit-distance search: ~ operator finds near-matches
    """

    def __init__(self):
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.trie_root = TrieNode()
        self.entity_tokens: Dict[str, List[str]] = {}  # entity_id → [tokens]
        self.entity_names: Dict[str, str] = {}  # entity_id → display name
        self.entity_types: Dict[str, str] = {}  # entity_id → entity type

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens.

        Devanagari is transliterated first so cross-script name variants
        ("नेहरू नगर" / "Nehru Nagar") produce comparable tokens.
        """
        from .translit import transliterate_devanagari
        text = transliterate_devanagari(text)
        return [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', text.lower()) if len(t) > 1]

    def _insert_trie(self, token: str, entity_id: str):
        """Insert a token into the trie."""
        node = self.trie_root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
            node.entity_ids.add(entity_id)

    def add_entity(self, entity_id: str, name: str, entity_type: str = "", aliases: List[str] = None):
        """Add an entity to the index."""
        all_names = [name]
        if aliases:
            all_names.extend(aliases)

        tokens = []
        for n in all_names:
            tokens.extend(self._tokenize(n))

        # Deduplicate tokens
        tokens = list(set(tokens))

        self.entity_tokens[entity_id] = tokens
        self.entity_names[entity_id] = name
        self.entity_types[entity_id] = entity_type

        for token in tokens:
            self.inverted_index[token].add(entity_id)
            self._insert_trie(token, entity_id)

    def remove_entity(self, entity_id: str):
        """Remove an entity from the index."""
        tokens = self.entity_tokens.pop(entity_id, [])
        self.entity_names.pop(entity_id, None)
        self.entity_types.pop(entity_id, None)

        for token in tokens:
            if entity_id in self.inverted_index.get(token, set()):
                self.inverted_index[token].discard(entity_id)

    def search_exact(self, query: str) -> List[str]:
        """Exact token match. Returns entity IDs."""
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Intersect results for all tokens (AND semantics)
        result_sets = [self.inverted_index.get(t, set()) for t in tokens]
        if not result_sets:
            return []

        result = result_sets[0]
        for s in result_sets[1:]:
            result = result & s
        return list(result)

    def search_fuzzy(self, query: str, max_distance: int = 2, max_results: int = 20) -> List[Tuple[str, str, float]]:
        """
        Fuzzy search using edit distance (~ operator).
        Returns: [(entity_id, name, similarity), ...] sorted by similarity.

        Scoring is entity-level with AND semantics: EVERY query token must
        fuzzy-match some token of the entity. Score = mean of per-token best
        similarities. Best-single-token scoring (old behavior) let "Delhi
        Herald" match "Delhi Police" at 1.0 via the shared token "delhi".
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # entity_id → {query_token: best_similarity}
        best: Dict[str, Dict[str, float]] = defaultdict(dict)

        for qtoken in query_tokens:
            for index_token, entity_ids in self.inverted_index.items():
                # Quick length filter
                if abs(len(index_token) - len(qtoken)) > max_distance:
                    continue

                dist = self._edit_distance(qtoken, index_token)
                if dist <= max_distance:
                    similarity = 1.0 - (dist / max(len(qtoken), len(index_token)))
                    for eid in entity_ids:
                        if qtoken not in best[eid] or best[eid][qtoken] < similarity:
                            best[eid][qtoken] = similarity

        # AND semantics: every query token must match some entity token.
        # Score = mean of per-token similarities.
        # (Best-single-token scoring let "Delhi Herald" match "Delhi Police"
        # at 1.0 via only the shared token "delhi".)
        n_q = len(query_tokens)
        results = []
        for eid, token_sims in best.items():
            if len(token_sims) < n_q:
                continue
            score = sum(token_sims.values()) / n_q
            results.append((eid, self.entity_names.get(eid, ""), score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:max_results]

    def search_prefix(self, prefix: str, max_results: int = 10) -> List[Tuple[str, str, str]]:
        """
        Prefix search using trie (for autocomplete).
        Returns: [(entity_id, name, entity_type), ...]
        """
        prefix_lower = prefix.lower().strip()
        if not prefix_lower:
            return []

        # Walk the trie
        node = self.trie_root
        for ch in prefix_lower:
            if ch not in node.children:
                return []
            node = node.children[ch]

        # Collect all entity IDs under this prefix
        results = []
        self._collect_trie_entities(node, results, max_results)

        # Deduplicate and return with names
        seen = set()
        output = []
        for eid in results:
            if eid not in seen:
                seen.add(eid)
                output.append((eid, self.entity_names.get(eid, ""), self.entity_types.get(eid, "")))
                if len(output) >= max_results:
                    break
        return output

    def _collect_trie_entities(self, node: TrieNode, results: list, max_results: int):
        """Recursively collect entity IDs from trie node."""
        results.extend(node.entity_ids)
        if len(results) >= max_results:
            return
        for child in node.children.values():
            if len(results) >= max_results:
                return
            self._collect_trie_entities(child, results, max_results)

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein edit distance."""
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        m, n = len(s1), len(s2)
        dp = list(range(n + 1))

        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i-1] == s2[j-1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j-1])
                prev = temp

        return dp[n]

    def __len__(self) -> int:
        return len(self.entity_tokens)

    def stats(self) -> dict:
        """Index statistics."""
        return {
            "entities_indexed": len(self.entity_tokens),
            "unique_tokens": len(self.inverted_index),
            "avg_tokens_per_entity": sum(len(v) for v in self.entity_tokens.values()) / max(len(self.entity_tokens), 1),
        }
