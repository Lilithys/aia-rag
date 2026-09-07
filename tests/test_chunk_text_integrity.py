import unittest

from src.chunking.splitting import _hard_token_split, split_to_size
from src.chunking.tokenizer import tail_text
from src.chunking.strategies.semantic_section import chunk_document
from src.chunking.strategies.semantic_section_v2 import chunk_document as chunk_document_v2
from src.evaluation.evidence import attach_provenance, score_window


class ChunkTextIntegrityTests(unittest.TestCase):
    def test_hard_splits_reconstruct_chinese_without_replacement_characters(self):
        text = "驾驶员必须联系当地法院处理罚款。" * 20
        for budget in (1, 7, 64):
            pieces = _hard_token_split(text, budget)
            self.assertEqual("".join(pieces), text)
            self.assertTrue(all("�" not in p for p in pieces))

    def test_overlap_is_valid_unicode_suffix(self):
        text = "您的驾驶执照可能会因违反交通规则而被暂停。"
        for budget in range(1, 25):
            suffix = tail_text(text, budget)
            self.assertTrue(text.endswith(suffix))
            self.assertNotIn("�", suffix)

    def test_sentence_delimiters_survive_recursive_split(self):
        text = "支付罚款。不要忽略法院通知！联系当地法院？" * 10
        self.assertEqual("".join(split_to_size(text, 20)), text)

    def test_nested_sections_retain_inherited_lead_and_breadcrumb_evidence(self):
        body = "# Main\nUnique introductory condition applies.\n## Parent\nParent information.\n### Child A\n" + "Facts. " * 80 + "\n### Child B\n" + "Other facts. " * 80
        meta = {"doc_id": "d", "domain": "ssa", "language": "en", "source_file": "d.md"}
        chunks = chunk_document(body, meta, chunk_size=40, overlap=0)
        self.assertIn("Unique introductory condition applies.", "\n".join(c.text for c in chunks))
        attach_provenance(chunks, body, [{"source_block_id": "parent", "text": "Parent"}])
        self.assertEqual(score_window(chunks, {("d", "parent")})["all_annotated_answer_blocks_hit"], 1)

    def test_semantic_v2_preserves_heading_only_leaf_when_parent_splits(self):
        body = "# Required good-cause condition\n\n# Other\n" + "Other details. " * 100
        meta = {"doc_id": "d", "domain": "va", "language": "en", "source_file": "d.pdf"}
        old = chunk_document(body, meta, chunk_size=32, overlap=0)
        new = chunk_document_v2(body, meta, chunk_size=32, overlap=0)
        self.assertNotIn("Required good-cause condition", "\n".join(c.text for c in old))
        self.assertIn("Required good-cause condition", "\n".join(c.text for c in new))
        self.assertTrue(all(c.strategy == "semantic_section_v2" for c in new))


if __name__ == "__main__":
    unittest.main()
