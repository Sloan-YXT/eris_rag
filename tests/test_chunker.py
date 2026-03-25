"""Tests for the novel chunker: chapter splitting, pagination merging, metadata cleaning."""

import pytest
from pathlib import Path

from src.preprocess.chunker import parse_novel, parse_novel_to_chunks, _clean_lines


@pytest.fixture
def sample_novel(tmp_path):
    """Create a minimal sample novel file matching the chunker's expected format."""
    content = """# 第1卷 幼年期

这是第一卷的内容。

★ ★ ★

艾莉丝正在练剑。
「哼，这种程度的对手！」
她挥舞着剑。

★ ★ ★

日常的一天结束了。

# 第2卷 少年期(1/2)

第二卷上半部分。

# 第2卷 少年期(2/2)

第二卷下半部分。

★ ★ ★

「我变强了。」
"""
    novel_path = tmp_path / "test_novel.txt"
    novel_path.write_text(content, encoding="utf-8")
    return novel_path


class TestChunker:
    def test_split_chapters(self, sample_novel, tmp_config):
        chapters = parse_novel(str(sample_novel), str(tmp_config.taxonomy_path))
        # Should have chapters from 2 volumes
        volumes = set(ch.volume for ch in chapters)
        assert 1 in volumes
        assert 2 in volumes

    def test_text_chunking(self, sample_novel, tmp_config):
        """parse_novel_to_chunks 应该输出 Chunk 列表。"""
        chunks = parse_novel_to_chunks(str(sample_novel), str(tmp_config.taxonomy_path))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.raw_text.strip() != ""
            assert chunk.volume > 0

    def test_pagination_merge(self, sample_novel, tmp_config):
        chapters = parse_novel(str(sample_novel), str(tmp_config.taxonomy_path))
        vol2_chapters = [ch for ch in chapters if ch.volume == 2]
        # Two paginated headers "少年期(1/2)" and "少年期(2/2)" should merge into one chapter
        assert len(vol2_chapters) == 1
        text = vol2_chapters[0].text
        # Both halves' content should be present
        assert "上半部分" in text
        assert "下半部分" in text

    def test_clean_lines(self):
        lines = [
            "正常内容",
            "铅笔小说 www.example.com",
            "_第一章_",
            "",
            "",
            "",
            "更多内容",
        ]
        cleaned = _clean_lines(lines)
        assert "铅笔小说" not in cleaned
        # Underscored title should be removed
        assert "_第一章_" not in cleaned
        # Multiple blank lines should be collapsed to at most one
        assert "\n\n\n" not in cleaned
