"""小说文本解析器：按卷切分 → 清理 → 固定窗口分块 → 输出 Chunk 列表。

流程：novel.txt → 按卷header切分 → 清理元数据 → text_chunker 分块 → Chunk 对象
不涉及 LLM，纯文本处理。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.models import Chunk
from src.preprocess.text_chunker import chunk_text

# 中文数字 → 阿拉伯数字
_CN_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24, "二十五": 25,
}


def _cn_to_int(s: str) -> int:
    """中文数字或阿拉伯数字字符串 → int。支持 '20.5' 这种特殊卷号。"""
    s = s.strip()
    if s in _CN_NUM:
        return _CN_NUM[s]
    try:
        return int(float(s))  # 处理 "20.5" → 20
    except ValueError:
        return 0


# Regex patterns — 支持中文数字和阿拉伯数字
CHAPTER_HEADER = re.compile(
    r"^# 第([^\s卷]+)卷\s+(.+?)(?:\((\d+)/(\d+)\))?$"
)
UNDERSCORED_TITLE = re.compile(r"^_第[^_]+_$")
SKIP_LINES = re.compile(
    r"^(铅笔小说|www\.x23qb\.com|台版|扫图[：:]|录入[：:]|着[：:]|译[：:])"
)


@dataclass
class ChapterChunk:
    """按卷header切分出的一个章节（中间产物）。"""
    volume: int
    chapter_index: int
    title: str
    text: str


def parse_novel_to_chunks(
    txt_path: str | Path,
    taxonomy_path: str | Path | None = None,
    target_size: int = 384,
    max_size: int = 480,
    overlap: int = 64,
) -> list[Chunk]:
    """解析小说文件，输出可直接 embed 的 Chunk 列表。

    Args:
        txt_path: 小说 TXT 文件路径。
        taxonomy_path: tags_taxonomy.yaml 路径（用于卷号→时期映射）。
        target_size: 目标块大小（字符）。
        max_size: 最大块大小（字符）。
        overlap: 重叠字符数。

    Returns:
        Chunk 列表，每个 chunk.raw_text 就是 embedding 的输入。
    """
    txt_path = Path(txt_path)
    text = txt_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    period_map, weight_map = _load_period_config(taxonomy_path)
    raw_chapters = _split_chapters(lines)

    # 没有找到卷标记时，把整个文件当作一个章节（蛇足篇等番外）
    if not raw_chapters:
        raw_chapters = [(99, 0, txt_path.stem, lines)]

    all_chunks: list[Chunk] = []
    global_seq = 0

    for vol, chap_idx, title, content_lines in raw_chapters:
        cleaned = _clean_lines(content_lines)
        if not cleaned.strip():
            continue

        period = _get_period(vol, period_map)
        weight = weight_map.get(period, 1.0)

        # 用 text_chunker 做固定窗口分块
        text_chunks = chunk_text(
            cleaned,
            target_size=target_size,
            max_size=max_size,
            overlap=overlap,
        )

        for tc in text_chunks:
            chunk = Chunk(
                id=f"v{vol:02d}_ch{chap_idx:02d}_{global_seq:04d}",
                raw_text=tc.text,
                volume=vol,
                chapter=chap_idx,
                chunk_index=global_seq,
                char_offset=tc.char_offset,
                source_file=Path(txt_path).name,
                period=period,
                period_weight=weight,
            )
            all_chunks.append(chunk)
            global_seq += 1

    return all_chunks


# ── 保留旧接口（向后兼容，部分测试依赖） ─────────────────

def parse_novel(
    txt_path: str | Path,
    taxonomy_path: str | Path | None = None,
) -> list[ChapterChunk]:
    """旧接口：按卷切分，返回 ChapterChunk 列表。"""
    txt_path = Path(txt_path)
    text = txt_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    period_map, weight_map = _load_period_config(taxonomy_path)
    raw_chapters = _split_chapters(lines)

    chapters = []
    for vol, chap_idx, title, content_lines in raw_chapters:
        cleaned = _clean_lines(content_lines)
        if not cleaned.strip():
            continue
        chapters.append(ChapterChunk(
            volume=vol,
            chapter_index=chap_idx,
            title=title,
            text=cleaned,
        ))
    return chapters


# ── 内部工具函数 ─────────────────────────────────────────

def _split_chapters(lines: list[str]) -> list[tuple[int, int, str, list[str]]]:
    """按 # 第X卷 header 切分，处理分页合并。"""
    chapters: list[tuple[int, int, str, list[str]]] = []
    current_vol = 0
    current_title = ""
    current_lines: list[str] = []
    chapter_counter: dict[int, int] = {}
    seen_headers: set[str] = set()

    for line in lines:
        m = CHAPTER_HEADER.match(line.strip())
        if m:
            vol = _cn_to_int(m.group(1))
            raw_title = m.group(2).strip()

            header_key = f"v{vol}_{raw_title}"
            if header_key in seen_headers:
                continue

            if current_lines and current_title:
                chap_idx = chapter_counter.get(current_vol, 0)
                chapters.append((current_vol, chap_idx, current_title, current_lines))
                chapter_counter[current_vol] = chap_idx + 1

            seen_headers.add(header_key)
            current_vol = vol
            current_title = raw_title
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines and current_title:
        chap_idx = chapter_counter.get(current_vol, 0)
        chapters.append((current_vol, chap_idx, current_title, current_lines))

    return chapters


def _clean_lines(lines: list[str]) -> str:
    """清理元数据行，合并空行。"""
    cleaned = []
    prev_blank = False

    for line in lines:
        stripped = line.strip()

        if UNDERSCORED_TITLE.match(stripped):
            continue
        if SKIP_LINES.match(stripped):
            continue
        if CHAPTER_HEADER.match(stripped):
            continue

        if not stripped:
            if not prev_blank:
                cleaned.append("")
                prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False

    return "\n".join(cleaned).strip()


def _load_period_config(
    taxonomy_path: str | Path | None,
) -> tuple[dict[str, list[int]], dict[str, float]]:
    if taxonomy_path is None:
        return {}, {}
    path = Path(taxonomy_path)
    if not path.exists():
        return {}, {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    volume_periods = data.get("volume_periods", {})
    weights_default = data.get("period_weights", {}).get("default", {})
    return volume_periods, weights_default


def _get_period(volume: int, period_map: dict[str, list[int]]) -> str:
    for period_name, volumes in period_map.items():
        if volume in volumes:
            return period_name
    return "未知"
