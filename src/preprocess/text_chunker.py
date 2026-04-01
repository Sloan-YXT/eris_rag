"""固定窗口分块器：将小说原文切成适合 embedding 的文本块。

分块策略：
- 目标大小 384 字符，最大 480，最小 100
- 重叠 64 字符
- 切割优先级：段落分隔 > ★ ★ ★ > 句末标点 > 逗号 > 硬切
- 不依赖 LLM，纯文本处理
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 优先切割点的正则
_PARA_BREAK = re.compile(r"\n\s*\n")  # 段落分隔（双换行）
_SCENE_BREAK = re.compile(r"★ ★ ★")
_SENTENCE_END = re.compile(r"[。？！…」）]")
_COMMA = re.compile(r"[，、；：]")


@dataclass
class TextChunkResult:
    """分块结果。"""
    text: str
    char_offset: int   # 在输入文本中的起始字符位置
    char_length: int


def chunk_text(
    text: str,
    target_size: int = 384,
    max_size: int = 480,
    min_size: int = 100,
    overlap: int = 64,
) -> list[TextChunkResult]:
    """将文本切成固定窗口的块，尊重段落和句子边界。

    Args:
        text: 输入文本。
        target_size: 目标块大小（字符数）。
        max_size: 最大块大小（硬上限）。
        min_size: 最小块大小（低于此则合并到前一个块）。
        overlap: 相邻块的重叠字符数。

    Returns:
        TextChunkResult 列表。
    """
    if not text or not text.strip():
        return []

    text_len = len(text)
    if text_len <= max_size:
        return [TextChunkResult(text=text.strip(), char_offset=0, char_length=text_len)]

    chunks: list[TextChunkResult] = []
    pos = 0

    while pos < text_len:
        # 确定搜索窗口的末尾
        end = min(pos + max_size, text_len)

        if end >= text_len:
            # 剩余文本不超过 max_size，直接取完
            chunk_text_raw = text[pos:]
            if chunk_text_raw.strip():
                # 如果太短且有前一个块，合并
                if len(chunk_text_raw.strip()) < min_size and chunks:
                    prev = chunks[-1]
                    merged = text[prev.char_offset:end]
                    # 合并后如果不超过 max_size * 1.2，就合并
                    if len(merged) <= max_size * 1.2:
                        chunks[-1] = TextChunkResult(
                            text=merged.strip(),
                            char_offset=prev.char_offset,
                            char_length=end - prev.char_offset,
                        )
                    else:
                        chunks.append(TextChunkResult(
                            text=chunk_text_raw.strip(),
                            char_offset=pos,
                            char_length=end - pos,
                        ))
                else:
                    chunks.append(TextChunkResult(
                        text=chunk_text_raw.strip(),
                        char_offset=pos,
                        char_length=end - pos,
                    ))
            break

        # 在 target_size 附近找最佳切割点
        cut = _find_best_cut(text, pos, target_size, max_size)
        chunk_text_raw = text[pos:cut]

        if chunk_text_raw.strip():
            chunks.append(TextChunkResult(
                text=chunk_text_raw.strip(),
                char_offset=pos,
                char_length=cut - pos,
            ))

        # 下一个块的起始位置（减去 overlap）
        next_pos = cut - overlap
        if next_pos <= pos:
            next_pos = cut  # 防止死循环
        pos = next_pos

    return chunks


@dataclass
class ParentChildResult:
    """父子块结果。"""
    parent: TextChunkResult
    children: list[TextChunkResult]


def chunk_text_parent_child(
    text: str,
    parent_size: int = 512,
    parent_overlap: int = 64,
    child_size: int = 80,
    child_overlap: int = 24,
    min_size: int = 30,
) -> list[ParentChildResult]:
    """父子块切分：先切父块，再从每个父块中切子块。

    子块用于 embedding 搜索（小而精准），
    父块用于返回上下文（大而完整）。
    """
    if not text or not text.strip():
        return []

    # 第一层：切父块
    parents = chunk_text(
        text,
        target_size=parent_size,
        max_size=int(parent_size * 1.2),
        min_size=min_size,
        overlap=parent_overlap,
    )

    results = []
    for parent in parents:
        # 第二层：在每个父块内切子块
        children = chunk_text(
            parent.text,
            target_size=child_size,
            max_size=int(child_size * 1.5),
            min_size=min_size,
            overlap=child_overlap,
        )
        # 子块的 char_offset 要加上父块在全文中的偏移
        for child in children:
            child.char_offset += parent.char_offset
        results.append(ParentChildResult(parent=parent, children=children))

    return results


def _find_best_cut(text: str, start: int, target: int, maximum: int) -> int:
    """在 [start+target, start+maximum] 范围内找最佳切割点。

    优先级：段落分隔 > ★ ★ ★ > 句末标点 > 逗号 > 硬切在 target 处。
    搜索从 target 位置往两边扩展。
    """
    # 搜索范围：target 位置前后各 search_range 字符
    search_start = start + max(target - 80, 0)
    search_end = start + maximum
    window = text[search_start:search_end]

    best_pos = None
    best_priority = 99

    # 优先级 0: 段落分隔
    for m in _PARA_BREAK.finditer(window):
        candidate = search_start + m.end()
        if _is_valid_cut(candidate, start, target, maximum):
            dist = abs(candidate - (start + target))
            if best_priority > 0 or (best_priority == 0 and dist < abs(best_pos - (start + target))):
                best_pos = candidate
                best_priority = 0

    # 优先级 1: ★ ★ ★ 场景分隔
    if best_priority > 1:
        for m in _SCENE_BREAK.finditer(window):
            candidate = search_start + m.end()
            if _is_valid_cut(candidate, start, target, maximum):
                if best_priority > 1 or abs(candidate - (start + target)) < abs(best_pos - (start + target)):
                    best_pos = candidate
                    best_priority = 1

    # 优先级 2: 句末标点
    if best_priority > 2:
        closest = None
        closest_dist = float('inf')
        for m in _SENTENCE_END.finditer(window):
            candidate = search_start + m.end()
            if _is_valid_cut(candidate, start, target, maximum):
                dist = abs(candidate - (start + target))
                if dist < closest_dist:
                    closest = candidate
                    closest_dist = dist
        if closest is not None:
            best_pos = closest
            best_priority = 2

    # 优先级 3: 逗号
    if best_priority > 3:
        closest = None
        closest_dist = float('inf')
        for m in _COMMA.finditer(window):
            candidate = search_start + m.end()
            if _is_valid_cut(candidate, start, target, maximum):
                dist = abs(candidate - (start + target))
                if dist < closest_dist:
                    closest = candidate
                    closest_dist = dist
        if closest is not None:
            best_pos = closest
            best_priority = 3

    # 兜底：硬切在 target 处
    if best_pos is None:
        best_pos = start + target

    return best_pos


def _is_valid_cut(cut: int, start: int, target: int, maximum: int) -> bool:
    """检查切割点是否产生合理大小的块。"""
    chunk_len = cut - start
    return chunk_len >= target * 0.5 and chunk_len <= maximum
