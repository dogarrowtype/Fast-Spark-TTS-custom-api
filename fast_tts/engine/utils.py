# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:32
# Author    :Hui Huang
import asyncio
import re
from typing import Callable, Optional

import regex


def limit_concurrency(semaphore: asyncio.Semaphore):
    def decorator(func):
        async def wrapped(*args, **kwargs):
            async with semaphore:  # 在这里限制并发请求数
                return await func(*args, **kwargs)

        return wrapped

    return decorator


def contains_chinese(s: str) -> bool:
    """
    判断字符串中是否包含中文字符
    """
    return bool(re.search(r'[\u4e00-\u9fff]', s))


# 以下代码从cosyvoice项目copy的
def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))


def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


def replace_blank(text: str):
    out_str = []
    for i, c in enumerate(text):
        if c == " ":
            if ((text[i + 1].isascii() and text[i + 1] != " ") and
                    (text[i - 1].isascii() and text[i - 1] != " ")):
                out_str.append(c)
        else:
            out_str.append(c)
    return "".join(out_str)


def text_normalize(text: str) -> str:
    if contains_chinese(text):
        text = text.replace("\n", "")
        text = replace_blank(text)
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r'[，,、]+$', '。', text)
        if text[-1] not in ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']:
            text += "。"
    return text


def split_text(
        text: str,
        window_size: int,
        tokenize_fn: Callable[[str], list[str]],
        split_fn: Optional[Callable[[str], list[str]]] = None,
        length_threshold: int = 50,
) -> list[str]:
    """
    将长文本拆分成多个片段。首先使用中英文句号、问号、感叹号等切分文本，
    然后根据传入的窗口大小将切分后的句子合并成不超过窗口大小的片段。
    如果单个句子长度超过窗口大小，则会对该句子进行切割。

    :param text: 输入的长文本
    :param window_size: 每个片段的最大长度
    :param tokenize_fn: 分词函数
    :param split_fn: 片段切分方法，如果传入就使用自定义切分函数
    :param length_threshold: 长度阈值，超过这个值将进行切分
    :return: 切分后的文本片段列表
    """
    text = text_normalize(text)
    if len(tokenize_fn(text)) <= length_threshold:
        return [text]

    if split_fn is None:
        sentences = re.split(r'(?<=[。？！；;.!?：:])', text)
        # 去除拆分过程中产生的空字符串，并去除两侧空白
    else:
        sentences = split_fn(text)

    sentences = [s.strip() for s in sentences if s.strip()]
    segments = []
    current_segment = ""
    current_length = 0
    for sentence in sentences:
        sent_len = len(tokenize_fn(sentence))
        if sent_len > window_size:
            segments.append(current_segment)
            segments.append(sentence)  # 不进一步细分
            current_segment = ""
            current_length = 0
        else:
            if current_length + sent_len > window_size:
                segments.append(current_segment)
                current_segment = sentence
                current_length = sent_len
            else:
                current_length += sent_len
                current_segment += sentence
    if current_segment:
        segments.append(current_segment)
    return [seg for seg in segments if not is_only_punctuation(seg)]
