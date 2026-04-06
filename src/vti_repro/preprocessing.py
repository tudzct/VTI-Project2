"""Code preprocessing helpers for the classical baseline."""

from __future__ import annotations

import re

_COMMENT_PATTERN = re.compile(
    r"//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|\"(?:\\.|[^\\\"])*\"",
    re.DOTALL | re.MULTILINE,
)


def remove_comments(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        matched = match.group(0)
        if matched.startswith("/"):
            return " "
        return matched

    return re.sub(_COMMENT_PATTERN, replacer, text)


def replace_string_literals(text: str) -> str:
    return re.sub(r"\".*?\"", " strlitplaceholder ", text)


def split_identifier(identifier: str) -> list[str]:
    identifier = re.sub(r"[^a-zA-Z0-9_]", " ", identifier)
    parts = [part for part in re.split(r"[\s_]+", identifier) if part]
    subtokens: list[str] = []
    for part in parts:
        if part.isdigit():
            continue
        split_part = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", part))
        for token in split_part.split():
            token = re.sub(r"[^a-zA-Z0-9]", "", token).lower()
            if token:
                subtokens.append(token)
    return subtokens


def preprocess_code(text: str) -> str:
    text = remove_comments(text)
    text = replace_string_literals(text)
    text = text.replace("\x00", " ")
    tokens: list[str] = []
    for raw_token in re.split(r"\s+", text):
        tokens.extend(split_identifier(raw_token))
    return " ".join(tokens)


def minimal_clean_text(text: str) -> str:
    return re.sub(r"\s+\Z", "", (text or "").replace("\r\n", "\n").replace("\r", "\n").strip())
