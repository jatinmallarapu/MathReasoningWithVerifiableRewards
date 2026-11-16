import re
from fractions import Fraction

FORMAT = re.compile(r"<think>\s*(.+?)\s*</think>\s*<answer>\s*(.+?)\s*</answer>", re.S)


def number(text):
    clean = text.strip().replace(",", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)", clean):
        return None
    try:
        return Fraction(clean)
    except (ValueError, ZeroDivisionError):
        return None


def prediction(text):
    match = FORMAT.fullmatch(text.strip())
    if match:
        return number(match[2])
    tagged = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    if tagged:
        return number(tagged[-1])
    if "####" in text:
        return number(text.rsplit("####", 1)[-1])
    return number(text.strip().splitlines()[-1]) if text.strip() else None


def correctness(completions, answer, **_):
    return [float(prediction(c[0]["content"]) == number(a) and number(a) is not None)
            for c, a in zip(completions, answer, strict=True)]


def format_reward(completions, **_):
    return [0.25 * bool(FORMAT.fullmatch(c[0]["content"].strip())) for c in completions]
