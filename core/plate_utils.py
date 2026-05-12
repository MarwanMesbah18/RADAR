from dataclasses import dataclass


@dataclass
class PlateChars:
    numbers: list
    letters: list
    text: str


def separate_chars(detections):
    """Separate detections into numbers and Arabic letters (RTL reversed).

    Returns PlateChars with numbers (LTR), letters (RTL), and combined text.
    """
    numbers = [d.class_name for d in detections if d.class_name.isdigit()]
    letters = [d.class_name for d in detections if not d.class_name.isdigit()]
    letters.reverse()

    parts = []
    if numbers:
        parts.append(" ".join(numbers))
    if letters:
        parts.append(" ".join(letters))

    return PlateChars(
        numbers=numbers,
        letters=letters,
        text=" | ".join(parts),
    )
