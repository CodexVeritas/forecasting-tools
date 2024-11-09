from typing import Any


def determine_percent_correct(actual: list[Any], expected: list[Any]) -> float:
    if len(actual) != len(expected):
        raise ValueError(
            f"Length of actual ({len(actual)}) does not match length of expected ({len(expected)})"
        )

    number_of_correct: int = 0
    for i in range(len(actual)):
        if actual[i] == expected[i]:
            number_of_correct += 1
    percent_correct: float = number_of_correct / len(actual)

    return percent_correct
