from utils.image import get_score
import pytest


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # (1, 100),
        # (2, 168),
        # (3, 168),
        # (4, 122),  # Manually count
        (5, 173),
        (6, 212),
        (7, 127),
        (8, 50),
    ],
)
def test_get_score(test_input, expected):
    assert get_score(f"test-images/test-image-{test_input}.png") == expected
