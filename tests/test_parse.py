import pytest

from discovery import prompts


def test_parse_tolerates_fences_prose_and_duplicates():
    text = 'Sure:\n```json\n{"candidates": [{"name": "a", "code": "```python\\nx = 1\\n```"}, {"name": "a", "code": "y = 2"}]}\n```'
    c = prompts.parse_response(text)
    assert [x["name"] for x in c] == ["a", "a_1"] and c[0]["code"] == "x = 1"


@pytest.mark.parametrize("text", ["no json here", '{"candidates": []}', '{"candidates": [{"name": "x", "code": "  "}]}'])
def test_parse_errors(text):
    with pytest.raises(prompts.ParseError):
        prompts.parse_response(text)
