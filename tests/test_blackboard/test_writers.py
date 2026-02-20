"""Tests for built-in code-computed blackboard writers."""

from doc2md.blackboard.writers import (
    count_tables,
    detect_continuations,
    get_writer,
    list_writers,
)


class TestDetectContinuations:
    def test_ends_mid_table(self):
        assert detect_continuations("| Col A | Col B |", page_num=1) is True

    def test_ends_with_period(self):
        assert detect_continuations("This is a complete sentence.", page_num=1) is False

    def test_ends_mid_sentence(self):
        assert detect_continuations("This sentence does not end", page_num=1) is True

    def test_empty_text(self):
        assert detect_continuations("", page_num=1) is False

    def test_ends_with_question_mark(self):
        assert detect_continuations("Is this complete?", page_num=1) is False


class TestCountTables:
    def test_no_tables(self):
        assert count_tables("Just plain text.\n\nMore text.") == 0

    def test_one_table(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert count_tables(md) == 1

    def test_two_tables(self):
        md = "| A | B |\n|---|---|\n| 1 | 2 |\n\n| X | Y |\n|---|---|\n| 3 | 4 |"
        assert count_tables(md) == 2


class TestWriterRegistry:
    def test_writers_registered(self):
        names = list_writers()
        assert "detect_continuations" in names
        assert "count_tables" in names

    def test_get_writer(self):
        entry = get_writer("detect_continuations")
        assert entry is not None
        fn, key = entry
        assert callable(fn)

    def test_get_unknown_writer(self):
        assert get_writer("nonexistent") is None
