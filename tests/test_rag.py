import pytest

from fitness_agent.rag.chunker import chunk_pages, split_text
from fitness_agent.rag.context_builder import build_context
from fitness_agent.rag.models import Page, RetrievedChunk


def test_split_text_overlaps_chunks():
    text = " ".join(str(number) for number in range(10))

    assert split_text(text, chunk_size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_split_text_keeps_short_text_whole():
    assert split_text("just a few words", chunk_size=50, overlap=10) == ["just a few words"]


def test_split_text_rejects_overlap_as_big_as_chunk():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=5, overlap=5)


def test_chunk_pages_keeps_source_and_page():
    pages = [Page(text="one two three four five", source="guide.pdf", page=3)]

    chunks = chunk_pages(pages, chunk_size=3, overlap=1)

    assert [chunk.id for chunk in chunks] == ["guide.pdf-p3-c0", "guide.pdf-p3-c1"]
    assert all(chunk.source == "guide.pdf" and chunk.page == 3 for chunk in chunks)


def test_build_context_numbers_chunks_with_sources():
    chunks = [
        RetrievedChunk(text="Squats build legs.", source="guide.pdf", page=2, distance=0.1),
        RetrievedChunk(text="Sleep aids recovery.", source="rest.pdf", page=5, distance=0.2),
    ]

    context = build_context(chunks)

    assert context == (
        "[1] Source: guide.pdf, page 2\nSquats build legs.\n\n"
        "[2] Source: rest.pdf, page 5\nSleep aids recovery."
    )
