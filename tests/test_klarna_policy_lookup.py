"""A natural policy query reaches the published policy document."""

import pytest

from mini_agents.domains.klarna.state import KlarnaState
from mini_agents.domains.klarna.tools import KlarnaToolExecutor
from mini_agents.domains.klarna.tools_unsafe import (
    KlarnaToolExecutor as UnsafeKlarnaToolExecutor,
)

REFUND_QUERIES = (
    "refund policy",
    "what is the refund policy?",
    "refund policy details",
)


def _retrieve(executor_type, query):
    return executor_type(KlarnaState()).execute("retrieve_policy", {"query": query})


def _doc_ids(result):
    return [document["doc_id"] for document in result["documents"]]


@pytest.mark.parametrize("query", REFUND_QUERIES)
def test_a_natural_refund_query_returns_the_published_refund_policy(query):
    result = _retrieve(KlarnaToolExecutor, query)

    assert result["status"] == "OK"
    assert "POL-REFUND" in _doc_ids(result)
    assert "POL-FEES-DRAFT" not in _doc_ids(result)


@pytest.mark.parametrize("query", REFUND_QUERIES)
def test_the_unsafe_twin_returns_the_same_documents(query):
    safe = _retrieve(KlarnaToolExecutor, query)
    unsafe = _retrieve(UnsafeKlarnaToolExecutor, query)

    assert unsafe["status"] == "OK"
    assert _doc_ids(unsafe) == _doc_ids(safe)
    assert "POL-FEES-DRAFT" not in _doc_ids(unsafe)


def test_a_query_with_no_shared_word_misses():
    result = _retrieve(KlarnaToolExecutor, "late fees")

    assert result["status"] == "NO_WHITELIST_HIT"
    assert result["documents"] == []


def test_the_draft_policy_stays_out_of_safe_results():
    result = _retrieve(KlarnaToolExecutor, "credit apr")

    assert result["status"] == "NO_WHITELIST_HIT"
    assert result["documents"] == []


def test_multiple_hits_keep_the_seeded_order():
    first = _retrieve(KlarnaToolExecutor, "refund schedule")
    second = _retrieve(KlarnaToolExecutor, "refund schedule")

    assert _doc_ids(first) == ["POL-REFUND", "POL-SCHEDULE"]
    assert _doc_ids(second) == _doc_ids(first)
