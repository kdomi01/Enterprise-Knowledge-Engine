import pytest
from fastapi import status

def test_search_query_endpoint_success(test_client):
    """
    Verifies that the hybrid search route successfully returns a structured results payload.
    """
    response = test_client.get("/api/v1/search/query?q=Python developer skills&limit=3")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "query" in data
    assert "matches" in data
    assert isinstance(data["matches"], list)

def test_search_query_empty_bad_request(test_client):
    """
    Guards against empty queries to ensure validation layer intercepts bad inputs.
    """
    response = test_client.get("/api/v1/search/query?q=   ")
    assert response.status_code == status.HTTP_400_BAD_REQUEST

def test_agent_graph_engine_post_success(test_client):
    """
    Verifies that the LangGraph orchestrator accepts structured JSON payloads
    and processes it through the compiled RAG tracks.
    """
    payload = {"q": "What computer tools are listed?"}
    response = test_client.post("/api/v1/query/engine", json=payload)
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert data["query"] == payload["q"]
    assert "generation" in data
    assert "retrieved_context" in data
    assert "visited_steps" in data
    
    # Assert that it successfully traveled through your updated Reranker tracks!
    assert "retrieve_node" in data["visited_steps"]
    assert "rerank_node" in data["visited_steps"]
    assert "generate_node" in data["visited_steps"]