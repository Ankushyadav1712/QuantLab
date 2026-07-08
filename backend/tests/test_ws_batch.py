"""Streaming batch WebSocket (/ws/batch): pushes a result per alpha then a
final correlation matrix, and reports bad input as an error message (not a
crash). Uses the session `client` fixture so market data is loaded."""


def _drain(ws):
    msgs = []
    while True:
        m = ws.receive_json()
        msgs.append(m)
        if m["type"] in ("complete", "error"):
            return msgs


def test_ws_batch_streams_results_then_complete(client):
    with client.websocket_connect("/ws/batch") as ws:
        ws.send_json(
            {
                "alphas": [
                    {"id": "a", "expression": "rank(close)"},
                    {"id": "b", "expression": "rank(volume)"},
                ],
                "settings": {},
            }
        )
        msgs = _drain(ws)

    assert "error" not in [m["type"] for m in msgs], msgs
    results = [m for m in msgs if m["type"] == "result"]
    assert len(results) == 2
    assert results[0]["index"] == 0
    assert results[0]["total"] == 2
    assert results[1]["index"] == 1
    assert "id" in results[0]["row"]  # private "_returns" is stripped
    assert not any(k.startswith("_") for k in results[0]["row"])

    complete = msgs[-1]
    assert complete["type"] == "complete"
    assert complete["n_alphas"] == 2
    assert complete["n_ok"] == 2
    assert "correlation_matrix" in complete
    assert "universe_id" in complete


def test_ws_batch_reports_empty_alphas_as_error(client):
    with client.websocket_connect("/ws/batch") as ws:
        ws.send_json({"alphas": [], "settings": {}})
        m = ws.receive_json()
    assert m["type"] == "error"
    assert "No alphas" in m["detail"]


def test_ws_batch_rejects_malformed_payload(client):
    with client.websocket_connect("/ws/batch") as ws:
        ws.send_json({"nonsense": True})  # missing required 'alphas'
        m = ws.receive_json()
    assert m["type"] == "error"


def test_ws_batch_surfaces_bad_expression_per_row(client):
    # A single unparseable alpha becomes a per-row error, not a batch failure.
    with client.websocket_connect("/ws/batch") as ws:
        ws.send_json(
            {
                "alphas": [
                    {"id": "ok", "expression": "rank(close)"},
                    {"id": "bad", "expression": "this is not valid !!!"},
                ],
                "settings": {},
            }
        )
        msgs = _drain(ws)
    rows = {m["row"]["id"]: m["row"] for m in msgs if m["type"] == "result"}
    assert "metrics" in rows["ok"]
    assert "error" in rows["bad"]
    assert msgs[-1]["type"] == "complete"
    assert msgs[-1]["n_ok"] == 1
