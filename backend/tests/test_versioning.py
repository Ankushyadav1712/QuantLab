"""Alpha versioning + tags: name-based lineage, heads-only list, ?tag= filter,
version history, and non-destructive rollback."""


def _save(client, name, expression, tags=None):
    r = client.post(
        "/api/alphas",
        json={"name": name, "expression": expression, "notes": "ci", "tags": tags or []},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _cleanup(client, name):
    """Delete every version in the named lineage (heads-only list → walk versions)."""
    heads = [a for a in client.get("/api/alphas").json() if a["name"] == name]
    for h in heads:
        versions = client.get(f"/api/alphas/{h['id']}/versions").json()["versions"]
        for ver in versions:
            client.delete(f"/api/alphas/{ver['id']}")


def test_versioning_lineage_and_rollback(client):
    name = "_ci_ver_alpha"
    _cleanup(client, name)
    try:
        v1 = _save(client, name, "rank(close)", tags=["momentum", "ci"])
        assert v1["version"] == 1
        assert v1["tags"] == ["momentum", "ci"]

        v2 = _save(client, name, "rank(ts_mean(close, 5))", tags=["ci"])
        assert v2["version"] == 2

        # List is heads-only: one row for this name, at version 2, count 2.
        rows = [a for a in client.get("/api/alphas").json() if a["name"] == name]
        assert len(rows) == 1
        assert rows[0]["version"] == 2
        assert rows[0]["version_count"] == 2
        assert rows[0]["expression"] == "rank(ts_mean(close, 5))"  # head = latest

        # Version history: full lineage, oldest → newest, parent linkage set.
        versions = client.get(f"/api/alphas/{v2['id']}/versions").json()
        assert versions["name"] == name
        assert [x["version"] for x in versions["versions"]] == [1, 2]
        assert versions["versions"][1]["parent_id"] == v1["id"]

        # Non-destructive rollback to v1 → appends v3 carrying v1's expression.
        rb = client.post(f"/api/alphas/{v2['id']}/rollback/1")
        assert rb.status_code == 200, rb.text
        assert rb.json()["restored_from_version"] == 1
        assert rb.json()["version"] == 3
        head = client.get(f"/api/alphas/{rb.json()['id']}").json()
        assert head["expression"] == "rank(close)"
        assert head["version"] == 3
    finally:
        _cleanup(client, name)


def test_tag_filter_uses_head_tags(client):
    name = "_ci_tag_alpha"
    _cleanup(client, name)
    try:
        # v1 tagged momentum; v2 (the head) tagged only ci.
        _save(client, name, "rank(close)", tags=["momentum"])
        _save(client, name, "rank(volume)", tags=["ci"])

        mom = client.get("/api/alphas?tag=momentum").json()
        assert not any(a["name"] == name for a in mom)  # head no longer has 'momentum'
        ci = client.get("/api/alphas?tag=ci").json()
        assert any(a["name"] == name for a in ci)

        # get_alpha parses tags into a list.
        head_id = next(a["id"] for a in client.get("/api/alphas").json() if a["name"] == name)
        got = client.get(f"/api/alphas/{head_id}").json()
        assert got["tags"] == ["ci"]
    finally:
        _cleanup(client, name)


def test_rollback_missing_version_is_404(client):
    name = "_ci_rb_alpha"
    _cleanup(client, name)
    try:
        saved = _save(client, name, "rank(close)")
        assert client.post(f"/api/alphas/{saved['id']}/rollback/99").status_code == 404
    finally:
        _cleanup(client, name)


def test_versions_404_for_unknown_alpha(client):
    assert client.get("/api/alphas/99999999/versions").status_code == 404
