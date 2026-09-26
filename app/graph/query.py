from app.graph import get_driver


def article_subgraph(article_number: str) -> dict:
    """Everything connected to one Article: who supported/opposed it,
    and which passages discuss it. This is what renders as the visual
    graph on the kiosk's Article Explorer screen."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:Article {id: $num})
            OPTIONAL MATCH (person:Person)-[r]->(a)
            OPTIONAL MATCH (person)-[:SPOKE]->(ps:Passage)
            RETURN a, collect(DISTINCT {person: person, rel: type(r),
                                         passage_id: r.passage_id,
                                         confidence: r.confidence}) AS edges
            """,
            num=article_number,
        )
        record = result.single()

    if not record or record["a"] is None:
        return {"nodes": [], "edges": []}

    nodes = [{"id": article_number, "type": "Article", "label": f"Article {article_number}"}]
    edges = []
    seen_people = set()

    for e in record["edges"]:
        if e["person"] is None or e["rel"] is None:
            continue
        pid = e["person"]["id"]
        if pid not in seen_people:
            nodes.append({"id": pid, "type": "Person", "label": e["person"].get("name", pid)})
            seen_people.add(pid)
        edges.append({
            "source": pid,
            "target": article_number,
            "type": e["rel"],
            "passage_id": e["passage_id"],
        })

    return {"nodes": nodes, "edges": edges}


def passages_for_article(article_number: str) -> list[str]:
    """Used by search.graph_expand: passage IDs connected to an Article
    via an approved relationship, so a text-only search that missed the
    right vocabulary still surfaces them."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (:Article {id: $num})<-[r]-(person:Person)-[:SPOKE]->(ps:Passage)
            WHERE r.passage_id = ps.id
            RETURN DISTINCT ps.id AS passage_id
            LIMIT 10
            """,
            num=article_number,
        )
        return [r["passage_id"] for r in result]
