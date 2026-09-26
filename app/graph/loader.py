"""
Two jobs:

1. review_pending() — a bare-bones CLI reviewer. For the hackathon demo,
   your team is the reviewer: read each candidate edge with its source
   sentence, approve or reject. This is what you'd point to if a judge
   asks "how do you stop the AI from inventing history?"

2. load_approved_to_neo4j() — pushes only status='approved' rows into
   Neo4j, plus the base Person/Article/Document/Passage nodes. Re-running
   this is safe (MERGE is idempotent) — it's how the graph gets rebuilt
   if you ever need to start over.
"""

from app.db import get_conn
from app.graph import get_driver


def review_pending() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.edge_id, e.subj_id, e.predicate, e.obj_id,
                       e.confidence, p.text, p.page_start
                FROM edges_staging e
                JOIN passages p ON p.passage_id = e.passage_id
                WHERE e.status = 'pending'
                ORDER BY e.edge_id
                """
            )
            rows = cur.fetchall()

        if not rows:
            print("Nothing pending.")
            return

        for r in rows:
            print("\n" + "-" * 70)
            print(f"[{r['edge_id']}] {r['subj_id']}  --{r['predicate']}-->  Article {r['obj_id']}"
                  f"  (confidence {r['confidence']:.2f}, page {r['page_start']})")
            print(f"Source: {r['text'][:300]}...")
            choice = input("Approve? [y/n/skip]: ").strip().lower()
            if choice == "y":
                _set_status(conn, r["edge_id"], "approved")
            elif choice == "n":
                _set_status(conn, r["edge_id"], "rejected")
            # anything else = skip, leave pending


def _set_status(conn, edge_id: int, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE edges_staging SET status = %s, reviewed_by = %s WHERE edge_id = %s",
            (status, "team-review", edge_id),
        )
    conn.commit()


def load_approved_to_neo4j() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Base nodes first
            cur.execute("SELECT speaker_id, canonical_name FROM speakers")
            people = cur.fetchall()

            cur.execute(
                """
                SELECT DISTINCT unnest(article_refs) AS article_number
                FROM passages WHERE array_length(article_refs, 1) > 0
                """
            )
            articles = cur.fetchall()

            cur.execute(
                """
                SELECT passage_id, doc_id, speaker_id, page_start
                FROM passages
                """
            )
            passages = cur.fetchall()

            cur.execute(
                """
                SELECT subj_id, predicate, obj_id, passage_id, confidence
                FROM edges_staging WHERE status = 'approved'
                """
            )
            approved = cur.fetchall()

    driver = get_driver()
    with driver.session() as session:
        for p in people:
            session.run(
                "MERGE (n:Person {id: $id}) SET n.name = $name",
                id=p["speaker_id"], name=p["canonical_name"],
            )
        for a in articles:
            session.run(
                "MERGE (n:Article {id: $id}) SET n.number = $id",
                id=a["article_number"],
            )
        for pg in passages:
            session.run(
                """
                MERGE (n:Passage {id: $id})
                SET n.doc_id = $doc_id, n.page_start = $page_start
                """,
                id=pg["passage_id"], doc_id=pg["doc_id"], page_start=pg["page_start"],
            )
            if pg["speaker_id"]:
                session.run(
                    """
                    MATCH (p:Person {id: $speaker_id}), (ps:Passage {id: $passage_id})
                    MERGE (p)-[:SPOKE]->(ps)
                    """,
                    speaker_id=pg["speaker_id"], passage_id=pg["passage_id"],
                )

        for e in approved:
            # predicate is validated against ALLOWED_PREDICATES before this
            # point (extract.py), so it's safe to interpolate here.
            session.run(
                f"""
                MATCH (subj:Person {{id: $subj_id}}), (obj:Article {{id: $obj_id}})
                MERGE (subj)-[r:{e['predicate']}]->(obj)
                SET r.passage_id = $passage_id, r.confidence = $confidence
                """,
                subj_id=e["subj_id"], obj_id=e["obj_id"],
                passage_id=e["passage_id"], confidence=e["confidence"],
            )

    print(f"Loaded {len(people)} people, {len(articles)} articles, "
          f"{len(passages)} passages, {len(approved)} approved edges into Neo4j.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        review_pending()
    else:
        load_approved_to_neo4j()
