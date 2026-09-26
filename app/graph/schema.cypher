// Abhilekh graph schema
// Run once via cypher-shell or the Neo4j Browser after Neo4j starts.
//
// Design rule: nodes here carry only IDs + light labels. Full passage
// text always stays in Postgres — the graph is for traversal, not for
// storing content. Every relationship that represents a claim (OPPOSED,
// SUPPORTED, etc.) carries a passage_id property so it can be traced
// back to the exact source text — this is what makes the graph
// trustworthy instead of just "an LLM said so".

CREATE CONSTRAINT person_id IF NOT EXISTS
FOR (n:Person) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT article_id IF NOT EXISTS
FOR (n:Article) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT topic_id IF NOT EXISTS
FOR (n:Topic) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT event_id IF NOT EXISTS
FOR (n:Event) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT act_id IF NOT EXISTS
FOR (n:Act) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT passage_id IF NOT EXISTS
FOR (n:Passage) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT document_id IF NOT EXISTS
FOR (n:Document) REQUIRE n.id IS UNIQUE;

// Allowed relationship types (enforced in code, not by Neo4j itself —
// Neo4j has no built-in relationship-type constraint):
//   (Person)-[:SPOKE]->(Passage)
//   (Passage)-[:PART_OF]->(Document)
//   (Passage)-[:DISCUSSES]->(Article|Topic|Act)
//   (Person)-[:OPPOSED {passage_id, confidence}]->(Article)
//   (Person)-[:SUPPORTED {passage_id, confidence}]->(Article)
//   (Article)-[:EVOLVED_FROM]->(Article)
//   (Event)-[:INVOLVED]->(Person)
