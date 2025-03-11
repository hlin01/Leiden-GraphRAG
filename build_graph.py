import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from graphrag_extractor import GraphRAGExtractor
from graphrag_store import GraphRAGStore
import re
from typing import Any
from llama_index.core import Document, PropertyGraphIndex
from neo4j import GraphDatabase


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")


llm = OpenAI(model="gpt-4o", temperature=1)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")


with open("anonymized_data_cl1.txt", "r") as f:
    notes = f.readlines()


documents = [Document(text=note.strip()) for note in notes]


splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=32,
)


nodes = splitter.get_nodes_from_documents(documents, show_progress=True)


KG_TRIPLET_EXTRACT_TMPL = """
Goal:
Extract up to {max_knowledge_triplets} entity-relation triplets from the provided text. For each triplet, identify entities (along with their types and descriptions) and determine the relationships between them.

Instructions:

1. Entity Extraction:
   - Identify all entities mentioned in the text.
   - For each entity, extract:
      - entity_name: The name of the entity (ensure it is properly capitalized).
      - entity_type: The category or type of the entity.
      - entity_description: A detailed description covering the entity’s attributes and activities.
   - Format each entity with no additional formatting as:
      ([ENTITY] | <entity_name> | <entity_type> | <entity_description>)

2. Relationship Extraction:
   - From the entities identified in step 1, find all pairs of entities that have a clear, meaningful relationship.
   - For each related pair, extract:
      - source_entity: The originating entity (as identified in step 1).
      - target_entity: The related entity (as identified in step 1).
      - relation: The type or nature of the relationship.
      - relationship_description: A brief explanation detailing why the relationship exists.
   - Format each relationship with no additional formatting as:
      ([RELATIONSHIP] | <source_entity> | <target_entity> | <relation> | <relationship_description>)

3. Output:
   - Provide the extracted entities and relationships in the given formats exactly as specified.

Actual Data:
====================
Text: {text}

====================
Output:
"""


entity_pattern = r'\(\[ENTITY\] \| (.+?) \| (.+?) \| (.+?)\)'
relationship_pattern = r'\(\[RELATIONSHIP\] \| (.+?) \| (.+?) \| (.+?) \| (.+?)\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    parse_fn=parse_fn,
    max_paths_per_chunk=5,
)


def clear_graph(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()


clear_graph("bolt://localhost:7687", "neo4j", "abcd@1234")


graph_store = GraphRAGStore(
    username="neo4j", password="abcd@1234", url="bolt://localhost:7687",
)
graph_store.llm = llm
graph_store.max_cluster_size = 5


index = PropertyGraphIndex(
    nodes=nodes,
    llm=llm,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    embed_model=embed_model,
    show_progress=True,
)
