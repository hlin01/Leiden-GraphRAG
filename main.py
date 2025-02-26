import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from llama_index.core.node_parser import SentenceSplitter
from graphrag_extractor import GraphRAGExtractor
from graphrag_store import GraphRAGStore
from graphrag_queryengine import GraphRAGQueryEngine
import re
from typing import Any
from llama_index.core import Document

from llama_index.core import PropertyGraphIndex

load_dotenv()
os.environ["OPENAI_API_KEY"]

llm = OpenAI(model="gpt-4o", temperature=0)

with open("anonymized_data_cl1.txt", "r") as f:
    notes = f.readlines()

documents = [Document(text=note.strip()) for note in notes]

splitter = SentenceSplitter(
    chunk_size=1024,
    chunk_overlap=32,
)

nodes = splitter.get_nodes_from_documents(documents)
print(len(nodes))

KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"$$$$""$$$$""$$$$"")

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

Format each relationship as ("relationship"$$$$""$$$$""$$$$""$$$$"")

3. When finished, output.

-Real Data-
######################
text: {text}
######################
output:
"""

entity_pattern = r'\("entity"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'
relationship_pattern = r'\("relationship"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\$\$\$\$"(.+?)"\)'


def parse_fn(response_str: str) -> Any:
    entities = re.findall(entity_pattern, response_str)
    relationships = re.findall(relationship_pattern, response_str)
    return entities, relationships


kg_extractor = GraphRAGExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
    max_paths_per_chunk=2,
    parse_fn=parse_fn,
)

graph_store = GraphRAGStore(
    username="neo4j", password="abcd@1234", url="bolt://localhost:7687"
)

index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store,
    show_progress=True,
)

print(index.property_graph_store.get_triplets()[10])
print(index.property_graph_store.get_triplets()[10][0].properties)
print(index.property_graph_store.get_triplets()[10][1].properties)

index.property_graph_store.build_communities()

query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    index=index,
    similarity_top_k=10,
)

response = query_engine.query(
    "What are the main ideas discussed in the document?"
)
print(response)
