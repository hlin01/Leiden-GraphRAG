import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from graphrag_store import GraphRAGStore
from graphrag_queryengine import GraphRAGQueryEngine
from llama_index.core import PropertyGraphIndex


load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")


llm = OpenAI(model="gpt-4o", temperature=1)
embed_model = OpenAIEmbedding(model="text-embedding-3-small")


graph_store = GraphRAGStore(
    username="neo4j", password="abcd@1234", url="bolt://localhost:7687",
)


graph_store.load_state()
graph_store.llm = llm
graph_store.max_cluster_size = 5


index = PropertyGraphIndex.from_existing(
    llm=llm,
    property_graph_store=graph_store,
    embed_model=embed_model,
    show_progress=True,
)


# triplets = index.property_graph_store.get_triplets()
# print(triplets)
community_summaries = index.property_graph_store.get_community_summaries()
# print(community_summaries)
index.property_graph_store.save_state()


query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    index=index,
    llm=llm,
    similarity_top_k=10,
)


response = query_engine.query("What are some pressing issues CL is facing?")
print(response)
