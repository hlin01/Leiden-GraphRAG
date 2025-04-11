import os
import re
import pickle
import networkx as nx
from graspologic.partition import hierarchical_leiden
from collections import defaultdict
from llama_index.core.llms import ChatMessage
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class GraphRAGStore(Neo4jPropertyGraphStore):
    """
    A property graph store that builds communities from the graph
    and generates summaries for each community using an LLM.
    """

    llm = None
    community_summary = {}
    entity_info = None
    max_cluster_size = 10
    resolution = 1.0
    randomness = 0.001


    def generate_community_summary(self, text):
        """
        Generate a summary for a given text using an LLM.

        Args:
            text (str): The text containing relationship details.

        Returns:
            str: A clean summary generated by the LLM.
        """
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as 'entity1 -> entity2 -> relation -> relationship_description'. Your task is to create a comprehensive summary of these relationships. The summary should include the names of the entities involved and a concise synthesis of the relationship descriptions, capturing the most critical and relevant details that highlight the nature and significance of each relationship. Ensure that the summary is coherent, integrates the provided information, and emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response


    def _summarize_communities(self, community_info):
        """
        Generate and store summaries for each community.

        Args:
            community_info (dict): Mapping from community IDs to relationship details.
        """
        for community_id, details in community_info.items():
            details_text = "\n".join(details)
            self.community_summary[community_id] = self.generate_community_summary(details_text)


    def _create_nx_graph(self):
        """
        Converts the internal graph representation to a NetworkX graph.

        Returns:
            networkx.Graph: The generated graph.
        """
        nx_graph = nx.Graph()
        triplets = self.get_triplets()
        for entity1, relation, entity2 in triplets:
            nx_graph.add_node(entity1.name)
            nx_graph.add_node(entity2.name)
            nx_graph.add_edge(
                relation.source_id,
                relation.target_id,
                relationship=relation.label,
                description=relation.properties["relationship_description"],
            )
        return nx_graph


    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on its community,
        allowing entities to belong to multiple clusters.

        Args:
            nx_graph (networkx.Graph): The graph.
            clusters (iterable): Cluster items with 'node' and 'cluster' attributes.

        Returns:
            tuple: A tuple containing:
                - entity_info (dict): Mapping from node to list of cluster IDs.
                - community_info (dict): Mapping from cluster IDs to relationship details.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = (
                        f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    )
                    community_info[cluster_id].append(detail)

        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)


    def build_communities(self):
        """
        Builds communities from the graph and summarizes them.
        """
        nx_graph = self._create_nx_graph()
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size, resolution=self.resolution, randomness=self.randomness,
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters,
        )
        self._summarize_communities(community_info)


    def get_community_summaries(self):
        """
        Returns the community summaries, building them if not already done.

        Returns:
            dict: The community summaries.
        """
        if not self.community_summary:
            self.build_communities()
        return self.community_summary


    def load_state(self, state_file="state.pkl"):
        if os.path.exists(state_file):
            try:
                with open(state_file, "rb") as f:
                    state = pickle.load(f)
                self.community_summary = state.get("community_summary", {})
                self.entity_info = state.get("entity_info", None)
                print(f"State loaded successfully from {state_file}.")
            except Exception as e:
                print(f"An error occurred while loading state: {e}")
        else:
            print("No saved state file found.")


    def save_state(self, state_file="state.pkl"):
        state = {
            "community_summary": self.community_summary,
            "entity_info": self.entity_info,
        }
        try:
            with open(state_file, "wb") as f:
                pickle.dump(state, f)
            print(f"State saved successfully to {state_file}.")
        except Exception as e:
            print(f"An error occurred while saving state: {e}")
