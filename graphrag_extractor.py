import asyncio
from typing import Any, List, Callable, Optional, Union
from llama_index.core.async_utils import run_jobs
from llama_index.core.indices.property_graph.utils import default_parse_triplets_fn
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode
from llama_index.core import Settings


class GraphRAGExtractor(TransformComponent):
    """
    Extract triples from a graph using an LLM and a simple prompt/output parsing approach.
    
    This class extracts entity and relation triples (subject, predicate, object)
    from a node's text content and appends the resulting knowledge graph nodes
    and relations to the node's metadata.
    """

    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int


    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = default_parse_triplets_fn,
        max_paths_per_chunk: int = 5,
        num_workers: int = 5,
    ) -> None:        
        """
        Initialize the GraphRAGExtractor.

        Args:
            llm (LLM): The language model to use. If None, a default is used.
            extract_prompt (str or PromptTemplate): The prompt to guide the LLM.
            parse_fn (callable): Function to parse the LLM output.
            max_paths_per_chunk (int): Maximum number of paths/triples per node.
            num_workers (int): Number of parallel workers for asynchronous processing.
        """

        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)

        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk,
        )


    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"


    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Synchronously extract triples from a list of nodes.
        """
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )


    async def _aextract(self, node: BaseNode) -> BaseNode:
        """
        Asynchronously extract triples from a single node.
        """
        assert hasattr(node, "text")

        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        entity_metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            entity_metadata["entity_description"] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=entity_metadata
            )
            existing_nodes.append(entity_node)

        relation_metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, obj, rel, description = triple
            relation_metadata["relationship_description"] = description
            rel_node = Relation(
                label=rel,
                source_id=subj,
                target_id=obj,
                properties=relation_metadata,
            )
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node


    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Asynchronously extract triples from a list of nodes.
        """
        jobs = [self._aextract(node) for node in nodes]
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text",
        )
