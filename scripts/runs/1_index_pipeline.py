from langgraph.graph import StateGraph, START, END

from merlin.rag.index import IndexState
from merlin.rag.index.nodes import DoclingLoadSplitNode, EmbedStoreNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec


def build_index_graph():
    graph = StateGraph(IndexState)

    spec = EmbeddingsSpec(
        provider="huggingface",
        model_name="bflhc/Octen-Embedding-0.6B",
    )
    embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

    load_split_node = DoclingLoadSplitNode(embeddings)
    embed_store_node = EmbedStoreNode(embeddings)

    graph.add_node("load_split", load_split_node)
    graph.add_node("embed_store", embed_store_node)

    graph.add_edge(START, "load_split")
    graph.add_edge("load_split", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()


index_app = build_index_graph()

initial_state: IndexState = {
    "filepaths": ["../documents/test.pdf"],
    "vector_store_path": "../vector_dbs/qdrant_db"
}

final_state = index_app.invoke(initial_state)

print("SUCKS ASS")