from langgraph.graph import StateGraph, START, END

from merlin.rag.index import IndexState
from merlin.rag.index.nodes import EmbedStoreNode, PyPDFLoadNode, SplitNode
from merlin.models.embeddings import EmbeddingsFactory, EmbeddingsSpec


def build_index_graph():
    """
    Index pipeline configuration:
        Load: PyPDFLoader
        Split: RecursiveCharacterTextSplitter(size=400, overlap=0, start_index=true)
        Embed+Store: Qdrand + Octen-Embedding-0.6B
    """
    graph = StateGraph(IndexState)

    spec = EmbeddingsSpec(
        provider="huggingface",
        model_name="bflhc/Octen-Embedding-0.6B",
    )
    embeddings = EmbeddingsFactory.create_all_embeddings_factory().create(spec)

    load_node = PyPDFLoadNode()
    split_node = SplitNode(
        RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            add_start_index=True,
        )
    )
    embed_store_node = EmbedStoreNode(embeddings)

    graph.add_node("load", load_node)
    graph.add_node("split", split_node)
    graph.add_node("embed_store", embed_store_node)

    graph.add_edge(START, "load")
    graph.add_edge("load", "split")
    graph.add_edge("split", "embed_store")
    graph.add_edge("embed_store", END)

    return graph.compile()


index_app = build_index_graph()

initial_state: IndexState = {
    "filepaths": ["../documents/test.pdf"],
    "vector_store_path": "../vector_dbs/qdrant_db"
}

final_state = index_app.invoke(initial_state)

# TODO: ADD RAG Runner
