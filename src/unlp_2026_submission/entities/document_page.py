from llama_index.core.schema import NodeWithScore


class DocumentPage:
    document_id: str
    text: str
    page_number: int

    def __init__(
            self,
            document_id: str,
            text: str,
            page_number: int,
    ):
        self.document_id = document_id
        self.text = text
        self.page_number = page_number

    @classmethod
    def get_from_node_with_sore(cls, node: NodeWithScore):
        return DocumentPage(
            document_id=node.metadata['file_name'],
            text=node.text,
            page_number=int(node.metadata['page_label']),
        )
