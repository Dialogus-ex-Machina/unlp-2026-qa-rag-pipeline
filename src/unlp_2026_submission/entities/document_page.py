from llama_index.core.schema import NodeWithScore


class DocumentPage:
    file_name: str
    text: str
    page_label: str
    file_path: str

    def __init__(
            self,
            file_name: str,
            text: str,
            page_label: str,
            file_path: str
    ):
        self.file_name = file_name
        self.text = text
        self.page_label = page_label
        self.file_path = file_path

    @classmethod
    def get_from_node_with_sore(cls, node: NodeWithScore):
        return DocumentPage(
            file_name=node.metadata['file_name'],
            text=node.text,
            page_label=node.metadata['page_label'],
            file_path=node.metadata['file_path'],
        )

    @property
    def document_id(self):
        return self.file_name

    @property
    def page_number(self):
        try:
            return int(self.page_label)
        except Exception as e:
            return -1
