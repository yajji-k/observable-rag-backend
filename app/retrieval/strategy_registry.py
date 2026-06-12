class StrategyRegistry:

    COLLECTIONS = {
        "character": "rag_documents_character",
        "recursive": "rag_documents_recursive",
        "token": "rag_documents_token",
        "semantic": "rag_documents_semantic"
    }

    @classmethod
    def get_collection_name(
        cls,
        strategy: str
    ) -> str:

        if strategy not in cls.COLLECTIONS:
            raise ValueError(
                f"Unsupported chunk strategy: {strategy}"
            )

        return cls.COLLECTIONS[strategy]
    
    @classmethod
    def get_all_strategies(cls):
        return cls.COLLECTIONS.keys()