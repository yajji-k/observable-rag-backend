from app.services.reranking.base import BaseReranker


class RerankerRegistry:

    _rerankers = {
        "bge": "app.services.reranking.bge_reranker.BGEReranker",
    }
    _instances: dict[str, BaseReranker] = {}

    @classmethod
    def get_available_rerankers(cls) -> list[str]:
        return list(cls._rerankers.keys())

    @classmethod
    def get(
        cls,
        model: str
    ) -> BaseReranker:
        reranker_class = cls._rerankers.get(model)

        if reranker_class is None:
            available = ", ".join(cls._rerankers.keys())

            raise ValueError(
                f"Unsupported reranker '{model}'. "
                f"Available rerankers: {available}"
            )

        if model not in cls._instances:
            cls._instances[model] = (
                cls._load_reranker_class(
                    reranker_class
                )()
            )

        return cls._instances[model]

    @staticmethod
    def _load_reranker_class(
        import_path: str
    ) -> type[BaseReranker]:
        module_path, class_name = import_path.rsplit(
            ".",
            1
        )
        module = __import__(
            module_path,
            fromlist=[class_name]
        )

        return getattr(module, class_name)
