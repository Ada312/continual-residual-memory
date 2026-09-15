import logging

import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    _available = {}

    @classmethod
    def register(cls, new_cls):
        cls_name = new_cls.__name__
        if cls_name in cls._available:
            raise NameError(
                f"Cannot register two models with the name '{cls_name}'")

        cls._available[cls_name] = new_cls

    def __getitem__(self, item):
        model_name, cfg = item
        if model_name not in self._available:
            raise ValueError(
                f"Model {model_name} unknown. Choose from {self._available.keys()}")

        model_cls = self._available[model_name]

        model = model_cls(cfg)

        if cfg.load is not None:
            logger.info('Loading Model weights')
            map_location = None if torch.cuda.is_available() else torch.device('cpu')
            state_dict = torch.load(cfg.load, map_location=map_location, weights_only=True)

            model.load_state_dict(state_dict)

        return model
