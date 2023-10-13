import numpy as np

from abc import ABC, abstractmethod


class AdapterBase(ABC):
    """Base adapter"""

    def detect(self, image: np.ndarray):
        # image - RGB
        image = self._transform_image(image)
        model = self._build_model()
        raw_predictions = self._detect(model, image)
        predictions = self._postprocess_predictions(raw_predictions)
        # predictions - lines [x_start, y_start, x_end, y_end]
        return predictions

    @abstractmethod
    def _detect(self, model, image):
        pass

    @abstractmethod
    def _transform_image(self, image: np.ndarray):
        pass

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _postprocess_predictions(self, raw_predictions):
        pass
