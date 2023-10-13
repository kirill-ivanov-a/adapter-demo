import cv2
import numpy as np

from adapter_base import AdapterBase
from second_algo.detect_lines import Detector


class FirstAdapter(AdapterBase):
    def __init__(self, important_param):
        self.important_param = important_param

    def _detect(self, model, image):
        return model.detect_lines(image)

    def _transform_image(self, image: np.ndarray):
        transformed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return transformed_image

    def _build_model(self):
        return Detector(self.important_param)

    def _postprocess_predictions(self, raw_predictions):
        return raw_predictions


if __name__ == "__main__":
    image = np.array(
        [
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    adapter = FirstAdapter(42)
    result = adapter.detect(image)
    print(result)
