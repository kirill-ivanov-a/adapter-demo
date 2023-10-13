import numpy as np

from PIL import Image

from adapter_base import AdapterBase
from first_algo.detect_lines import detect_lines


class FirstAdapter(AdapterBase):
    def _detect(self, model, image):
        return model(image)

    def _transform_image(self, image: np.ndarray):
        transformed_image = Image.fromarray(image.astype("uint8"), "RGB")
        return transformed_image

    def _build_model(self):
        return detect_lines

    def _postprocess_predictions(self, raw_predictions):
        return raw_predictions.reshape(-1, 4)


if __name__ == "__main__":
    image = np.array(
        [
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
            [[0, 255, 255], [255, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    adapter = FirstAdapter()
    result = adapter.detect(image)
    print(result)
