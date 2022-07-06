import numpy as np
import PIL.Image

import bentoml

runner = bentoml.pytorch.get("pytorch_mnist:latest").to_runner()

img = PIL.Image.open("samples/0.png")
arr = np.array(img) / 255.0
arr = arr.astype("float32")

# add color channel dimension for greyscale image
arr = np.expand_dims(arr, 0)
arr = np.expand_dims(arr, 0)
runner.init_local()
print(runner.run(arr))  # => tensor(0)

