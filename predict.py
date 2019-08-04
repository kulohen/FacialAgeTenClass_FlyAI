from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

data = Dataset()
model = Model(data)
p = model.predict_all([])
print(p)
