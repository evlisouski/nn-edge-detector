# nn-edge-detector

В данном репозитории представлена кастомная модель нейронной сети для детекции граней объектов.<br>
В директории `datasets` расположен аугментированный набор данных `BSDS500` с преобразованными `*.mat` файлами масок в `*.png` изображения.<br>
Предобученные модели находятся в директории `saved_models`.<br>

Корневую директорию репозитория рекомендуется использовать в качестве рабочей для интерпретатора python.<br>
- `train.py` - обучение модели.
- `predict` - инференс модели.
- `comparison_NN_and_Canny.py` - выводит три изображения: оригинал, после предобученной нейронной сети, после фильтра Canny. 
- `nn-edge-detector.ipynb` - notebook со всеми этапами создания, обучения, инференса нейронной сети и сравнения результатов с Canny.


Ссылка на notebook с созданием, обучением, инференсом нейронной сети для детекции граней объектов и сравнение результатов с работой Canny edge detector:<br>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/evlisouski/nn-edge-detector/blob/main/nn-edge-detector.ipynb)

![Описание изображения](./docs/img.png)