import pandas as pd
import dill
import os
import glob
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    # Загружаем обученную модель - самую свежую в нашей папке models.
    # Имя явно не указываем, т.к. оно каждый раз разное, ищем самый свежий файл:

    files_path = os.path.join(f'{path}/data/models/', '*')
    files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)

    path_model = files[0].replace('\\', '/')
    with open(path_model, 'rb') as file:
        model = dill.load(file)

    # Делаем предсказания для всех объектов из папки data/test и сохраняем в файл:
    i = 0
    res = pd.DataFrame(columns=['obj', 'prediction'])

    for filename in os.listdir(f'{path}/data/test'):
        df = pd.DataFrame(pd.read_json(f'{path}/data/test/{filename}', typ='series')).transpose()
        y = model.predict(df)
        res.loc[i] = [filename, y]
        i += 1
    res.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')



if __name__ == '__main__':
    predict()
