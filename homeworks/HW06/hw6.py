import nbformat as nbf

# New notebook for writing everythint in notebook. I am too lazy to use VSCODE
# So writing scripts to save it as ipynb file. That is too awesome if u ask me. And Yes I AM A VIM USER!!!! LESSGOOO

nb = nbf.v4.new_notebook()

# """""""" 1  """"""""
# Markdown
nb.cells.append(nbf.v4.new_markdown_cell(
    "# 1. Загрузка данных и первичный анализ"
))

# Code
code1 = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('./S06-hw-dataset-01.csv')
x = df.iloc[:,:-1]
y = df.iloc[:,-1] # Target

display(df.head())
display(df.info())
display(df.describe())

print("Нулевые значения")
display(df.isna())

print("Типы (dtypes) значений столбцов/колонок")
print(df.dtypes)
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# """""""" 2 """"""""
# Markdown
nb.cells.append(nbf.v4.new_markdown_cell(
    "# 2. Train/Test-сплит и воспроизводимость"
))

# Code
code2 = """x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.25,
    random_state = 42, stratify = y
)
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Markdown
nb.cells.append(nbf.v4.new_markdown_cell(
    """ ## Фиксированные `random_state` очень важен, в обратном случае при каждом перезапуске скрипта/пайплайна случайным образом будет разделен на train/test подвыборки датасета. Seed же указывает, что разделение происходит по заданному значению.

## Стратификация данных же позволяет сохранить пропорцию классов в соотношении с test_size"""
))


# """"""" 3 """""""
# """" 3.1  """"
nb.cells.append(nbf.v4.new_markdown_cell(
    "# 3. Baseline’ы\n## DummyClassifier"))

code3_1 = """ # Классификатор
dummCls = DummyClassifier(strategy="most_frequent")
## Обучение
dummCls.fit(x_train,y_train)
## Предсказывание
dummCls.predict(x_test)
## Метрики
dummCls.score(x_test,y_test)

"""
nb.cells.append(nbf.v4.new_code_cell(code3_1))

# """" 3.2  """"
nb.cells.append(nbf.v4.new_markdown_cell("## LogisticRegression"))

code3_2 = """ # Пайплайн
pipe = Pipeline(
    [("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipe.fit(x_train,y_train)
pipe.predict(x_test)
pipe.score(x_test,y_test)
"""
nb.cells.append(nbf.v4.new_code_cell(code3_2))

nb.cells.append(nbf.v4.new_markdown_cell(
    "Здесь должна быть краткая интерпретация метрик и моделей"))

# Save it as ipynb
with open('HW06.ipynb', 'w') as f:
    nbf.write(nb, f)
