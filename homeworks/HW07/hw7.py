import nbformat as nbf

nb = nbf.v4.new_notebook()

md1 = "# 1. Загрузка данных и первичный анализ (для каждого датасета)"
nb.cells.append(nbf.v4.new_markdown_cell(md1))

code1 = """import numpy as np\nimport pandas as pd
df1 = pd.read_csv('./data/S07-hw-dataset-01.csv')
df2 = pd.read_csv('./data/S07-hw-dataset-02.csv')
df3 = pd.read_csv('./data/S07-hw-dataset-03.csv')

print("--------------CSV-1----------------")
display(df1.head())
display(df1.info())
display(df1.describe())
print("-----Пропуски------")
display(df1.isna().value_counts())
print("-----Типы признаков------")
display(df1.dtypes)


print("--------------CSV-2----------------")
display(df2.head())
display(df2.info())
display(df2.describe())
print("-----Пропуски------")
display(df2.isna().value_counts())
print("-----Типы признаков------")
display(df2.dtypes)

print("--------------CSV-3----------------")
display(df3.head())
display(df3.info())
display(df3.describe())
print("-----Пропуски------")
display(df2.isna().value_counts())
print("-----Типы признаков------")
display(df3.dtypes)


X1 = df1.iloc[:,1:]
X2 = df2.iloc[:,1:]
X3 = df3.iloc[:,1:]
sample_id1 = df1.iloc[:,0]
sample_id2 = df2.iloc[:,0]
sample_id3 = df3.iloc[:,0]

"""

nb.cells.append(nbf.v4.new_code_cell(code1))

with open('HW07.ipynb', 'w') as f:
    nbf.write(nb, f)
