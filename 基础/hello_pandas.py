import pandas as pd
import numpy as np

city_names = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
city_populations = pd.Series([100,200,50])

citys = pd.DataFrame({"names" : city_names , "populations": city_populations})

# print(city_names)
# print(city_populations)
# print(citys)
#
# print(citys.head(1))
#
# print(city_names[0:1])

california_housing_dataframe = pd.read_csv("california_housing_train.csv", sep=",")
# print(california_housing_dataframe.describe())

# print(city_populations / 20)

applys = city_populations.apply(lambda val : val > 100)
# print(applys)

citys["applys"] = applys

# this is error
# citys["test"] = citys.apply(lambda val : val["names"].startswith("San") & val["populations"] > 150)

# 2列分别计算
citys["test"] = (citys["populations"] > 150) & (citys["names"].apply(lambda val : val.startswith("San")))

print(citys.index)
print(citys)
# 随机排序 抽样，递归 梯度下降
citys = citys.reindex(np.random.permutation(citys.index))
print(citys.index)
print(citys)
