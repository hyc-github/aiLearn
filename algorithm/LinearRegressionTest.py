from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 创建线性回归模型
model = LinearRegression()

# 准备数据
X = np.array([35, 45, 40, 60, 65]).reshape(-1, 1)
y = np.array([30, 40, 35, 60, 65])

# 训练模型
model.fit(X, y)

# 预测
new_X = np.array([50]).reshape(-1, 1)
predicted_price = model.predict(new_X)

# 输出预测结果
print("预测的房价为：", predicted_price[0], "万元")

# 绘制图像
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('面积（平方米）')
plt.ylabel('价格（万元）')
plt.title('线性回归预测房价')
plt.show()