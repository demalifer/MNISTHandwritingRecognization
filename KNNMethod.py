from torchvision.datasets import MNIST
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 数据集
train_data = MNIST(root='./data', train=True, download=True)
test_data = MNIST(root='./data', train=False, download=True)

# 把每个输入数据展平成一维向量，并获取 X 和 y
X_train, y_train = train_data.data.numpy().reshape(-1, 28 * 28), train_data.targets.numpy()
X_test, y_test = test_data.data.numpy().reshape(-1, 28 * 28), test_data.targets.numpy()

# 归一化
X_train, X_test = X_train / 255.0, X_test / 255.0

# KNN
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('KNN Accuracy: ', accuracy)
