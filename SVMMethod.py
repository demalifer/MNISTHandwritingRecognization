import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
def load_dataset():
    # 数据集
    train_dataset = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    # 转换成 numpy
    X_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
    X_test, y_test = test_dataset.data.numpy(), test_dataset.targets.numpy()
    return X_train, y_train, X_test, y_test

# 提取 hog 特征
def extract_hog_features(imgs):
    hog_features = []
    for img in imgs:
        hog_feature = hog(img, orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2), block_norm='L2-Hys')
        hog_features.append(hog_feature)
    return np.array(hog_features)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_dataset()
    X_train_hog, X_test_hog = extract_hog_features(X_train), extract_hog_features(X_test)
    model = SVC(kernel='rbf', C=10, gamma='scale')
    model.fit(X_train_hog, y_train)
    y_pred = model.predict(X_test_hog)
    accuracy = accuracy_score(y_test, y_pred)
    print('SVM Accuracy: ', accuracy)