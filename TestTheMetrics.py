import tensorflow as tf
import os
import cv2
import numpy as np

# modelFile = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.h5'
modelFile = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/dogs.keras'

model = tf.keras.models.load_model(modelFile)

# print(model.summary())

inputShape = (311,311)
allLables = np.load("/csse/users/yzo17/PycharmProjects/dogBreedsClassification/temp/allDogsLabels.npy")
categories = np.unique(allLables)

import scipy.io

# 加载.mat文件
test_data = scipy.io.loadmat('/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/test_data.mat')
test_labels = []
# prepare image
dir = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/Images/'
# Define the list of substrings
from sklearn.preprocessing import LabelEncoder

# 假定 substrings 是所有可能的类别标签
label_encoder = LabelEncoder()
substrings = ['Afghan_hound', 'basset', 'Chihuahua', 'Maltese_dog', 'papillon', 'Pekinese', 'Pomeranian', 'Shih-Tzu', 'toy_terrier']
label_encoder.fit(substrings)  # 训练 LabelEncoder 以理解所有可能的类别

def prepareImage(img):
    imgPath = dir + img
    found_substrings = [sub for sub in substrings if sub in img]
    if found_substrings:
        label = found_substrings[0]
        if label == 'Pomeranian':
            labelText = 'pomeranian'
        elif label == 'Chihuahua':
            labelText = 'chihuahua'
        elif label == 'Maltese_dog':
            labelText = 'maltese'
        elif label == 'Pekinese':
            labelText = 'pekinese'
        elif label == 'Shih-Tzu':
            labelText = 'shitzu'
        elif label == 'papillon':
            labelText = 'papillon'
        elif label == 'toy_terrier':
            labelText = 'toy_terrier'
        elif label == 'Afghan_hound':
            labelText = 'afghan_hound'
        elif label == 'basset':
            labelText = 'basset'
        image = cv2.imread(imgPath)
        if image is None:
            raise ValueError("Can't find the images: " + imgPath)
        resized = cv2.resize(image, inputShape, interpolation=cv2.INTER_AREA)
        imgResult = np.expand_dims(resized, axis=0)
        imgResult = imgResult / 255.
        return imgResult, labelText
    return None, None

# 处理图像和标签
test_images_list = []
test_labels_list = []
test_image_paths = test_data['test_info']['file_list'][0][0][:, 0]
for imgPath in test_image_paths:
    [img] = imgPath
    image, label = prepareImage(img)
    if image is not None:
        test_images_list.append(image)
        test_labels_list.append(label)

if test_images_list:
    test_images = np.vstack(test_images_list)
    test_labels = np.array(test_labels_list)
    from sklearn.preprocessing import LabelEncoder

    print('test_labels', test_labels)
    le = LabelEncoder()
    integerLables = le.fit_transform(test_labels)
    print('integerLables', integerLables)

    numOfCategories = len(np.unique(integerLables))

    from tensorflow.keras.utils import to_categorical

    allLablesForTestModel = to_categorical(integerLables, num_classes=numOfCategories)

    # 如果数据格式正确且非空，则评估模型
    if test_images.size > 0 and len(test_images) == len(allLablesForTestModel):
        result = model.evaluate(test_images, allLablesForTestModel)
        print("Result:", result)
        from sklearn.metrics import precision_score, recall_score, classification_report
        from sklearn.metrics import confusion_matrix
        # 假设 allLablesForTestModel 是你的真实标签的独热编码
        # 获取模型的预测结果
        predictions = model.predict(test_images)

        # 转换预测结果为标签索引
        predicted_labels = np.argmax(predictions, axis=1)

        # 转换真实标签为标签索引（如果尚未这样做）
        true_labels = np.argmax(allLablesForTestModel, axis=1)

        # 计算精确率和召回率
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')

        # 输出精确率和召回率
        print("Precision:", precision)
        print("Recall:", recall)

        # 可选：输出详细的分类报告
        print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

        cm = confusion_matrix(true_labels, predicted_labels)
        print("confusion_matrix", cm)
        classes = ['Afghan_hound', 'Basset', 'Chihuahua', 'Maltese', 'Papillon', 'Pekingese', 'Pomeranian', 'Shih-Tzu', 'Toy Terrier']
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    else:
        print("数据格式不正确或缺失数据。")
else:
    print("没有有效的测试图像处理。")
# testImagePath = '/csse/users/yzo17/PycharmProjects/dogBreedsClassification/test/test.jpg'

# load image
# img = cv2.imread(testImagePath)
# imageForModel = prepareImage(img)

# prediction
# resultArray = model.predict(imageForModel)
# answers = np.argmax(resultArray,axis=1)
# print(answers)

# text = categories[answers[0]]
# print(categories)
# print(text)

# font = cv2.FONT_HERSHEY_COMPLEX
# cv2.putText(img,text,(0,20),font,1,(209,19,77),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
