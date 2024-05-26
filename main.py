import sys
import joblib
from nltk.stem.porter import PorterStemmer
import nltk
import re
from nltk.corpus import stopwords

# 指定下载路径为你的项目目录
project_directory = "D:/NJU/SEIII/nlp_fakeNews"
nltk.data.path.append(project_directory)

# 下载stopwords并保存到指定路径
# nltk.download('stopwords', download_dir=project_directory)


def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

if __name__ == '__main__':
    # 准备输入字符串
    input_string = sys.argv[1]

    # input_string = "hello world!I am huge tree!"
    port_stem = PorterStemmer()
    processed_input = stemming(input_string)

    vectorizer = joblib.load(r'D:\NJU\SEIII\nlp_fakeNews\tfidf_vectorizer.pkl')

    # 使用加载的向量化器将文本转换为数值特征
    input_vectorized = vectorizer.transform([processed_input])

    # 加载模型
    model = joblib.load(r'D:\NJU\SEIII\nlp_fakeNews\logistic_regression_model.pkl')

    # 使用加载的模型进行预测
    prediction = model.predict(input_vectorized)

    # 使用加载的模型进行预测
    probabilities = model.predict_proba(input_vectorized)

    print(probabilities[0][0])

# 如果在外网环境可以用谷歌翻译
# from googletrans import Translator
#
# # 设置翻译服务器地址
# translator = Translator(service_urls=['translate.google.com'])
# # 第一个参数是待翻译文本，dest是目标语言
# results = translator.translate('你好，世界！', dest='en')
# print(results.text)