import nltk

# 添加新的数据路径
nltk.data.path.append('C:/Users/TimE/nltk_data')

# 打印所有 NLTK 查找的路径
print(nltk.data.path)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    print("资源已成功加载！")
except LookupError:
    print("无法找到该资源。")