import nltk

# 确保添加路径
nltk.data.path.append('C:/Users/TimE/nltk_data')

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

# 测试句子
sentence = "This is a test sentence."
tokens = word_tokenize(sentence)

# 使用 pos_tag 进行词性标注
print(pos_tag(tokens))