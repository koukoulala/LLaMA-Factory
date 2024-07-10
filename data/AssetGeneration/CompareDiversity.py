import json
import re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
import string
import argparse

# 读取pred.jsonl文件内容
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

# 去除标点符号并转换为小写
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.lower()  # 转换为小写
    return text

# 计算Lexical Diversity
def lexical_diversity(texts):
    words = [word for text in texts for word in text.split()]
    word_count = Counter(words)
    return len(word_count) / len(words) if words else 0

# 计算mBLEU Diversity
def internal_mbleu_diversity(texts):
    smoothie = SmoothingFunction().method4
    scores = []
    for i in range(len(texts)):
        ref = texts[i].split()
        for j in range(len(texts)):
            if i != j:
                pred = texts[j].split()
                score = sentence_bleu([ref], pred, smoothing_function=smoothie)
                scores.append(score)
    return sum(scores) / len(scores) if scores else 0
# 主函数
def main(file_path):
    data = read_jsonl(file_path)
    label_lexical_diversities = []
    predict_lexical_diversities = []
    label_mbleu_diversities = []
    predict_mbleu_diversities = []
    
    for entry in data:
        # 切分并处理label和predict
        labels = [preprocess_text(ad.replace("Ad:", "").strip()) for ad in entry['label'].split("\n") if ad]
        predicts = [preprocess_text(ad.replace("Ad:", "").strip()) for ad in entry['predict'].split("\n") if ad]
        
        # 计算Lexical Diversity
        label_lexical_diversity = lexical_diversity(labels)
        predict_lexical_diversity = lexical_diversity(predicts)
        
        # 计算mBLEU Diversity
        label_mbleu = internal_mbleu_diversity(labels)
        predict_mbleu = internal_mbleu_diversity(predicts)
        
        label_lexical_diversities.append(label_lexical_diversity)
        predict_lexical_diversities.append(predict_lexical_diversity)
        label_mbleu_diversities.append(label_mbleu)
        predict_mbleu_diversities.append(predict_mbleu)

    # 计算平均值
    avg_label_lexical_diversity = sum(label_lexical_diversities) / len(label_lexical_diversities) if label_lexical_diversities else 0
    avg_predict_lexical_diversity = sum(predict_lexical_diversities) / len(predict_lexical_diversities) if predict_lexical_diversities else 0
    avg_label_mbleu_diversity = sum(label_mbleu_diversities) / len(label_mbleu_diversities) if label_mbleu_diversities else 0
    avg_predict_mbleu_diversity = sum(predict_mbleu_diversities) / len(predict_mbleu_diversities) if predict_mbleu_diversities else 0
    
    return {
        'avg_label_lexical_diversity': avg_label_lexical_diversity,
        'avg_predict_lexical_diversity': avg_predict_lexical_diversity,
        'avg_label_mbleu_diversity': avg_label_mbleu_diversity,
        'avg_predict_mbleu_diversity': avg_predict_mbleu_diversity
    }
    

# 运行主函数并输出结果
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GenerateTrainTestDataForLLM')
    parser.add_argument('--input', help='input file', default="./saves/mistral/orpo_qlora_continue_2/predict_add_diversity/generated_predictions.jsonl")

    args = parser.parse_args()

    file_path = args.input
    results = main(file_path)
    print("Average Label Lexical Diversity: {:.4f}".format(results['avg_label_lexical_diversity']))
    print("Average Predict Lexical Diversity: {:.4f}".format(results['avg_predict_lexical_diversity']))
    print("Average Label mBLEU Diversity: {:.4f}".format(results['avg_label_mbleu_diversity']))
    print("Average Predict mBLEU Diversity: {:.4f}".format(results['avg_predict_mbleu_diversity']))
