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

# 运行主函数并输出结果
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input file', default="./saves/qwen/fsdp_qlora_sft/predict/generated_predictions.jsonl")
    parser.add_argument('--output', help='output file', default="./data/AssetGeneration/BackupTestData.jsonl")

    args = parser.parse_args()

    file_path = args.input
    results = main(file_path)
    
