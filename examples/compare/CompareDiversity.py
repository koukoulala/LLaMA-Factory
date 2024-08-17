import json
import re
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.corpus import stopwords
import string
import argparse

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = text.lower()  # 转换为小写
    return text

def lexical_diversity(texts):
    words = [word for text in texts for word in text.split()]
    word_count = Counter(words)
    return len(word_count) / len(words) if words else 0

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

def main(file_path):
    data = read_jsonl(file_path)
    label_lexical_diversities = []
    predict_lexical_diversities = []
    label_mbleu_diversities = []
    predict_mbleu_diversities = []
    
    for entry in data:
        labels = [preprocess_text(ad.replace("Ad:", "").strip()) for ad in entry['label'].split("\n") if ad]
        predicts = [preprocess_text(ad.replace("Ad:", "").strip()) for ad in entry['predict'].split("\n") if ad]
        
        label_lexical_diversity = lexical_diversity(labels)
        predict_lexical_diversity = lexical_diversity(predicts)
        
        label_mbleu = internal_mbleu_diversity(labels)
        predict_mbleu = internal_mbleu_diversity(predicts)
        
        label_lexical_diversities.append(label_lexical_diversity)
        predict_lexical_diversities.append(predict_lexical_diversity)
        label_mbleu_diversities.append(label_mbleu)
        predict_mbleu_diversities.append(predict_mbleu)

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GenerateTrainTestDataForLLM')
    #parser.add_argument('--input', help='input file', default="./saves/qwen/fsdp_qlora_sft/predict_new/generated_predictions.jsonl")
    #parser.add_argument('--input', help='input file', default="./saves/mistral/fsdp_qlora_sft_new/predict_new/generated_predictions.jsonl")
    parser.add_argument('--input', help='input file', default="./saves/phi/fsdp_qlora_sft_new/predict_new/generated_predictions.jsonl")
    parser.add_argument('--output', help='output file', default="./saves/phi/fsdp_qlora_sft_new/predict_new/compare_diversity.json")

    args = parser.parse_args()

    results = main(args.input)
    print("Average Label Lexical Diversity: {:.4f}".format(results['avg_label_lexical_diversity']))
    print("Average Predict Lexical Diversity: {:.4f}".format(results['avg_predict_lexical_diversity']))
    print("Average Label mBLEU Diversity: {:.4f}".format(results['avg_label_mbleu_diversity']))
    print("Average Predict mBLEU Diversity: {:.4f}".format(results['avg_predict_mbleu_diversity']))

    # Save results to file
    with open(args.output, 'w') as file:
        json.dump(results, file, indent=4)

    print("Results saved to", args.output)
