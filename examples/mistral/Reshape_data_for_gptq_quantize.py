import json
import argparse

def reshape_data(args):
    
    test_data = args.test_data
    messages = []
    with open(test_data, 'r', encoding='utf8') as fr:
        test_data = json.load(fr)

    for RowId, data in enumerate(test_data):
        text = data["instruction"] + "\n" + data["input"]
        tmp_message = {"text": text}
        messages.append(tmp_message)
    print("count of messages: ", len(messages))
    print("messages examples: ", messages[:2])

    with open(args.reshaped_data, 'w', encoding='utf-8') as fw:
        json.dump(messages, fw, ensure_ascii=False, indent=4)

# write a main function with arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str, default="/data/xiaoyukou/LLM_Inference/data/small_test.json", help="The path of the test data.")
    parser.add_argument("--reshaped_data", type=str, default="/data/xiaoyukou/LLM_Inference/data/small_test_reshaped.json", help="The path of the test data.")
    args = parser.parse_args()

    reshape_data(args)
    