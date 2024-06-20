import json
import os
import argparse
import re
import random
from collections import Counter
import pandas as pd
from googletrans import Translator
from nltk.corpus import wordnet
import nltk


def reshape_category(CategoryName):
    if CategoryName == "":
        return ""
    category_list = CategoryName.split('--')
    reshaped_category_name = ""
    for i, category in enumerate(category_list):
        category = category.strip()
        if category != "" and category.lower() != "others" and category.lower() != "other" and category.lower() != "unspecified":
            category = ' '.join([word for word in category.split() if word.lower() != "other"])
            if i == 0:
                reshaped_category_name += category
            else:
                reshaped_category_name += " -- " + category
    return reshaped_category_name

def construct_message_orpo(user_prompt_template, detail_info, Asset_list, AssetCnt, AssetType, FullLanguage, rejected_Asset_list):
    instruction = user_prompt_template.format(AssetCnt, AssetType, FullLanguage)
    output = ""
    for asset in Asset_list:
        output += "Ad:" + asset + "\n"
    rejected_output = ""
    for asset in rejected_Asset_list:
        rejected_output += "Ad:" + asset + "\n"
    message = {"instruction": instruction, "input": detail_info, "output": output, "rejected": rejected_output}

    return message

def get_insight_indices(string_list):
    count = Counter(string_list)
    candidates = [string for string, freq in count.items() if freq >= 1 and string != "" and len(string) < 70]

    if not candidates:
        return None

    return candidates

nltk.download('wordnet')
nltk.download('punkt')
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def replace_with_synonyms(sentence):
    words = nltk.word_tokenize(sentence)
    new_sentence = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms and random.random() > 0.5:  # 50%的几率替换
            new_sentence.append(random.choice(synonyms))
        else:
            new_sentence.append(word)
    return ' '.join(new_sentence)

def delete_random_words(sentence, deletion_prob=0.2):
    words = nltk.word_tokenize(sentence)
    new_sentence = [word for word in words if random.random() > deletion_prob]
    return ' '.join(new_sentence)

def add_random_words(sentence, addition_prob=0.2):
    words = nltk.word_tokenize(sentence)
    new_sentence = []
    for word in words:
        if random.random() < addition_prob:
            new_sentence.append(random.choice(words))
        new_sentence.append(word)
    return ' '.join(new_sentence)

def get_rejected_assets(Chosen_Asset_list, AssetCnt, FullLanguage, Asset_list, all_data, idx, rejected_reasons):
    # Get rejected assets based on diversity, language, asset count, wrong assets
    p = random.random()
    rejected_Asset_list = []

    asset_list_len = len(Asset_list)

    if AssetCnt > 1 and p <= 0.6:
        # diversity
        sample_cnt = random.randint(1, AssetCnt - 1)
        rejected_Asset_list = random.sample(Chosen_Asset_list, sample_cnt)
        rejected_Asset_list = (rejected_Asset_list * (AssetCnt // sample_cnt + 1))[:AssetCnt]
        if random.random() < 0.2:
            updated_rejected_Asset_list = []
            for asset in rejected_Asset_list:
                asset = replace_with_synonyms(asset)
                asset = delete_random_words(asset)
                asset = add_random_words(asset)
                updated_rejected_Asset_list.append(asset)
            rejected_Asset_list = updated_rejected_Asset_list
        random.shuffle(rejected_Asset_list)
        rejected_reasons["Diversity"] += 1
        #print("Chosen_Asset_list: ", Chosen_Asset_list)
        #print("Diversity: ", rejected_Asset_list)
    elif p <= 0.8:
        # wrong assets
        CategoryName = all_data.iloc[idx]["CategoryName"]
        same_category_idxs = all_data[all_data["CategoryName"] == CategoryName].index
        same_category_idxs = [i for i in same_category_idxs if i != idx]
        if same_category_idxs:
            random_idx = random.choice(same_category_idxs)
        else:
            # randomly choose a line in all_data
            random_idx = random.randint(0, len(all_data) - 1)
        rejected_Asset_list = all_data.iloc[random_idx]["JointAsset"].split('[SEP]')
        rejected_Asset_list = [asset.strip() for asset in rejected_Asset_list]
        rejected_Asset_list = random.sample(rejected_Asset_list, min(AssetCnt, len(rejected_Asset_list)))
        rejected_reasons["Wrong Assets"] += 1
        #print("Chosen_Asset_list: ", Chosen_Asset_list)
        #print("Wrong Assets: ", rejected_Asset_list)
    elif p <= 0.9:
        # asset count
        choices = [num for num in range(1, asset_list_len + 5) if num != AssetCnt]
        rejected_AssetCount = random.choice(choices)
        rejected_Asset_list = random.sample(Asset_list, min(rejected_AssetCount, asset_list_len))
        rejected_Asset_list = (rejected_Asset_list * (AssetCnt // len(rejected_Asset_list) + 1))[:AssetCnt]
        random.shuffle(rejected_Asset_list)
        rejected_reasons["Asset Count"] += 1
    else:
        # different language
        translator = Translator()
        dest_langs = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh-cn", "zh-tw"]
        corresponding_full_langs = ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Japanese", "Korean", "Simplified Chinese", "Chinese"]
        diff_dest_langs = [lang for lang, full_lang in zip(dest_langs, corresponding_full_langs) if full_lang != FullLanguage]
        diff_dest_lang = random.choice(diff_dest_langs)
        rejected_Asset_list = []
        try:
            rejected_Asset_list = [translator.translate(item, dest=diff_dest_lang).text for item in Chosen_Asset_list]
        except:
            rejected_Asset_list = [sent[:-2] for sent in Chosen_Asset_list]
        rejected_reasons["Different Language"] += 1
        #print("Chosen_Asset_list: ", Chosen_Asset_list)
        #print("Different Language: ", rejected_Asset_list)

    return rejected_Asset_list, rejected_reasons

def main(args):
    inputfile = args.input

    input_row = 0
    data_idx = 0
    data_withDKI = 0
    data_withInsight = 0
    full_data_list = []
    rejected_reasons = {
        "DKI Insight": 0,
        "Other Insight": 0,
        "Diversity": 0,
        "Different Language": 0,
        "Wrong Assets": 0,
        "Asset Count": 0
    }

    columns = ["FinalUrl", "Domain", "CategoryName", "DescriptionOfAdvertiser", "FullLanguage", "AssetType", "JointAsset", "JointIsDKI", "JointInsight", "sd_doc"]
    dtype_dict = {col: str for col in columns}
    all_data = pd.read_csv(inputfile, sep='\t', header=None, names=columns, dtype=dtype_dict).fillna('')
    # shuffle the data
    all_data = all_data.sample(frac=1).reset_index(drop=True)

    user_prompt_template = "Please generate {} Ad {} in {} language, based on the following information:\n"
    for idx, row in all_data.iterrows():
        if idx <= 50000:
            FinalUrl, Domain, CategoryName, DescriptionOfAdvertiser, FullLanguage, AssetType, JointAsset, JointIsDKI, JointInsight, sd_doc = row
            Asset_list = JointAsset.split('[SEP]')
            Asset_list = [asset.strip() for asset in Asset_list]
            IsDKI_list = JointIsDKI.split('[SEP]')
            IsDKI_list = [isDKI.strip() for isDKI in IsDKI_list]
            Insight_list = JointInsight.split('[SEP]')
            Insight_list = [insight.strip() for insight in Insight_list]

            detail_info = "FinalUrl: " + FinalUrl + " \n"
            if len(Domain) > 2:
                detail_info += "Domain: " + Domain + " \n"
            reshaped_category_name = reshape_category(CategoryName)
            if len(reshaped_category_name) > 2:
                detail_info += "Category: " + reshaped_category_name + " \n"
            if len(DescriptionOfAdvertiser) > 2:
                detail_info += "DescriptionOfAdvertiser: " + DescriptionOfAdvertiser + " \n"
            if len(sd_doc) > 2:
                delimiters = ";\n"
                regexPattern = '|'.join(map(re.escape, delimiters))
                text_list = re.split(regexPattern, sd_doc)
                cleaned_text_list = [sent for sent in text_list if len(sent) > 3]
                SD_text = "; ".join(cleaned_text_list)
                sd_doc = SD_text[:400]
                detail_info += "LandingPage: " + sd_doc + " \n"
            if AssetType == "Headline":
                detail_info += "CharacterLimit: between 10 to 30 characters. \n"
            if AssetType == "Description":
                min_length = min([len(asset) for asset in Asset_list])
                detail_info += "CharacterLimit: between " + str(min_length) + " to 90 characters. \n"

            if "1" in IsDKI_list:
                DKI_Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "1"]
                DKI_Asset_list = list(set(DKI_Asset_list))
                AssetCnt = len(DKI_Asset_list)
                Insight = "Incorporate dynamic keyword insertion to make your ad more relevant to query."
                detail_DKI_info = detail_info + "Insight: " + Insight + " \n"

                # Get rejected assets
                if "0" in IsDKI_list and random.random() <= 0.6:
                    rejected_Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "0"]
                    rejected_Asset_list = (rejected_Asset_list * (AssetCnt // len(rejected_Asset_list) + 1))[:AssetCnt]
                    random.shuffle(rejected_Asset_list)
                    rejected_reasons["DKI Insight"] += 1
                    #print("Chosen_Asset_list: ", DKI_Asset_list)
                    #print("DKI Insight: ", rejected_Asset_list)
                else:
                    # Do other reject methods
                    rejected_Asset_list, rejected_reasons = get_rejected_assets(DKI_Asset_list, AssetCnt, FullLanguage, Asset_list, all_data, idx, rejected_reasons)

                message = construct_message_orpo(user_prompt_template, detail_DKI_info, DKI_Asset_list, AssetCnt, AssetType, FullLanguage, rejected_Asset_list)
                #print("IsDKI message: ", message)
                full_data_list.append(message)
                data_idx += 1
                data_withDKI += 1
                # refine non-DKI assets and insights
                Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "0"]
                Insight_list = [insight for insight, isDKI in zip(Insight_list, IsDKI_list) if isDKI == "0"]

            if any(Insight_list) and random.random() < 0.5:
                candidates = get_insight_indices(Insight_list)
                if candidates:
                    #print("Doing generation with insight for the non-DKI assets.")
                    Insight = random.choice(candidates)
                    Insight_Asset_list = [asset for asset, insight in zip(Asset_list, Insight_list) if insight == Insight]
                    Insight_Asset_list = list(set(Insight_Asset_list))
                    AssetCnt = len(Insight_Asset_list)
                    detail_insight_info = detail_info + "Insight: " + Insight + " \n"

                    if random.random() <= 0.4 and len(candidates) > 1:
                        rejected_Asset_list = [asset for asset, insight in zip(Asset_list, Insight_list) if insight != Insight and asset not in Insight_Asset_list]
                        if rejected_Asset_list:
                            rejected_Asset_list = (rejected_Asset_list * (AssetCnt // len(rejected_Asset_list) + 1))[:AssetCnt]
                            random.shuffle(rejected_Asset_list)
                        else:
                            rejected_Asset_list = []
                        rejected_reasons["Other Insight"] += 1
                        #print("Chosen_Asset_list: ", Insight_Asset_list)
                        #print("Other Insight: ", rejected_Asset_list)
                    else:
                        rejected_Asset_list, rejected_reasons = get_rejected_assets(Insight_Asset_list, AssetCnt, FullLanguage, Asset_list, all_data, idx, rejected_reasons)
               
                    message = construct_message_orpo(user_prompt_template, detail_insight_info, Insight_Asset_list, AssetCnt, AssetType, FullLanguage, rejected_Asset_list)
                    #print("Insight message: ", message)
                    full_data_list.append(message)
                    data_idx += 1
                    data_withInsight += 1

            if any(Asset_list):
                # Doing generation without insight for the non-DKI assets
                Asset_list = list(set(Asset_list))
                AssetCnt = random.randint(1, len(Asset_list))
                rand_Asset_list = random.sample(Asset_list, AssetCnt)
                if AssetCnt > 1 and random.random() <= 0.5:
                    detail_insight_info = detail_info + "Insight: " + "Ensure diversity by highlighting various selling pionts in each " + AssetType.lower() + "." + " \n"
                else:
                    detail_insight_info = detail_info

                rejected_Asset_list, rejected_reasons = get_rejected_assets(rand_Asset_list, AssetCnt, FullLanguage, Asset_list, all_data, idx, rejected_reasons)
                message = construct_message_orpo(user_prompt_template, detail_insight_info, rand_Asset_list, AssetCnt, AssetType, FullLanguage, rejected_Asset_list)
                #print("No Insight message: ", message)
                full_data_list.append(message)
                data_idx += 1

            if input_row % 10000 == 0:
                print("\nProcessing row: ", input_row)
                print("Total data rows: ", data_idx)
                print("Total data with DKI: ", data_withDKI)
                print("Total data with Insight: ", data_withInsight)
                print("Rejected reasons: ", rejected_reasons)

            input_row += 1


    print("\nTotal input rows: ", input_row)
    print("Total data rows for model: ", data_idx)
    print("Total data with DKI: ", data_withDKI)
    print("Total data with Insight: ", data_withInsight)
    print("Rejected reasons: ", rejected_reasons)

    random.shuffle(full_data_list)
    train_size = int(len(full_data_list) * 0.95)
    train_data = full_data_list[:train_size]
    test_data = full_data_list[train_size:]
    print("\nTrain data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    # output full data, train data, test data
    with open(args.FullData, 'w', encoding='utf-8') as fw_full:
        json.dump(full_data_list, fw_full, ensure_ascii=False, indent=4)

    with open(args.train, 'w', encoding='utf-8') as fw_train:
        json.dump(train_data, fw_train, ensure_ascii=False, indent=4)

    with open(args.test, 'w', encoding='utf-8') as fw_test:
        json.dump(test_data, fw_test, ensure_ascii=False, indent=4)

    samll_train_data = train_data[:2000]
    samll_test_data = test_data[:200]
    with open(args.small_train, 'w', encoding='utf-8') as fw_small_train:
        json.dump(samll_train_data, fw_small_train, ensure_ascii=False, indent=4)
    
    with open(args.small_test, 'w', encoding='utf-8') as fw_small_test:
        json.dump(samll_test_data, fw_small_test, ensure_ascii=False, indent=4)

    print("Data generation is done.")

def ConvertJsonToInferenceData(input_file, out_prompt_file, out_response_file):
    with open(input_file, 'r', encoding='utf-8') as fr_test:
        test_data = json.load(fr_test)

    fw_prompt = open(out_prompt_file, 'w', encoding='utf-8')
    fw_response = open(out_response_file, 'w', encoding='utf-8')

    for RowId, data in enumerate(test_data):
        prompt = data["instruction"] + "\n" + data["input"]
        fw_prompt.write(json.dumps({"RowId": RowId, "prompt": prompt}, ensure_ascii=False) + "\n")
        response = data["output"]
        fw_response.write(json.dumps({"RowId": RowId, "response": response}, ensure_ascii=False) + "\n")

    fw_prompt.close()
    fw_response.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GenerateTrainTestDataForLLM')
    parser.add_argument('-i', '--input', help='input file', default="./OriginalCombinedAssets.tsv")
    #parser.add_argument('-i', '--input', help='input file', default="../data/AssetGeneration/test.tsv")
    parser.add_argument('-fu', '--FullData', help='json file', default="./FullData_orpo.json")
    parser.add_argument('-tr', '--train', help='json file', default="./train_orpo.json")
    parser.add_argument('-te', '--test', help='json file', default="./test_orpo.json")
    parser.add_argument('-small_tr', '--small_train', help='json file', default="./small_train_orpo.json")
    parser.add_argument('-small_te', '--small_test', help='json file', default="./small_test_orpo.json")
    args = parser.parse_args()
    main(args)

    out_prompt_file = "./inference_prompt_orpo.tsv"
    out_response_file = "./inference_groundtruth_orpo.tsv"
    ConvertJsonToInferenceData(args.small_test, out_prompt_file, out_response_file)

