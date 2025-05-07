import json
import os
import argparse
import re
import random
from collections import Counter
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import difflib
from sklearn.metrics import jaccard_score
import numpy as np
from itertools import combinations

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

def construct_message(user_prompt_template, detail_info, Asset_list, data_idx, AssetCnt, AssetType, FullLanguage):
    instruction = user_prompt_template.format(AssetCnt, AssetType, FullLanguage)
    output = ""
    for asset in Asset_list:
        output += "Ad:" + asset + "\n"
    message = {"instruction": instruction, "input": detail_info, "output": output}

    return message

def get_insight_indices(string_list):
    count = Counter(string_list)
    candidates = [string for string, freq in count.items() if freq >= 1 and string != "" and len(string) < 70]

    if not candidates:
        return None, []

    random_string = random.choice(candidates)
    indices = [i for i, s in enumerate(string_list) if s == random_string]

    return random_string, indices

def get_asset_that_contain_keywords(Asset_list, NormKeywords_list):
    lower_Assets = [Asset.lower().strip(' +.,!=-#，。！&@￥$()') for Asset in Asset_list]

    ContainKW_Asset_list = []
    #Contained_KW_list = [kw for kw in NormKeywords_list if kw in assets]
    for idx in range(len(lower_Assets)):
        lower_asset = lower_Assets[idx]
        Asset = Asset_list[idx]
        if any(keyword in lower_asset for keyword in NormKeywords_list):
            ContainKW_Asset_list.append(Asset)
            break
    
    return ContainKW_Asset_list

def jaccard_similarity(asset1, asset2):
    set1, set2 = set(asset1), set(asset2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def select_most_dissimilar_assets(asset_list, num_assets):
    if num_assets >= len(asset_list):
        return asset_list

    # Calculate pairwise Jaccard similarity
    pairwise_similarity = np.zeros((len(asset_list), len(asset_list)))
    for i, j in combinations(range(len(asset_list)), 2):
        pairwise_similarity[i, j] = jaccard_similarity(asset_list[i], asset_list[j])
        pairwise_similarity[j, i] = pairwise_similarity[i, j]

    # Calculate average similarity for each asset
    average_similarity = np.mean(pairwise_similarity, axis=1)

    # Select assets with the lowest average similarity
    selected_indices = np.argsort(average_similarity)[:num_assets]
    selected_assets = [asset_list[i] for i in selected_indices]

    return selected_assets


def main(args):
    inputfile = args.input

    input_row = 0
    data_idx = 0
    data_withDKI, data_withInsight, data_withTopKW = 0, 0, 0
    full_data_list = []

    user_prompt_template = "Please generate {} Ad {} in {} language, based on the following information:\n"
    with open(inputfile, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            FinalUrl, Domain, CategoryName, DescriptionOfAdvertiser, FullLanguage, AssetType, JointAsset, JointIsDKI, JointInsight, JointNormKeywords, sd_doc = line.split('\t')
            Asset_list = JointAsset.split('[SEP]')
            Asset_list = [asset.strip() for asset in Asset_list]
            IsDKI_list = JointIsDKI.split('[SEP]')
            IsDKI_list = [isDKI.strip() for isDKI in IsDKI_list]
            Insight_list = JointInsight.split('[SEP]')
            Insight_list = [insight.strip() for insight in Insight_list]
            NormKeywords_list = JointNormKeywords.split('#')
            NormKeywords_list = [normKeyword.lower().strip(' +.,!=-#，。！&@￥$()') for normKeyword in NormKeywords_list]
            NormKeywords_list = [kw for kw in NormKeywords_list if len(kw) > 1]

            detail_info = "FinalUrl: " + FinalUrl + " \n"
            if len(Domain) > 2 and random.random() <= 0.9:
                detail_info += "Domain: " + Domain + " \n"
            reshaped_category_name = reshape_category(CategoryName)
            if len(reshaped_category_name) > 2 and random.random() <= 0.9:
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

                message = construct_message(user_prompt_template, detail_DKI_info, DKI_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                #print("IsDKI message: ", message)
                full_data_list.append(message)
                data_idx += 1
                data_withDKI += 1
                # refine non-DKI assets and insights
                Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "0"]
                Insight_list = [insight for insight, isDKI in zip(Insight_list, IsDKI_list) if isDKI == "0"]

            if any(Insight_list) and random.random() < 0.5:
                Insight, indices = get_insight_indices(Insight_list)
                if Insight:
                    #print("Doing generation with insight for the non-DKI assets.")
                    Insight_Asset_list = [Asset_list[i] for i in indices]
                    Insight_Asset_list = list(set(Insight_Asset_list))
                    AssetCnt = len(Insight_Asset_list)
                    detail_insight_info = detail_info + "Insight: " + Insight + " \n"

                    message = construct_message(user_prompt_template, detail_insight_info, Insight_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                    #print("Insight message: ", message)
                    full_data_list.append(message)
                    data_idx += 1
                    data_withInsight += 1

            if any(NormKeywords_list):
                # construct prompt that assets contain keywords
                len_kw = len(NormKeywords_list)
                if len_kw > 10:
                    part_kws = random.sample(NormKeywords_list, int(len_kw/2))
                    ContainKW_Asset_list = get_asset_that_contain_keywords(Asset_list, part_kws)
                    if len(ContainKW_Asset_list) > 1 or (len(ContainKW_Asset_list) == 1 and random.random() < 0.6):
                        ContainKW_Asset_list = list(set(ContainKW_Asset_list))
                        #print("\npart_kws: ", part_kws)
                        #print("ContainKW_Asset_list: ", ContainKW_Asset_list)
                        AssetCnt = len(ContainKW_Asset_list)
                        detail_KW_info = detail_info + "Keywords: " + "#".join(part_kws) + " \n"
                        detail_KW_info = detail_KW_info + "Insight: " + "Ensure relevance by including reasonable keywords in each " + AssetType.lower() + "." + " \n"
                        message = construct_message(user_prompt_template, detail_KW_info, ContainKW_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                        #print("ContainKW message: ", message)
                        full_data_list.append(message)
                        data_idx += 1
                        data_withTopKW += 1

            if any(Asset_list):
                # Doing generation without insight for the non-DKI assets
                Asset_list = list(set(Asset_list))
                AssetCnt = random.randint(1, len(Asset_list))
                #rand_Asset_list = random.sample(Asset_list, AssetCnt)
                rand_Asset_list = select_most_dissimilar_assets(Asset_list, AssetCnt)
                if AssetCnt > 1 and random.random() < 0.5:
                    detail_insight_info = detail_info + "Insight: " + "Ensure diversity by highlighting various selling pionts in each " + AssetType.lower() + "." + " \n"
                else:
                    detail_insight_info = detail_info
                message = construct_message(user_prompt_template, detail_insight_info, rand_Asset_list, data_idx, AssetCnt, AssetType, FullLanguage)
                #print("No Insight message: ", message)
                full_data_list.append(message)
                data_idx += 1

            if input_row % 10000 == 0:
                print("Processing row: ", input_row)
                print("Total data rows: ", data_idx)
                print("Total data with DKI: ", data_withDKI)
                print("Total data with Insight: ", data_withInsight)
                print("Total data with TopKW: ", data_withTopKW)

            input_row += 1


    print("\nTotal input rows: ", input_row)
    print("Total data rows for model: ", data_idx)
    print("Total data with DKI: ", data_withDKI)
    print("Total data with Insight: ", data_withInsight)
    print("Total data with TopKW: ", data_withTopKW)

    random.shuffle(full_data_list)
    train_size = int(len(full_data_list) * 0.95)
    train_data = full_data_list[:train_size]
    test_data = full_data_list[train_size:]
    print("Train data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    # output full data, train data, test data
    with open(args.FullData, 'w', encoding='utf-8') as fw_full:
        json.dump(full_data_list, fw_full, ensure_ascii=False, indent=4)

    with open(args.train, 'w', encoding='utf-8') as fw_train:
        json.dump(train_data, fw_train, ensure_ascii=False, indent=4)

    with open(args.test, 'w', encoding='utf-8') as fw_test:
        json.dump(test_data, fw_test, ensure_ascii=False, indent=4)

    samll_test_data = test_data[:200]
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
    parser.add_argument('-i', '--input', help='input file', default="./CombinedAssets_2024-07-28.tsv")
    #parser.add_argument('-i', '--input', help='input file', default="../data/AssetGeneration/test.tsv")
    parser.add_argument('-fu', '--FullData', help='json file', default="./FullData.json")
    parser.add_argument('-tr', '--train', help='json file', default="./train.json")
    parser.add_argument('-te', '--test', help='json file', default="./test.json")
    parser.add_argument('-small_te', '--small_test', help='json file', default="./small_test.json")
    args = parser.parse_args()
    main(args)
    '''
    out_prompt_file = "./inference_prompt_AddDiversity.tsv"
    out_response_file = "./inference_groundtruth_AddDiversity.tsv"
    ConvertJsonToInferenceData(args.small_test, out_prompt_file, out_response_file)
    '''

