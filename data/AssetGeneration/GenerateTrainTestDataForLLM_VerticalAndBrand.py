import json
import os
import argparse
import re
import random
from collections import Counter
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

def construct_message(user_prompt_template, detail_info, Asset_list, AssetCnt, AssetType):
    instruction = user_prompt_template.format(AssetCnt, AssetType)
    output = ""
    for asset in Asset_list:
        output += "<Ad>" + asset + "</Ad>\n"
    message = {"instruction": instruction, "input": detail_info, "output": output}

    return message

def construct_message_with_length(user_prompt_template, detail_info, Asset_list, AssetCnt, AssetType):
    min_length = max(0, min([len(asset) for asset in Asset_list]) - 5)
    max_length = max([len(asset) for asset in Asset_list]) + 5
    if max_length <= min_length:
        max_length = min_length + 5
    detail_info += "CharacterLimit: between " + str(min_length) + " to " + str(max_length) + " characters. \n"

    instruction = user_prompt_template.format(AssetCnt, AssetType)
    output = ""
    for asset in Asset_list:
        output += "<Ad>" + asset + "</Ad>\n"
    message = {"instruction": instruction, "input": detail_info, "output": output}

    return message

def get_asset_that_contain_keywords(Asset_list, NormKeywords_list):
    lower_Assets = [Asset.lower().strip(' +.,!=-#，。！&@￥$()') for Asset in Asset_list]

    ContainKW_Asset_list = []
    #Contained_KW_list = [kw for kw in NormKeywords_list if kw in assets]
    for idx, lower_asset in enumerate(lower_Assets):
        if any(kw in lower_asset for kw in NormKeywords_list):
            ContainKW_Asset_list.append(Asset_list[idx])

    ContainKW_Asset_list = list(set(ContainKW_Asset_list))
    
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
    data_withDKI, data_LowQuality, data_withTopKW, data_copilot, data_general = 0, 0, 0, 0, 0
    scenario_cnt = {
        "AssetGeneration": 0,
        "AssetGenerationBasedOnQuery": 0,
        "AssetGenerationBasedOnTheme": 0,
        "AssetGenerationBasedOnThemeAndUser": 0,
        "ThemeGeneration": 0,
        "ThemeGenerationBasedOnUser": 0,
        "ChangeTone": 0,
        "FindSimilarAssets": 0,
        "RewriteAsset": 0
    }
    scenario_probs = {
        "AssetGeneration": 0.4,
        "AssetGenerationBasedOnTheme": 0.65,
        "AssetGenerationBasedOnThemeAndUser": 0.65,
        "ThemeGeneration": 0.5,
        "ThemeGenerationBasedOnUser": 0.6,
        "ChangeTone": 0.15,
        "FindSimilarAssets": 0.2,
        "RewriteAsset": 0.2
    }
    full_data_list = []

    user_prompt_template = "Please generate {} Ad {}, based on the following information:\n"
    with open(inputfile, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            Scenario, FinalUrl, Domain, CategoryName, FullLanguage, AssetType, VerticalBP, BrandVoice, Instruction, JointAsset, JointIsDKI, JointQualityLabel, JointNormKeywords, sd_doc = line.split('\t')
            
            detail_info = "FinalUrl: " + FinalUrl + " \n"
            if random.random() <= 0.8:
                detail_info += "TargetLanguage: " + FullLanguage + " \n"
            if random.random() <= 0.5:
                detail_info += "Domain: " + Domain + " \n"
            reshaped_category_name = reshape_category(CategoryName)
            if len(reshaped_category_name) > 2 and random.random() <= 0.5:
                detail_info += "Category: " + reshaped_category_name + " \n"
            if Scenario in ("ThemeGeneration") and len(VerticalBP) > 2:
                detail_info += "VerticalBestPractice: " + VerticalBP + " \n"
            if Scenario in ("ThemeGeneration") and len(BrandVoice) > 2:
                detail_info += "BrandVoice: " + BrandVoice + " \n"

            raw_assets = JointAsset.split('[SEP]')
            raw_quality = JointQualityLabel.split('[SEP]') 
            raw_isDKI = JointIsDKI.split('[SEP]')
            filtered = [
                (asset.strip(), isDKI.strip())
                for asset, quality, isDKI in zip(raw_assets, raw_quality, raw_isDKI)
                if quality != "0"
            ]

            if not filtered:
                data_LowQuality += 1
                continue

            Asset_list, IsDKI_list = zip(*filtered)
            Asset_list = list(Asset_list)
            IsDKI_list = list(IsDKI_list)
            
            NormKeywords_list = JointNormKeywords.split('#')
            NormKeywords_list = [normKeyword.lower().strip(' +.,!=-#，。！&@￥$()') for normKeyword in NormKeywords_list]
            NormKeywords_list = [kw for kw in NormKeywords_list if len(kw) > 1]


            if len(sd_doc) > 2 and Scenario in ("AssetGeneration", "AssetGenerationBasedOnQuery", "AssetGenerationBasedOnTheme", "AssetGenerationBasedOnThemeAndUser", "ThemeGeneration", "ThemeGenerationBasedOnUser"):
                delimiters = ";\n"
                regexPattern = '|'.join(map(re.escape, delimiters))
                text_list = re.split(regexPattern, sd_doc)
                cleaned_text_list = [sent for sent in text_list if len(sent) > 3]
                SD_text = "; ".join(cleaned_text_list)
                sd_doc = SD_text[:400]
                detail_info += "LandingPage: " + sd_doc + " \n"
            '''
            min_length = max(0, min([len(asset) for asset in Asset_list]) - 5)
            max_length = max([len(asset) for asset in Asset_list]) + 5
            if max_length <= min_length:
                max_length = min_length + 5
            detail_info += "CharacterLimit: between " + str(min_length) + " to " + str(max_length) + " characters. \n"
            '''
            if Scenario in ["AssetGeneration"]:
                if "1" in IsDKI_list:
                    DKI_Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "1"]
                    DKI_Asset_list = list(set(DKI_Asset_list))
                    AssetCnt = len(DKI_Asset_list)
                    if AssetCnt > 1 or random.random() <= 0.5:
                        Instruction = "Incorporate dynamic keyword insertion to make your ad more relevant to query."
                        detail_DKI_info = detail_info + "Instruction: " + Instruction + " \n"

                        message = construct_message_with_length(user_prompt_template, detail_DKI_info, DKI_Asset_list, AssetCnt, AssetType)
                        #print("IsDKI message: ", message)
                        full_data_list.append(message)
                        data_idx += 1
                        data_withDKI += 1
                        scenario_cnt[Scenario] += 1
                        # refine non-DKI assets and insights
                        Asset_list = [asset for asset, isDKI in zip(Asset_list, IsDKI_list) if isDKI == "0"]
                if any(NormKeywords_list):
                    # construct prompt that assets contain keywords
                    len_kw = len(NormKeywords_list)
                    if len_kw >= 2:
                        part_kws = random.sample(NormKeywords_list, int(len_kw/2))
                        ContainKW_Asset_list = get_asset_that_contain_keywords(Asset_list, part_kws)
                        if len(ContainKW_Asset_list) > 2 or (len(ContainKW_Asset_list) >= 1 and random.random() <= 0.5):
                            ContainKW_Asset_list = list(set(ContainKW_Asset_list))
                            #print("\npart_kws: ", part_kws)
                            #print("ContainKW_Asset_list: ", ContainKW_Asset_list)
                            AssetCnt = len(ContainKW_Asset_list)
                            detail_KW_info = detail_info + "Keywords: " + "#".join(part_kws) + " \n"
                            detail_KW_info = detail_KW_info + "Instruction: " + "Ensure relevance by including reasonable keywords in each " + AssetType.lower() + "." + " \n"
                            message = construct_message_with_length(user_prompt_template, detail_KW_info, ContainKW_Asset_list, AssetCnt, AssetType)
                            #print("ContainKW message: ", message)
                            full_data_list.append(message)
                            data_idx += 1
                            data_withTopKW += 1
                            scenario_cnt[Scenario] += 1

                if any(Asset_list) and random.random() <= scenario_probs[Scenario]:
                    # Doing generation without insight for the non-DKI assets
                    Asset_list = list(set(Asset_list))
                    if len(Asset_list) > 3 or random.random() <= 0.2:
                        if len(Asset_list) <= 3:
                            AssetCnt = len(Asset_list)
                        else:
                            AssetCnt = random.randint(1, min(8, len(Asset_list)))
                        #rand_Asset_list = random.sample(Asset_list, AssetCnt)
                        #print("\nAsset_list: ", Asset_list)
                        rand_Asset_list = select_most_dissimilar_assets(Asset_list, AssetCnt)
                        #print("rand_Asset_list: ", rand_Asset_list)
                        if AssetCnt > 1 and random.random() <= 0.5:
                            detail_insight_info = detail_info + "Instruction: " + "Ensure diversity by highlighting various selling pionts." + " \n"
                        else:
                            detail_insight_info = detail_info
                        message = construct_message_with_length(user_prompt_template, detail_insight_info, rand_Asset_list, AssetCnt, AssetType)
                        full_data_list.append(message)
                        data_idx += 1
                        data_general += 1
                        scenario_cnt[Scenario] += 1

            else:
                prob = scenario_probs.get(Scenario, 1.0)
                if "1" not in IsDKI_list and random.random() > prob:
                    continue
                # Doing generation for copilot Scenario
                Asset_list = list(set(Asset_list))
                if len(Asset_list) > 3 or random.random() <= 0.2 or "1" in IsDKI_list or Scenario in ("AssetGenerationBasedOnQuery"):
                    if len(Asset_list) < 3:
                        AssetCnt = len(Asset_list)
                    else:
                        AssetCnt = random.randint(3, min(8, len(Asset_list)))
                    #print("\nAsset_list: ", Asset_list)
                    rand_Asset_list = select_most_dissimilar_assets(Asset_list, AssetCnt)
                    #print("rand_Asset_list: ", rand_Asset_list)
                    detail_insight_info = detail_info + "Instruction: " + Instruction.strip() + " \n"
                    message = construct_message_with_length(user_prompt_template, detail_insight_info, rand_Asset_list, AssetCnt, AssetType)
                    #print("No Insight message: ", message)
                    full_data_list.append(message)
                    data_idx += 1
                    data_copilot += 1
                    scenario_cnt[Scenario] += 1
            

            if input_row % 50000 == 0:
                print("\nProcessing row: ", input_row)
                print("Total low quality data rows: ", data_LowQuality)
                print("Total data rows: ", data_idx)
                print("Total data with DKI: ", data_withDKI)
                print("Total data with TopKW: ", data_withTopKW)
                print("Total data with General: ", data_general)
                print("Total data with Copilot: ", data_copilot)
                for scenario, count in scenario_cnt.items():
                    print(f"Scenario '{scenario}' count: {count}")

            input_row += 1


    print("\nTotal data rows for model: ", data_idx)
    print("Total low quality data rows: ", data_LowQuality)
    print("Total data with DKI: ", data_withDKI)
    print("Total data with TopKW: ", data_withTopKW)
    print("Total data with General: ", data_general)
    print("Total data with Copilot: ", data_copilot)
    for scenario, count in scenario_cnt.items():
        print(f"Scenario '{scenario}' count: {count}")

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

def split_data(input_file, output_file, split_ratio=0.5):
    with open(input_file, 'r', encoding='utf-8') as fr:
        data_list = json.load(fr)

    split_idx = int(len(data_list) * split_ratio)
    splited_data = data_list[:split_idx]
    print("Splited data size: ", len(splited_data))

    with open(output_file, 'w', encoding='utf-8') as fw:
        json.dump(splited_data, fw, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GenerateTrainTestDataForLLM')
    parser.add_argument('-i', '--input', help='input file', default="./CombinedAssets_20250528.tsv")
    #parser.add_argument('-i', '--input', help='input file', default="../data/AssetGeneration/test.tsv")
    parser.add_argument('-fu', '--FullData', help='json file', default="./FullData_20250528.json")
    parser.add_argument('-tr', '--train', help='json file', default="./train_20250528.json")
    parser.add_argument('-te', '--test', help='json file', default="./test_20250528.json")
    parser.add_argument('-small_te', '--small_test', help='json file', default="./small_test_20250528.json")
    parser.add_argument('-small_tr', '--small_train', help='json file', default="./small_train_20250528.json")
    args = parser.parse_args()
    
    main(args)
    '''
    out_prompt_file = "./inference_prompt_AddDiversity.tsv"
    out_response_file = "./inference_groundtruth_AddDiversity.tsv"
    ConvertJsonToInferenceData(args.small_test, out_prompt_file, out_response_file)
    '''
    #split_data(args.train, args.small_train, 0.5)


