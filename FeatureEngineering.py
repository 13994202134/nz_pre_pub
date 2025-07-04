import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from gpt_data_processing.FeatureCalculator import FeatureCalculator
from gpt_data_processing.NZDataProcessor import NZDataProcessor
from gpt_data_processing.NzFeatureCalculator import NzFeatureCalculator
from utils.logger_utils import log_info_message,log_warning_message,log_error_message
from utils.test_utils import check_column_duplicates_by_date
class FeatureEngineering:
    def __init__(self, file_path, processor, parameter_period=None):
        """
        初始化特征工程类

        :param file_path: 特征工程结果文件路径
        :param processor: 特征工程处理器对象
        :param parameter_period: 参数周期，默认为None
        """
        self.file_path = file_path
        self.processor = processor
        self.parameter_period = parameter_period if parameter_period is not None else None

        
        # 打印multiProcessor的名称
        # print(f"Processor class name: {self.processor.__class__.__name__}")

        # 根据类名决定调用FeatureCalculator或NzFeatureCalculator
        if isinstance(self.processor, NZDataProcessor):
            # print('进入了NZDataProcessor分支')
            self.feature_calculator = NZDataProcessor(self.processor.data_source)
            
        else:
            print('进入了DataProcessor分支')
            self.feature_calculator = FeatureCalculator(self.processor.data_source, self.parameter_period)
            


        # 如果需要再次设置数据，可以调用set_data
        self.feature_calculator.set_data(self.processor.data_source)
        
    def perform_engineering_with_params(self, feature_combinations, config, start_date, end_date):
        """
        执行特征工程 (带参数)
        """
        return self.processor.perform_feature_engineering(feature_combinations, self.parameter_period, config, start_date, end_date)

    def perform_engineering_without_params(self, feature_combinations):
        """
        执行特征工程 (不带参数)
        """
        return self.processor.perform_feature_engineering(feature_combinations, self.parameter_period)


    def load_results(self):
        """
        加载特征工程结果

        :return: 已保存的特征组合和归一化数据
        """
        # print(f'文件名：{self.file_path}')
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'rb') as f:
                    data =  pickle.load(f)
                    return data[0], data[1]
            except Exception as e:
                print(f"加载特征工程结果失败: {e}")
        return [], []


    def save_results(self, feature_combinations, normalized_data):
        """
        保存特征工程结果
        
        :param feature_combinations: 特征组合
        :param normalized_data: 归一化数据
        """
        # print(f"在保存函数中接收到的参数: 文件路径: {self.file_path}, 特征组合: {feature_combinations}, 归一化数据: {normalized_data}")

        def convert_to_serializable(data):
            if isinstance(data, pd.DataFrame):
                return data.to_dict(orient='list')
            elif isinstance(data, list):
                return [convert_to_serializable(item) for item in data]
            elif isinstance(data, dict):
                return {key: convert_to_serializable(value) for key, value in data.items()}
            else:
                return data

        try:
            serializable_normalized_data = convert_to_serializable(normalized_data)

            # 保存到文件或者其他存储系统
            with open(self.file_path, 'wb') as f:
                # pickle.dump({'feature_combinations': feature_combinations, 'normalized_data': serializable_normalized_data}, f)
                pickle.dump((feature_combinations, normalized_data), f)
            log_info_message(f"成功将特征工程结果保存到文件: {self.file_path}")
        except Exception as e:
            print(f"保存特征工程结果时出错: {e}")

    def process_combinations(self, saved_combinations, saved_data, all_combinations):
        """
        过滤已保存的特征组合

        :param saved_combinations: 已保存的特征组合
        :param saved_data: 已保存的归一化数据
        :param all_combinations: 所有特征组合
        :return: 过滤后的特征组合和归一化数据
        """
        filtered_combinations = []
        filtered_normalized_data = []
        for feature_comb in saved_combinations:
            if feature_comb in all_combinations:
                index = saved_combinations.index(feature_comb)
                filtered_combinations.append(feature_comb)
                filtered_normalized_data.append(saved_data[index])
        return filtered_combinations, filtered_normalized_data



    def process_and_save_combinations(self, saved_feature_combinations, saved_normalized_data, all_feature_combinations, config, start_date, end_date):
        """
        处理并保存特征组合。
        
        流程：
            检查已保存的特征工程数据
            找到与新传入特征组合的差异
                是（有差异）：删除多余的和不再用的特征组合
                否（无差异）：保留不变的特征组合
            执行特征工程

        参数:
        - saved_feature_combinations: 已保存的特征组合
        - saved_normalized_data: 已保存的标准化数据
        - all_feature_combinations: 所有特征组合
        - feature_combinations_0: 初始特征组合

        返回:
        - unique_feature_combinations: 独特的特征组合
        - unique_normalized_data: 独特的标准化数据
        """
        try:
            print("Starting process_and_save_combinations")

            # 转换 saved_feature_combinations 为元组形式
            if saved_feature_combinations:
                saved_feature_combinations = [
                    tuple(feature.split(',')) if isinstance(feature, str) else feature
                    for feature in saved_feature_combinations
                ]
            
            # 调试: 打印加载的 saved_normalized_data 内容
            # print("---- Debug: Before processing, saved_normalized_data ----")
            # for df in saved_normalized_data:
            #     print(df.head())  # 打印前5行
            
            # 计算 feature_combinations_0
            feature_combinations_0 = [
                feature_combination for feature_combination in all_feature_combinations
                if feature_combination not in saved_feature_combinations
            ]


            if (not saved_feature_combinations and len(saved_normalized_data) == 0) or \
   (saved_feature_combinations and len(saved_normalized_data) == 0):
                print("Saved data is present, processing combinations...")
                
                feature_combinations, normalized_data = self.process_combinations(
                    saved_feature_combinations, saved_normalized_data, all_feature_combinations
                )
                if isinstance(self.processor, NZDataProcessor):
                    # 如果processor是NZDataProcessor, 则传递config, start_date, end_date
                    new_feature_combinations, new_normalized_data = self.perform_engineering_with_params(
                        feature_combinations_0, config, start_date, end_date
                    )
                else:
                    # 否则，不传递这些参数
                    new_feature_combinations, new_normalized_data = self.perform_engineering_without_params(
                        feature_combinations_0
                    )
                # 调试: 打印生成的 normalized_data 内容
                # print("---- Debug: After feature engineering, normalized_data ----")
                # for i, df in enumerate(normalized_data):
                #     print(f"DataFrame {i} - Head of data:")
                #     print(df.head())
                #     if 'Close' in df.columns:
                #         # print(f"Close column for DataFrame {i}:")
                #         print(df['Close'].head())  # 打印 'Close' 列的前几行
                log_info_message(f"---读取成功，执行完特征工程得到的值：{feature_combinations}")

                print("Feature combinations and normalized data after engineering:")
                print("Feature combinations:", feature_combinations)
                print("Checking for NaN values in normalized_data...")
                for i, df in enumerate(normalized_data):
                    print(f"DataFrame {i} NaN check:")
                    print(df.isna().sum())

                combined_feature_combinations = feature_combinations + new_feature_combinations
                combined_normalized_data = normalized_data + new_normalized_data
                

                # 调试: 打印合并后的数据
                print("---- Debug: Combined normalized_data before saving ----")
                for i, df in enumerate(combined_normalized_data):
                    print(f"DataFrame {i} - Head of data:")
                    print(df.head())
                    if 'Close' in df.columns:
                        print(f"Close column for DataFrame {i}:")
                        print(df['Close'].head())  # 打印 'Close' 列的前几行

                # 构建组合到DataFrame的映射（后者覆盖前者，避免 index 找错）
                comb_to_df = {}
                for comb, df in zip(combined_feature_combinations, combined_normalized_data):
                    comb_to_df[tuple(comb)] = df  # 转成 tuple 以保证 hashable
                unique_feature_combinations = list(comb_to_df.keys())
                unique_normalized_data = list(comb_to_df.values())


                self.save_results(unique_feature_combinations, unique_normalized_data)
            else:
                if isinstance(self.processor, NZDataProcessor):
                    # 调用带参数的特征工程方法
                    feature_combinations, normalized_data = self.perform_engineering_with_params(
                        feature_combinations_0, config, start_date, end_date
                    )
                else:
                    # 调用不带参数的特征工程方法
                    feature_combinations, normalized_data = self.perform_engineering_without_params(
                        feature_combinations_0
                    )
                print(f"---读取不成功，执行完特征工程得到的值：{feature_combinations}")
                unique_feature_combinations = feature_combinations
                unique_normalized_data = normalized_data
                
                # 调试: 打印特征工程后数据
                # print("---- Debug: Normalized data after feature engineering ----")
                # for i, df in enumerate(normalized_data):
                #     print(f"DataFrame {i} - Head of data:")
                #     print(df.head())
                #     if 'Close' in df.columns:
                #         print(f"Close column for DataFrame {i}:")
                #         print(df['Close'].head())
 
                self.save_results(unique_feature_combinations, unique_normalized_data)

        except Exception as e:
            print(f"处理和保存特征组合时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈信息
            return [], []  # 在异常情况下返回空列表

        return unique_feature_combinations, unique_normalized_data