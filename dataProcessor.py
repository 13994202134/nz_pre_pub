from datetime import datetime
from gpt_data_processing.AbstractDataProcessor import AbstractDataProcessor
from gpt_data_processing.FeatureCalculator import FeatureCalculator

from gpt_data_processing.FeatureCalculator_fund_vix import FeatureCalculator_fund_vix
from gpt_data_processing.FeatureCalculator_sentiment import FeatureCalculator_sentiment
from gpt_data_processing.FeatureCalculator_fund_vix_merge import FeatureCalculator_fund_vix_merge
from gpt_data_processing.FeatureMapper import FeatureMapper

from itertools import combinations
import pandas as pd
import pprint
from main_class.AbstractDataSource import AbstractDataSource
from utils.logger_utils import log_info_message,log_warning_message,log_error_message
import time
import psutil
from sklearn.preprocessing import MinMaxScaler
from utils.convert import convert_dataframe
from tqdm import tqdm
class DataProcessor(AbstractDataProcessor):
    def perform_feature_engineering(self, feature_combinations, parameter_period):
        """
        执行特征工程，包括特征组合、计算特征、合并数据、去重处理、特征值归一化等步骤。

        参数:
            feature_combinations (list): 包含特征组合的列表。
            parameter_period: 参数期限（短期、中期或长期），例如 'short_term'、'medium_term' 或 'long_term'。

        返回:
            tuple: 包含所有特征组合和归一化后的数据列表。
        """
        log_info_message(f'-------进入perform_feature_engineering函数----:') 
        normalized_data_list = []  # 存储归一化后的数据
        feature_records = []  # 存储记录
        try:
            # 获取特征组合
            all_feature_combinations = self.get_feature_combinations(feature_combinations)
            log_info_message(f'-------特征组合----:\n{all_feature_combinations}') 
            print('----创建特征计算器实例前---------')
            # 创建特征计算器实例
            feature_calculator = FeatureCalculator(self.data_source, parameter_period)
            total_combinations = len(all_feature_combinations)
            total_time_spent = 0
            print('----创建特征计算器实例后---------')
            # 根据特征组合执行特征工程
            for idx, features in enumerate(all_feature_combinations):
                try:
 
                    # 处理每个特性对应的数据
                    dfs_to_merge = feature_calculator.calculate_features(features)
                    print('调用的特征：',features)
                    print('调用了处理每个特性对应的数据feature_calculator.calculate_features返回的：',dfs_to_merge)
                    # 转换数据帧中的逗号字符串为浮点数
                    dfs_to_merge = [convert_dataframe(df) for df in dfs_to_merge]
                    
                    # 合并处理后的数据
                    if dfs_to_merge:
                        df_merged = pd.concat(dfs_to_merge, axis=1)
                    else:
                        df_merged = pd.DataFrame()  # 如果没有数据要合并，创建一个空的DataFrame
                    
                    # ✅ 检查合并后的列是否有重复值
                    duplicates = df_merged.T.duplicated()
                    if duplicates.any():
                        dup_cols = df_merged.columns[duplicates].tolist()
                        print(f"⚠️ 特征组合 {features} 中发现重复列: {dup_cols}")
                    else:
                        print(f"✅ 特征组合 {features} 没有重复列")

                    # 去重处理
                    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]    
                    # 将列名进行处理，只保留每列的前半部分作为标识
                    column_names = [col.split()[0] for col in df_merged.columns]
                    # 为DataFrame设置新的列名
                    df_merged.columns = column_names

                    # 使用drop_duplicates方法删除重复的列，并保留每列的第一个出现的部分
                    df_merged_unique = df_merged.loc[:, ~df_merged.columns.duplicated()]    
                    # 合并后再进行去重处理
                    df_merged_unique = df_merged_unique.loc[:, ~df_merged_unique.columns.duplicated()]
                                        
                    # 处理可能的 NaN 值
                    df_merged = df_merged.fillna(0)
                    
                    # 检查 df_merged 是否为空
                    if df_merged.empty:
                        log_info_message('合并后的 DataFrame 为空，跳过归一化步骤。')
                        normalized_data_list.append(pd.DataFrame())  # 添加一个空的 DataFrame 以保持列表结构
                    else:
                        # 进行特征值归一化
                        scaler = MinMaxScaler()
                        values_normalized = scaler.fit_transform(df_merged.values)
                        
                        # 创建归一化后的 DataFrame，并设置列名
                        df_normalized = pd.DataFrame(values_normalized, columns=df_merged.columns)
                        
                        normalized_data_list.append(df_normalized)  # 将归一化后的数据添加到列表中
                    

                    # 获取当前特征组合的参数
                    feature_params = {feature: feature_calculator.parameters.get(feature, {}) for feature in features}  # 获取当前特征组合的参数

                    # print(f'特征工程返回的特征参数:\n{feature_params}')

                except Exception as e:
                    log_error_message(f'处理特征组合 {features} 时出错: {str(e)}')
                    continue
                
        except Exception as e:
            log_error_message(f'初始化特征工程时出错: {str(e)}')

        print(f'dataProcessor中特征工程返回的数据:\n{normalized_data_list}')   

        return all_feature_combinations, normalized_data_list

    
    
    
    

    