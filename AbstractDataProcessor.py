from abc import ABC, abstractmethod
from itertools import combinations
from main_class import AbstractDataSource
from utils.logger_utils import log_info_message,log_error_message,log_debug_message,log_warning_message

class AbstractDataProcessor(ABC):
    def __init__(self, data_source: AbstractDataSource):
        self.data_source = data_source
        
    def set_data(self, data_source: AbstractDataSource):
        self.data_source = data_source



    def get_feature_combinations(self, features):
        log_debug_message(f"features: {features}")
        processed_features = []
        for feature in features:
            processed_features.extend(feature.split(','))
        
        feature_combinations = []
        # 输出处理后的特征
        log_debug_message(f"Processed features: {processed_features}")
        
        for r in range(1, len(processed_features) + 1):
            current_combinations = list(combinations(processed_features, r))
            feature_combinations.extend(current_combinations)
            
            # 输出每次生成的组合
            log_debug_message(f"Generated combinations of length {r}: {current_combinations}")
        
        # 输出最终生成的所有特征组合
        log_debug_message(f"Total feature combinations generated: {len(feature_combinations)}")
        log_debug_message(f"Feature combinations: {feature_combinations}")
        
        return feature_combinations


    @abstractmethod
    def perform_feature_engineering(self, feature_combinations, parameter_period, config, start_date, end_date):
        pass
