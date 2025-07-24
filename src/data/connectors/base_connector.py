from abc import ABC, abstractmethod
from typing import List
import pandas as pd
from datetime import datetime
from src.data.models.base_models import PriceData


class BaseDataConnector(ABC):
    """数据连接器抽象基类"""
    
    @abstractmethod
    def connect(self) -> bool:
        """建立连接"""
        pass
    
    @abstractmethod
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """获取价格数据"""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """获取可用品种列表"""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据有效性"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭连接"""
        pass
    
    def standardize_data(self, data: pd.DataFrame) -> List[PriceData]:
        """标准化数据格式"""
        if not self.validate_data(data):
            raise ValueError("Invalid data format")
            
        price_data_list = []
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        # 检查必需列
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        for _, row in data.iterrows():
            try:
                price_data = PriceData(
                    timestamp=pd.to_datetime(row['timestamp']),
                    symbol=str(row['symbol']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=float(row['volume'])
                )
                price_data_list.append(price_data)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Data conversion error in row {row.name}: {e}")
                
        return price_data_list


class DataConnectorFactory:
    """数据连接器工厂"""
    
    @staticmethod
    def create_connector(connector_type: str, **kwargs) -> BaseDataConnector:
        """创建数据连接器实例"""
        if connector_type.lower() == 'csv':
            from .csv_connector import CSVDataConnector
            return CSVDataConnector(**kwargs)
        elif connector_type.lower() == 'mongodb':
            from .mongodb_connector import MongoDataConnector
            return MongoDataConnector(**kwargs)
        else:
            raise ValueError(f"Unsupported connector type: {connector_type}")