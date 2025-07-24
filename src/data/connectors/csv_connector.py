import pandas as pd
import os
from typing import List
from datetime import datetime
from pathlib import Path
from .base_connector import BaseDataConnector
from loguru import logger


class CSVDataConnector(BaseDataConnector):
    """CSV文件数据连接器"""
    
    def __init__(self, data_directory: str = "data/csv/"):
        """
        初始化CSV数据连接器
        
        Args:
            data_directory: CSV文件存储目录
        """
        self.data_directory = Path(data_directory)
        self.connected = False
        
    def connect(self) -> bool:
        """建立连接（检查目录是否存在）"""
        try:
            if not self.data_directory.exists():
                logger.warning(f"Data directory does not exist: {self.data_directory}")
                self.data_directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created data directory: {self.data_directory}")
            
            self.connected = True
            logger.info(f"Connected to CSV data source: {self.data_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to CSV data source: {e}")
            return False
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从CSV文件获取价格数据
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        if not self.connected:
            raise ConnectionError("Not connected to data source")
        
        # 查找匹配的CSV文件
        csv_files = list(self.data_directory.glob(f"*{symbol}*.csv"))
        if not csv_files:
            # 尝试直接使用symbol作为文件名
            csv_file = self.data_directory / f"{symbol}.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"No CSV file found for symbol: {symbol}")
            csv_files = [csv_file]
        
        # 如果有多个匹配文件，选择最新的
        csv_file = max(csv_files, key=os.path.getmtime)
        logger.info(f"Loading data from: {csv_file}")
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 标准化列名（处理可能的不同命名方式）
            column_mapping = {
                'date': 'timestamp',
                'datetime': 'timestamp',
                'time': 'timestamp',
                'symbol': 'symbol',
                'code': 'symbol',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'vol': 'volume'
            }
            
            # 重命名列
            df.columns = df.columns.str.lower()
            df = df.rename(columns=column_mapping)
            
            # 确保symbol列存在
            if 'symbol' not in df.columns:
                df['symbol'] = symbol
            
            # 转换时间戳
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise ValueError("No timestamp column found")
            
            # 按时间范围过滤
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df.loc[mask].copy()
            
            # 排序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records for {symbol} from {start_date} to {end_date}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from {csv_file}: {e}")
            raise
    
    def get_symbols(self) -> List[str]:
        """获取可用品种列表"""
        if not self.connected:
            raise ConnectionError("Not connected to data source")
        
        symbols = []
        try:
            # 扫描CSV文件
            csv_files = list(self.data_directory.glob("*.csv"))
            
            for csv_file in csv_files:
                # 从文件名提取品种代码
                symbol = csv_file.stem
                symbols.append(symbol)
            
            logger.info(f"Found {len(symbols)} symbols in CSV directory")
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Failed to get symbols list: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据有效性"""
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        
        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # 检查数据类型
        try:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 检查是否有无效数据
            if data[['open', 'high', 'low', 'close', 'volume']].isnull().any().any():
                logger.warning("Found invalid numeric data")
                return False
                
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False
        
        # 检查OHLC逻辑
        invalid_ohlc = (
            (data['high'] < data[['open', 'close']].max(axis=1)) |
            (data['low'] > data[['open', 'close']].min(axis=1))
        ).any()
        
        if invalid_ohlc:
            logger.warning("Found invalid OHLC data")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def close(self) -> None:
        """关闭连接"""
        self.connected = False
        logger.info("Disconnected from CSV data source")