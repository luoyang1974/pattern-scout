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
        self._data_range_cache = {}  # 缓存数据时间范围
        
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
    
    def get_data(self, symbol: str, start_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        从CSV文件获取价格数据
        
        Args:
            symbol: 品种代码
            start_date: 开始日期（可选，默认使用数据的实际开始时间）
            end_date: 结束日期（可选，默认使用数据的实际结束时间）
            
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
            
            # 数据去重处理（基于timestamp列去重，保留最后一条记录）
            original_count = len(df)
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            deduplicated_count = len(df)
            
            if original_count > deduplicated_count:
                logger.info(f"Data deduplication: removed {original_count - deduplicated_count} duplicate records")
                logger.info(f"After deduplication: {deduplicated_count} records")
            
            # 排序以确保时间顺序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 获取数据的实际时间范围
            actual_start = df['timestamp'].min()
            actual_end = df['timestamp'].max()
            
            # 如果没有指定时间范围，使用数据的实际范围
            if start_date is None:
                start_date = actual_start
                logger.info(f"Using actual data start date: {start_date}")
            
            if end_date is None:
                end_date = actual_end
                logger.info(f"Using actual data end date: {end_date}")
            
            # 验证时间范围是否合理
            if start_date > actual_end or end_date < actual_start:
                logger.warning(f"Requested date range ({start_date} to {end_date}) does not overlap with data range ({actual_start} to {actual_end})")
                logger.warning("Using full data range instead")
                start_date = actual_start
                end_date = actual_end
            
            # 按时间范围过滤
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df.loc[mask].copy()
            
            # 检查过滤后的数据量
            if len(df) == 0:
                raise ValueError(f"No data found for {symbol} in date range {start_date} to {end_date}")
            
            logger.info(f"Loaded {len(df)} records for {symbol} from {start_date} to {end_date}")
            logger.info(f"Data covers period: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
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
    
    def get_data_range(self, symbol: str) -> tuple:
        """
        获取指定品种的数据时间范围
        
        Args:
            symbol: 品种代码
            
        Returns:
            (start_date, end_date) 元组
        """
        if symbol in self._data_range_cache:
            return self._data_range_cache[symbol]
        
        try:
            # 查找CSV文件
            csv_files = list(self.data_directory.glob(f"*{symbol}*.csv"))
            if not csv_files:
                csv_file = self.data_directory / f"{symbol}.csv"
                if not csv_file.exists():
                    raise FileNotFoundError(f"No CSV file found for symbol: {symbol}")
                csv_files = [csv_file]
            
            csv_file = max(csv_files, key=os.path.getmtime)
            
            # 读取前几行来确定时间戳列
            df_sample = pd.read_csv(csv_file, nrows=5)
            
            # 找到时间戳列名（优先匹配原始列名）
            timestamp_col = None
            original_columns = df_sample.columns.tolist()
            
            # 按优先级检查可能的时间列名
            possible_timestamp_cols = ['Datetime', 'Date', 'Timestamp', 'Time', 
                                     'datetime', 'date', 'timestamp', 'time']
            
            for col in possible_timestamp_cols:
                if col in original_columns:
                    timestamp_col = col
                    break
            
            if timestamp_col is None:
                raise ValueError(f"No timestamp column found. Available columns: {original_columns}")
            
            logger.info(f"Using timestamp column: {timestamp_col}")
            
            # 读取完整时间戳列
            df_time = pd.read_csv(csv_file, usecols=[timestamp_col])
            df_time[timestamp_col] = pd.to_datetime(df_time[timestamp_col])
            
            start_date = df_time[timestamp_col].min()
            end_date = df_time[timestamp_col].max()
            
            # 缓存结果
            self._data_range_cache[symbol] = (start_date, end_date)
            
            logger.info(f"Data range for {symbol}: {start_date} to {end_date}")
            return start_date, end_date
            
        except Exception as e:
            logger.error(f"Failed to get data range for {symbol}: {e}")
            return None, None
    
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