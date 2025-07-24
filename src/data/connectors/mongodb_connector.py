import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from .base_connector import BaseDataConnector
from loguru import logger


class MongoDataConnector(BaseDataConnector):
    """MongoDB数据连接器"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "market_data",
        collection: str = "ohlcv",
        username: Optional[str] = None,
        password: Optional[str] = None,
        auth_source: str = "admin",
        timeout_ms: int = 5000
    ):
        """
        初始化MongoDB数据连接器
        
        Args:
            host: MongoDB主机地址
            port: MongoDB端口
            database: 数据库名称
            collection: 集合名称
            username: 用户名
            password: 密码
            auth_source: 认证数据库
            timeout_ms: 连接超时时间（毫秒）
        """
        self.host = host
        self.port = port
        self.database_name = database
        self.collection_name = collection
        self.username = username
        self.password = password
        self.auth_source = auth_source
        self.timeout_ms = timeout_ms
        
        self.client: Optional[MongoClient] = None
        self.database = None
        self.collection = None
        self.connected = False
    
    def connect(self) -> bool:
        """建立MongoDB连接"""
        try:
            # 构建连接字符串
            if self.username and self.password:
                uri = f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.auth_source}"
            else:
                uri = f"mongodb://{self.host}:{self.port}"
            
            # 创建客户端连接
            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms
            )
            
            # 测试连接
            self.client.admin.command('ping')
            
            # 获取数据库和集合
            self.database = self.client[self.database_name]
            self.collection = self.database[self.collection_name]
            
            self.connected = True
            logger.info(f"Connected to MongoDB: {self.host}:{self.port}/{self.database_name}")
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        从MongoDB获取价格数据
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            包含OHLCV数据的DataFrame
        """
        if not self.connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            # 构建查询条件
            query = {
                "symbol": symbol,
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            # 查询数据
            cursor = self.collection.find(query).sort("timestamp", 1)
            documents = list(cursor)
            
            if not documents:
                logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(documents)
            
            # 删除MongoDB的_id字段
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # 标准化列名
            column_mapping = {
                'date': 'timestamp',
                'datetime': 'timestamp',
                'time': 'timestamp',
                'code': 'symbol',
                'vol': 'volume'
            }
            df = df.rename(columns=column_mapping)
            
            # 确保必需列存在
            required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records for {symbol} from MongoDB")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from MongoDB: {e}")
            raise
    
    def get_symbols(self) -> List[str]:
        """获取可用品种列表"""
        if not self.connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            # 获取所有唯一的symbol值
            symbols = self.collection.distinct("symbol")
            symbols = [str(symbol) for symbol in symbols if symbol]
            
            logger.info(f"Found {len(symbols)} symbols in MongoDB")
            return sorted(symbols)
            
        except Exception as e:
            logger.error(f"Failed to get symbols list from MongoDB: {e}")
            return []
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据有效性"""
        if data.empty:
            logger.warning("DataFrame is empty")
            return False
        
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
        """关闭MongoDB连接"""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.collection = None
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    def insert_data(self, data: List[Dict[str, Any]]) -> bool:
        """
        插入数据到MongoDB（额外功能）
        
        Args:
            data: 要插入的数据列表
            
        Returns:
            插入是否成功
        """
        if not self.connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            if data:
                result = self.collection.insert_many(data)
                logger.info(f"Inserted {len(result.inserted_ids)} documents")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """创建索引以提高查询性能"""
        if not self.connected or not self.collection:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            # 创建复合索引
            self.collection.create_index([("symbol", 1), ("timestamp", 1)])
            self.collection.create_index("timestamp")
            self.collection.create_index("symbol")
            
            logger.info("Created indexes for better performance")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False