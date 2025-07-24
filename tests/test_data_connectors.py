import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile

from src.data.connectors.csv_connector import CSVDataConnector
from src.data.models.base_models import PriceData


class TestCSVDataConnector(unittest.TestCase):
    """测试CSV数据连接器"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.connector = CSVDataConnector(data_directory=self.temp_dir)
        
        # 创建测试CSV文件
        self.test_symbol = "TEST"
        test_data = {
            'timestamp': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'symbol': [self.test_symbol] * 30,
            'open': np.random.uniform(95, 105, 30),
            'high': np.random.uniform(105, 115, 30),
            'low': np.random.uniform(85, 95, 30),
            'close': np.random.uniform(95, 105, 30),
            'volume': np.random.randint(1000000, 5000000, 30)
        }
        
        # 确保OHLC数据逻辑正确
        for i in range(30):
            test_data['high'][i] = max(test_data['open'][i], test_data['close'][i], test_data['high'][i])
            test_data['low'][i] = min(test_data['open'][i], test_data['close'][i], test_data['low'][i])
        
        self.test_df = pd.DataFrame(test_data)
        csv_path = Path(self.temp_dir) / f"{self.test_symbol}.csv"
        self.test_df.to_csv(csv_path, index=False)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_connect(self):
        """测试连接"""
        result = self.connector.connect()
        self.assertTrue(result)
        self.assertTrue(self.connector.connected)
    
    def test_get_symbols(self):
        """测试获取品种列表"""
        self.connector.connect()
        symbols = self.connector.get_symbols()
        self.assertIn(self.test_symbol, symbols)
    
    def test_get_data(self):
        """测试获取数据"""
        self.connector.connect()
        
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        data = self.connector.get_data(self.test_symbol, start_date, end_date)
        
        # 检查数据不为空
        self.assertFalse(data.empty)
        
        # 检查必需列存在
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
        
        # 检查数据类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(data['timestamp']))
        
        # 检查符号正确
        self.assertTrue((data['symbol'] == self.test_symbol).all())
    
    def test_validate_data(self):
        """测试数据验证"""
        self.connector.connect()
        
        # 测试有效数据
        valid_result = self.connector.validate_data(self.test_df)
        self.assertTrue(valid_result)
        
        # 测试缺少列的数据
        invalid_df = self.test_df.drop('volume', axis=1)
        invalid_result = self.connector.validate_data(invalid_df)
        self.assertFalse(invalid_result)
    
    def test_standardize_data(self):
        """测试数据标准化"""
        self.connector.connect()
        
        price_data_list = self.connector.standardize_data(self.test_df)
        
        # 检查返回类型
        self.assertIsInstance(price_data_list, list)
        self.assertGreater(len(price_data_list), 0)
        
        # 检查第一个元素是PriceData类型
        self.assertIsInstance(price_data_list[0], PriceData)
        
        # 检查数据完整性
        first_price_data = price_data_list[0]
        self.assertEqual(first_price_data.symbol, self.test_symbol)
        self.assertIsInstance(first_price_data.timestamp, datetime)


class TestDataModels(unittest.TestCase):
    """测试数据模型"""
    
    def test_price_data_validation(self):
        """测试PriceData验证"""
        # 测试有效数据
        valid_data = PriceData(
            timestamp=datetime.now(),
            symbol="TEST",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000
        )
        
        # 应该创建成功
        self.assertEqual(valid_data.symbol, "TEST")
        
        # 测试无效数据（high < close）
        with self.assertRaises(ValueError):
            invalid_data = PriceData(
                timestamp=datetime.now(),
                symbol="TEST",
                open=100.0,
                high=90.0,  # high < close，应该报错
                low=95.0,
                close=102.0,
                volume=1000000
            )


if __name__ == '__main__':
    unittest.main()