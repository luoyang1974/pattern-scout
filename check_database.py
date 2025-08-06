#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查数据库中的旗形记录
"""
import sqlite3
import pandas as pd

def check_database():
    conn = sqlite3.connect('output/data/patterns.db')
    cursor = conn.cursor()
    
    # 检查表结构
    cursor.execute('PRAGMA table_info(patterns)')
    columns = cursor.fetchall()
    print('Patterns表结构:')
    for col in columns:
        print(f'  {col[1]}: {col[2]}')
    
    # 统计记录数量
    cursor.execute('SELECT COUNT(*) FROM patterns')
    total_count = cursor.fetchone()[0]
    print(f'\n总记录数: {total_count}')
    
    # 按品种统计
    cursor.execute('SELECT COUNT(*), symbol FROM patterns GROUP BY symbol')
    print('\n按品种统计:')
    for row in cursor.fetchall():
        print(f'  {row[1]}: {row[0]}条记录')
    
    # 查看最新记录
    try:
        # 先获取所有列名
        cursor.execute('SELECT * FROM patterns LIMIT 1')
        columns_from_data = [description[0] for description in cursor.description]
        print(f'\n实际列名: {columns_from_data}')
        
        # 查看几条记录
        cursor.execute('SELECT * FROM patterns LIMIT 5')
        rows = cursor.fetchall()
        print('\n前5条记录:')
        for i, row in enumerate(rows):
            print(f'记录 {i+1}:')
            for col_name, value in zip(columns_from_data, row):
                print(f'  {col_name}: {value}')
            print()
    except Exception as e:
        print(f'查询记录时出错: {e}')
    
    conn.close()

if __name__ == "__main__":
    check_database()