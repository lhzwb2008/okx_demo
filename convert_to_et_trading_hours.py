#!/usr/bin/env python3
"""
BTC数据时区转换和交易时间过滤脚本
将UTC时间转换为美东时区，并只保留美股交易时间（9:40-16:00）的数据
"""

import os
import sys
import pandas as pd# pyright: ignore
import pytz
from datetime import datetime, time
import argparse

class BTCETTradingHoursConverter:
    def __init__(self):
        """初始化转换器"""
        # 时区设置
        self.utc_tz = pytz.UTC
        self.et_tz = pytz.timezone('US/Eastern')  # 美东时区（自动处理夏令时）
        
        # 美股交易时间设置
        self.trading_start = time(9, 40)  # 9:40 AM
        self.trading_end = time(16, 0)    # 4:00 PM
        
        print("BTC数据时区转换器初始化完成")
        print(f"目标时区: {self.et_tz}")
        print(f"交易时间: {self.trading_start.strftime('%H:%M')} - {self.trading_end.strftime('%H:%M')} (美东时间)")
    
    def convert_timezone_and_filter(self, input_file: str, output_file: str = None) -> str:
        """
        转换时区并过滤交易时间
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径，如果为None则自动生成
            
        Returns:
            输出文件路径
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        print(f"正在读取文件: {input_file}")
        
        # 读取CSV文件
        df = pd.read_csv(input_file)
        print(f"原始数据: {len(df):,} 条记录")
        
        # 验证必要的列是否存在
        required_columns = ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"缺少必要的列: {missing_columns}")
        
        # 转换DateTime列为pandas datetime对象
        print("正在转换时间格式...")
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        
        # 假设原始数据是UTC时间，添加UTC时区信息
        df['DateTime_UTC'] = df['DateTime'].dt.tz_localize(self.utc_tz)
        
        # 转换为美东时区
        print("正在转换为美东时区...")
        df['DateTime_ET'] = df['DateTime_UTC'].dt.tz_convert(self.et_tz)
        
        # 提取时间部分用于过滤
        df['Time_ET'] = df['DateTime_ET'].dt.time
        df['Date_ET'] = df['DateTime_ET'].dt.date
        df['Weekday'] = df['DateTime_ET'].dt.weekday  # 0=Monday, 6=Sunday
        
        # 过滤条件：
        # 1. 工作日（周一到周五，0-4）
        # 2. 交易时间（9:40-16:00）
        print("正在过滤交易时间...")
        
        # 工作日过滤
        weekday_filter = df['Weekday'] <= 4  # 周一到周五
        
        # 交易时间过滤
        time_filter = (df['Time_ET'] >= self.trading_start) & (df['Time_ET'] <= self.trading_end)
        
        # 应用过滤条件
        trading_hours_df = df[weekday_filter & time_filter].copy()
        
        print(f"工作日数据: {len(df[weekday_filter]):,} 条记录")
        print(f"交易时间数据: {len(trading_hours_df):,} 条记录")
        print(f"过滤比例: {len(trading_hours_df)/len(df)*100:.1f}%")
        
        # 创建最终的DataFrame，使用美东时间
        result_df = trading_hours_df[['DateTime_ET', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # 重命名DateTime列并格式化为字符串
        result_df['DateTime'] = result_df['DateTime_ET'].dt.strftime('%Y-%m-%d %H:%M:%S')
        result_df = result_df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # 按时间排序
        result_df = result_df.sort_values('DateTime').reset_index(drop=True)
        
        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = f"{base_name}_et_trading_hours.csv"
        
        # 保存结果
        print(f"正在保存到: {output_file}")
        result_df.to_csv(output_file, index=False)
        
        # 统计信息
        print("\n" + "="*60)
        print("转换完成！")
        print("="*60)
        print(f"📁 输入文件: {input_file}")
        print(f"💾 输出文件: {output_file}")
        print(f"📊 原始记录: {len(df):,} 条")
        print(f"📈 交易时间记录: {len(result_df):,} 条")
        print(f"📅 时间范围: {result_df['DateTime'].iloc[0]} 到 {result_df['DateTime'].iloc[-1]}")
        print(f"📁 文件大小: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        # 显示样本数据
        print(f"\n前5条记录:")
        print(result_df.head().to_string(index=False))
        
        return output_file
    
    def analyze_trading_hours_distribution(self, input_file: str):
        """
        分析交易时间分布
        
        Args:
            input_file: 输入文件路径
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")
        
        print(f"正在分析文件: {input_file}")
        
        # 读取并转换数据（只读取前1000行用于分析）
        df = pd.read_csv(input_file, nrows=10000)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['DateTime_UTC'] = df['DateTime'].dt.tz_localize(self.utc_tz)
        df['DateTime_ET'] = df['DateTime_UTC'].dt.tz_convert(self.et_tz)
        df['Hour_ET'] = df['DateTime_ET'].dt.hour
        df['Weekday'] = df['DateTime_ET'].dt.weekday
        
        print("\n" + "="*50)
        print("数据分布分析 (基于前10,000条记录)")
        print("="*50)
        
        # 按小时分布
        print("\n美东时间小时分布:")
        hour_dist = df['Hour_ET'].value_counts().sort_index()
        for hour, count in hour_dist.items():
            marker = "🟢" if 9 <= hour <= 16 else "🔴"
            print(f"{marker} {hour:02d}:00 - {count:,} 条记录")
        
        # 按工作日分布
        print(f"\n工作日分布:")
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        weekday_dist = df['Weekday'].value_counts().sort_index()
        for day, count in weekday_dist.items():
            marker = "🟢" if day <= 4 else "🔴"
            print(f"{marker} {weekday_names[day]} - {count:,} 条记录")
        
        # 交易时间统计
        weekday_filter = df['Weekday'] <= 4
        time_filter = (df['Hour_ET'] >= 9) & (df['Hour_ET'] <= 16)
        trading_hours_count = len(df[weekday_filter & time_filter])
        
        print(f"\n📊 统计摘要:")
        print(f"总记录数: {len(df):,}")
        print(f"工作日记录: {len(df[weekday_filter]):,}")
        print(f"交易时间记录: {trading_hours_count:,}")
        print(f"交易时间占比: {trading_hours_count/len(df)*100:.1f}%")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BTC数据时区转换和交易时间过滤')
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('-o', '--output', help='输出CSV文件路径（可选）')
    parser.add_argument('-a', '--analyze', action='store_true', help='分析数据分布（不执行转换）')
    
    args = parser.parse_args()
    
    try:
        converter = BTCETTradingHoursConverter()
        
        if args.analyze:
            # 只分析数据分布
            converter.analyze_trading_hours_distribution(args.input_file)
        else:
            # 执行转换
            output_file = converter.convert_timezone_and_filter(args.input_file, args.output)
            print(f"\n✅ 转换成功完成！")
            print(f"输出文件: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ 文件错误: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ 数据错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n用户取消操作")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认文件
    if len(sys.argv) == 1:
        print("使用默认输入文件...")
        default_input = "btc_monthly_data/btc_1m_202401_202505_monthly_merged.csv"
        if os.path.exists(default_input):
            converter = BTCETTradingHoursConverter()
            output_file = converter.convert_timezone_and_filter(default_input)
            print(f"\n✅ 转换成功完成！")
            print(f"输出文件: {output_file}")
        else:
            print(f"❌ 默认文件不存在: {default_input}")
            print("请提供输入文件路径作为参数")
            sys.exit(1)
    else:
        main() 