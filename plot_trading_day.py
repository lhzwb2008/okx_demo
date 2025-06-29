import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import os

def plot_trading_day(day_df, trades, save_path=None):
    """
    生成交易日的图表，显示价格、上下边界、VWAP、MACD和交易点
    
    参数:
        day_df: 包含单日数据的DataFrame
        trades: 当天交易列表
        save_path: 保存图表的路径（如果为None，则显示图表而不保存）
    """
    # 确保日期数据正确
    trade_date = day_df['Date'].iloc[0]
    
    # 计算VWAP
    prices = day_df['Close'].values
    volumes = day_df['Volume'].values
    cum_vol = 0
    cum_pv = 0
    vwaps = []
    
    for price, volume in zip(prices, volumes):
        cum_vol += volume
        cum_pv += price * volume
        
        if cum_vol > 0:
            vwap = cum_pv / cum_vol
        else:
            vwap = price
            
        vwaps.append(vwap)
    
    day_df['VWAP'] = vwaps
    
    # 检查是否有MACD数据
    has_macd = 'MACD_histogram' in day_df.columns
    
    # 创建图表 - 如果有MACD数据，创建两个子图
    if has_macd:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig = plt.figure(figsize=(14, 7))
        ax1 = fig.add_subplot(111)
    
    # 绘制价格线
    ax1.plot(day_df['Time'], day_df['Close'], label='Price', color='black', linewidth=1.5)
    
    # 绘制上下边界
    ax1.plot(day_df['Time'], day_df['UpperBound'], label='Upper Bound', color='green', linestyle='--', alpha=0.7)
    ax1.plot(day_df['Time'], day_df['LowerBound'], label='Lower Bound', color='red', linestyle='--', alpha=0.7)
    
    # 绘制VWAP
    ax1.plot(day_df['Time'], day_df['VWAP'], label='VWAP', color='blue', linestyle='-', alpha=0.7)
    
    # 标记交易点
    for trade in trades:
        # 获取入场和出场时间
        entry_time = trade['entry_time'].strftime('%H:%M')
        exit_time = trade['exit_time'].strftime('%H:%M')
        
        # 找到对应的数据点
        entry_idx = day_df[day_df['Time'] == entry_time].index
        exit_idx = day_df[day_df['Time'] == exit_time].index
        
        if len(entry_idx) > 0 and len(exit_idx) > 0:
            entry_idx = entry_idx[0]
            exit_idx = exit_idx[0]
            
            # 获取价格
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            
            # 标记入场点
            if trade['side'] == 'Long':
                ax1.scatter(entry_time, entry_price, color='green', s=100, marker='^', label='Long Entry' if 'Long Entry' not in ax1.get_legend_handles_labels()[1] else "")
            else:
                ax1.scatter(entry_time, entry_price, color='red', s=100, marker='v', label='Short Entry' if 'Short Entry' not in ax1.get_legend_handles_labels()[1] else "")
            
            # 标记出场点
            if trade['side'] == 'Long':
                ax1.scatter(exit_time, exit_price, color='red', s=100, marker='x', label='Long Exit' if 'Long Exit' not in ax1.get_legend_handles_labels()[1] else "")
            else:
                ax1.scatter(exit_time, exit_price, color='green', s=100, marker='x', label='Short Exit' if 'Short Exit' not in ax1.get_legend_handles_labels()[1] else "")
            
            # 连接入场和出场点
            ax1.plot([entry_time, exit_time], [entry_price, exit_price], color='gray', linestyle='-', alpha=0.5)
            
            # 添加P&L标签
            mid_time_idx = (day_df.index.get_loc(entry_idx) + day_df.index.get_loc(exit_idx)) // 2
            if mid_time_idx < len(day_df):
                mid_time = day_df['Time'].iloc[mid_time_idx]
                mid_price = (entry_price + exit_price) / 2
                ax1.annotate(f"P&L: ${trade['pnl']:.2f}", 
                             xy=(mid_time, mid_price),
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             fontsize=9,
                             bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    
    # 如果有MACD数据，绘制MACD子图
    if has_macd:
        # 在下方子图中绘制MACD柱状图
        bar_colors = ['green' if x > 0 else 'red' for x in day_df['MACD_histogram']]
        ax2.bar(day_df['Time'], day_df['MACD_histogram'], color=bar_colors, alpha=0.7, label='MACD Histogram')
        ax2.plot(day_df['Time'], day_df['MACD'], color='blue', linewidth=1, label='MACD')
        ax2.plot(day_df['Time'], day_df['MACD_signal'], color='red', linewidth=1, label='Signal')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 设置MACD子图的标签和图例
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('MACD', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        all_times = day_df['Time'].unique()
        time_ticks = [t for t in all_times if t.endswith(':00') or t.endswith(':30')]
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_ticks, rotation=45)
    
    # 设置图表标题和标签
    side_text = ""
    if trades:
        sides = [trade['side'] for trade in trades]
        if 'Long' in sides and 'Short' not in sides:
            side_text = "Long"
        elif 'Short' in sides and 'Long' not in sides:
            side_text = "Short"
        elif 'Long' in sides and 'Short' in sides:
            side_text = "Long/Short"
    
    ax1.set_title(f"Trading Day: {trade_date} ({side_text})", fontsize=14)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    
    # 设置x轴刻度
    # 每30分钟一个刻度
    all_times = day_df['Time'].unique()
    time_ticks = [t for t in all_times if t.endswith(':00') or t.endswith(':30')]
    ax1.set_xticks(time_ticks)
    ax1.set_xticklabels(time_ticks, rotation=45)
    
    # 添加网格
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    ax1.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图表
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"图表已保存至: {save_path}")
    else:
        plt.show()
