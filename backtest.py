import pandas as pd # pyright: ignore
import numpy as np# pyright: ignore
from math import floor
import matplotlib.pyplot as plt# pyright: ignore
from datetime import datetime, time, timedelta, date
import random
import os
from plot_trading_day import plot_trading_day# pyright: ignore



def simulate_day(day_df, prev_close, allowed_times, position_size, config):
    """
    模拟单日交易，使用噪声空间策略
    
    参数:
        day_df: 包含日内数据的DataFrame
        prev_close: 前一日收盘价
        allowed_times: 允许交易的时间列表
        position_size: 仓位大小
        config: 配置字典，包含所有交易参数
    """
    # 从配置中提取参数
    transaction_fee_per_share = config.get('transaction_fee_per_share', 0.01)
    trading_end_time = config.get('trading_end_time', (15, 50))
    max_positions_per_day = config.get('max_positions_per_day', float('inf'))
    print_details = config.get('print_trade_details', False)
    debug_time = config.get('debug_time', None)
    position = 0  # 0: 无仓位, 1: 多头, -1: 空头
    entry_price = np.nan
    trailing_stop = np.nan
    trade_entry_time = None
    trades = []
    positions_opened_today = 0  # 今日开仓计数器
    
    # 调试时间点标记，确保只打印一次
    debug_printed = False
    
    for idx, row in day_df.iterrows():
        current_time = row['Time']
        price = row['Close']
        upper = row['UpperBound']
        lower = row['LowerBound']
        sigma = row.get('sigma', 0)
        
        # 在允许时间内的入场信号
        if position == 0 and current_time in allowed_times and positions_opened_today < max_positions_per_day:
            # 检查潜在多头入场
            if price > upper:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 多头入场:")
                    print(f"  价格: {price:.2f} > 上边界: {upper:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower:.2f}")
                
                # 允许多头入场
                position = 1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 初始止损设为上边界
                trailing_stop = upper
                    
            # 检查潜在空头入场
            if price < lower:
                # 打印边界计算详情（如果需要）
                if print_details:
                    date_str = row['DateTime'].strftime('%Y-%m-%d')
                    sigma = row.get('sigma', 0)
                    upper_ref = row.get('upper_ref', 0)
                    lower_ref = row.get('lower_ref', 0)
                    day_open = row.get('day_open', 0)
                    
                    print(f"\n交易点位详情 [{date_str} {current_time}] - 空头入场:")
                    print(f"  价格: {price:.2f} < 下边界: {lower:.2f}")
                    print(f"  边界计算详情:")
                    print(f"    - 日开盘价: {day_open:.2f}, 前日收盘价: {prev_close:.2f}")
                    print(f"    - 上边界参考价: max({day_open:.2f}, {prev_close:.2f}) = {upper_ref:.2f}")
                    print(f"    - 下边界参考价: min({day_open:.2f}, {prev_close:.2f}) = {lower_ref:.2f}")
                    print(f"    - Sigma值: {sigma:.6f}")
                    print(f"    - 上边界计算: {upper_ref:.2f} * (1 + {sigma:.6f}) = {upper:.2f}")
                    print(f"    - 下边界计算: {lower_ref:.2f} * (1 - {sigma:.6f}) = {lower:.2f}")
                
                # 允许空头入场
                position = -1
                entry_price = price
                trade_entry_time = row['DateTime']
                positions_opened_today += 1  # 增加开仓计数器
                # 初始止损设为下边界
                trailing_stop = lower
        
        # 更新止损并检查出场信号
        if position != 0:
            if position == 1:  # 多头仓位
                # 计算当前时刻的止损水平（使用上边界）
                trailing_stop = upper
                
                # 如果价格跌破当前止损，则平仓
                exit_condition = price < trailing_stop
                
                # 检查是否出场
                if exit_condition and current_time in allowed_times:
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 多头出场:")
                        print(f"  价格: {price:.2f} < 当前止损: {trailing_stop:.2f}")
                        print(f"  止损计算: 上边界={upper:.2f}")
                        print(f"  买入价: {entry_price:.2f}, 卖出价: {price:.2f}, 股数: {position_size}")
                    
                    # 平仓多头
                    exit_time = row['DateTime']
                    # 计算交易费用（开仓和平仓）
                    transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
                    pnl = position_size * (price - entry_price) - transaction_fees
                    
                    exit_reason = 'Stop Loss'
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Long',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'transaction_fees': transaction_fees
                    })
                    
                    position = 0
                    trailing_stop = np.nan
                    
            elif position == -1:  # 空头仓位
                # 计算当前时刻的止损水平（使用下边界）
                trailing_stop = lower
                
                # 如果价格涨破当前止损，则平仓
                exit_condition = price > trailing_stop
                
                # 检查是否出场
                if exit_condition and current_time in allowed_times:
                    # 打印出场详情（如果需要）
                    if print_details:
                        date_str = row['DateTime'].strftime('%Y-%m-%d')
                        print(f"\n交易点位详情 [{date_str} {current_time}] - 空头出场:")
                        print(f"  价格: {price:.2f} > 当前止损: {trailing_stop:.2f}")
                        print(f"  止损计算: 下边界={lower:.2f}")
                        print(f"  卖出价: {entry_price:.2f}, 买入价: {price:.2f}, 股数: {position_size}")
                    
                    # 平仓空头
                    exit_time = row['DateTime']
                    # 计算交易费用（开仓和平仓）
                    transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
                    pnl = position_size * (entry_price - price) - transaction_fees
                    
                    exit_reason = 'Stop Loss'
                    trades.append({
                        'entry_time': trade_entry_time,
                        'exit_time': exit_time,
                        'side': 'Short',
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'position_size': position_size,
                        'transaction_fees': transaction_fees
                    })
                    
                    position = 0
                    trailing_stop = np.nan
    
    # 获取交易结束时间字符串，格式为HH:MM
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    
    # 寻找结束时间的数据点（如果存在）
    close_time_rows = day_df[day_df['Time'] == end_time_str]
    
    # 如果有结束时间的数据点且仍有未平仓位，则平仓
    if not close_time_rows.empty and position != 0:
        close_row = close_time_rows.iloc[0]
        exit_time = close_row['DateTime']
        close_price = close_row['Close']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 多头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            pnl = position_size * (close_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees
            })
            
            position = 0
            trailing_stop = np.nan
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {end_time_str}] - 空头收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {close_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            pnl = position_size * (entry_price - close_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': close_price,
                'pnl': pnl,
                'exit_reason': 'Intraday Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees
            })
            
            position = 0
            trailing_stop = np.nan
    
    # 如果仍有未平仓位且没有结束时间数据点，则在一天结束时平仓
    elif position != 0:
        exit_time = day_df.iloc[-1]['DateTime']
        last_price = day_df.iloc[-1]['Close']
        last_time = day_df.iloc[-1]['Time']
        
        if position == 1:  # 多头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 多头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            pnl = position_size * (last_price - entry_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Long',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees
            })
                
        else:  # 空头仓位
            # 打印出场详情（如果需要）
            if print_details:
                date_str = exit_time.strftime('%Y-%m-%d')
                print(f"\n交易点位详情 [{date_str} {last_time}] - 空头市场收盘平仓:")
                print(f"  入场价: {entry_price:.2f}, 出场价: {last_price:.2f}, 股数: {position_size}")
            
            # 计算交易费用（开仓和平仓）
            transaction_fees = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            pnl = position_size * (entry_price - last_price) - transaction_fees
            trades.append({
                'entry_time': trade_entry_time,
                'exit_time': exit_time,
                'side': 'Short',
                'entry_price': entry_price,
                'exit_price': last_price,
                'pnl': pnl,
                'exit_reason': 'Market Close',
                'position_size': position_size,
                'transaction_fees': transaction_fees
            })
    
    return trades 

def run_backtest(config):
    """
    运行回测 - 噪声空间策略
    
    参数:
        config: 配置字典，包含所有回测参数
        
    返回:
        日度结果DataFrame
        月度结果DataFrame
        交易记录DataFrame
        性能指标字典
    """
    # 从配置中提取参数
    data_path = config.get('data_path')
    ticker = config.get('ticker')
    initial_capital = config.get('initial_capital', 100000)
    lookback_days = config.get('lookback_days', 90)
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    plot_days = config.get('plot_days')
    random_plots = config.get('random_plots', 0)
    plots_dir = config.get('plots_dir', 'trading_plots')
    check_interval_minutes = config.get('check_interval_minutes', 30)
    transaction_fee_per_share = config.get('transaction_fee_per_share', 0.01)
    trading_start_time = config.get('trading_start_time', (10, 00))
    trading_end_time = config.get('trading_end_time', (15, 40))
    max_positions_per_day = config.get('max_positions_per_day', float('inf'))
    print_daily_trades = config.get('print_daily_trades', True)
    print_trade_details = config.get('print_trade_details', False)
    debug_time = config.get('debug_time')
    leverage = config.get('leverage', 1)  # 资金杠杆倍数，默认为1
    # 如果未提供ticker，从文件名中提取
    if ticker is None:
        # 从文件名中提取ticker
        file_name = os.path.basename(data_path)
        # 移除_market_hours.csv（如果存在）
        ticker = file_name.replace('_market_hours.csv', '')
    
    # 加载和处理数据
    print(f"加载{ticker}数据，从{data_path}...")
    price_df = pd.read_csv(data_path, parse_dates=['DateTime'])
    price_df.sort_values('DateTime', inplace=True)
    
    # 提取日期和时间组件
    price_df['Date'] = price_df['DateTime'].dt.date
    price_df['Time'] = price_df['DateTime'].dt.strftime('%H:%M')
    
    # 如果没有Volume列，添加默认Volume列
    if 'Volume' not in price_df.columns:
        price_df['Volume'] = 1.0  # 为BTC数据添加默认成交量
    
    # 按日期范围过滤数据（如果指定）
    if start_date is not None:
        price_df = price_df[price_df['Date'] >= start_date]
        print(f"筛选数据，开始日期为{start_date}")
    
    if end_date is not None:
        price_df = price_df[price_df['Date'] <= end_date]
        print(f"筛选数据，结束日期为{end_date}")
    
    # 检查DayOpen和DayClose列是否存在，如果不存在则创建
    if 'DayOpen' not in price_df.columns or 'DayClose' not in price_df.columns:
        # 对于BTC数据，使用特定时间点
        # 今天开盘价：取21:30分K线的开盘价
        opening_prices = []
        closing_prices = []
        
        for date in price_df['Date'].unique():
            day_data = price_df[price_df['Date'] == date]
            
            # 寻找21:30的开盘价
            open_time_data = day_data[day_data['Time'] == '21:30']
            if not open_time_data.empty:
                day_open = open_time_data.iloc[0]['Open']
            else:
                # 如果没有21:30的数据，使用当天第一个数据点的开盘价
                day_open = day_data.iloc[0]['Open']
            
            opening_prices.append({'Date': date, 'DayOpen': day_open})
            
            # 昨日收盘价：取昨天16:00分K线的收盘价
            # 先找前一天的数据
            prev_date = pd.to_datetime(date) - pd.Timedelta(days=1)
            prev_date = prev_date.date()
            prev_day_data = price_df[price_df['Date'] == prev_date]
            
            if not prev_day_data.empty:
                # 寻找16:00的收盘价
                close_time_data = prev_day_data[prev_day_data['Time'] == '16:00']
                if not close_time_data.empty:
                    day_close = close_time_data.iloc[0]['Close']
                else:
                    # 如果没有16:00的数据，使用前一天最后一个数据点的收盘价
                    day_close = prev_day_data.iloc[-1]['Close']
            else:
                # 如果没有前一天数据，使用当天开盘价
                day_close = day_open
            
            closing_prices.append({'Date': date, 'DayClose': day_close})
        
        # 转换为DataFrame并合并
        opening_prices_df = pd.DataFrame(opening_prices)
        closing_prices_df = pd.DataFrame(closing_prices)
        
        # 将开盘价和收盘价合并回主DataFrame
        price_df = pd.merge(price_df, opening_prices_df, on='Date', how='left')
        price_df = pd.merge(price_df, closing_prices_df, on='Date', how='left')
    
    # 使用筛选后数据的DayOpen和DayClose
    # 这些代表9:30 AM开盘价和4:00 PM收盘价
    price_df['prev_close'] = price_df.groupby('Date')['DayClose'].transform('first').shift(1)
    
    # 使用9:30 AM价格作为当天的开盘价
    price_df['day_open'] = price_df.groupby('Date')['DayOpen'].transform('first')
    
    # 为每个交易日计算一次参考价格，并将其应用于该日的所有时间点
    # 这确保了整个交易日使用相同的参考价格
    unique_dates = price_df['Date'].unique()
    
    # 创建临时DataFrame来存储每个日期的参考价格
    date_refs = []
    for d in unique_dates:
        day_data = price_df[price_df['Date'] == d].iloc[0]  # 获取该日第一行数据
        day_open = day_data['day_open']
        prev_close = day_data['prev_close']
        
        # 计算该日的参考价格
        if not pd.isna(prev_close):
            upper_ref = max(day_open, prev_close)
            lower_ref = min(day_open, prev_close)
        else:
            upper_ref = day_open
            lower_ref = day_open
            
        date_refs.append({
            'Date': d,
            'upper_ref': upper_ref,
            'lower_ref': lower_ref
        })
    
    # 创建日期参考价格DataFrame
    date_refs_df = pd.DataFrame(date_refs)
    
    # 将参考价格合并回主DataFrame
    price_df = price_df.drop(columns=['upper_ref', 'lower_ref'], errors='ignore')
    price_df = pd.merge(price_df, date_refs_df, on='Date', how='left')
    
    # 计算每分钟相对开盘的回报（使用day_open保持一致性）
    price_df['ret'] = price_df['Close'] / price_df['day_open'] - 1 

    # 计算噪声区域边界
    print(f"计算噪声区域边界...")
    # 将时间点转为列
    pivot = price_df.pivot(index='Date', columns='Time', values='ret').abs()
    # 计算每个时间点的绝对回报的滚动平均值
    # 这确保我们对每个时间点使用前lookback_days天的数据
    sigma = pivot.rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1)
    # 转回长格式
    sigma = sigma.stack().reset_index(name='sigma')
    
    # 保存一个原始数据的副本，用于计算买入持有策略
    price_df_original = price_df.copy()
    
    # 将sigma合并回主DataFrame
    price_df = pd.merge(price_df, sigma, on=['Date', 'Time'], how='left')
    
    # 检查每个交易日是否有足够的sigma数据
    # 创建一个标记，记录哪些日期的sigma数据不完整
    incomplete_sigma_dates = set()
    print(f"检查{len(price_df['Date'].unique())}个日期的sigma数据...")
    print(f"Pivot表形状: {pivot.shape}")
    print(f"Sigma表形状: {sigma.shape}")
    print(f"Sigma中NaN的数量: {sigma['sigma'].isna().sum()}")
    
    for date in price_df['Date'].unique():
        day_data = price_df[price_df['Date'] == date]
        na_count = day_data['sigma'].isna().sum()
        total_count = len(day_data)
        if na_count > 0:
            print(f"日期 {date} 有 {na_count}/{total_count} 个缺失的sigma值")
            incomplete_sigma_dates.add(date)
        else:
            print(f"日期 {date} sigma数据完整 ({total_count}条)")
    
    # 移除sigma数据不完整的日期
    price_df = price_df[~price_df['Date'].isin(incomplete_sigma_dates)]
    
    print(f"处理后剩余数据: {len(price_df)} 条")
    print(f"剩余日期数: {len(price_df['Date'].unique())}")
    
    # 确保所有剩余的sigma值都有有效数据
    if price_df['sigma'].isna().any():
        print(f"警告: 仍有{price_df['sigma'].isna().sum()}个缺失的sigma值")
    
    # 使用正确的参考价格计算噪声区域的上下边界
    # 从配置中获取K1和K2参数
    K1 = config.get('K1', 1)  # 如果未设置，默认为1
    K2 = config.get('K2', 1)  # 如果未设置，默认为1
    
    print(f"使用上边界乘数K1={K1}，下边界乘数K2={K2}")
    
    # 将K1和K2应用于sigma进行边界计算
    price_df['UpperBound'] = price_df['upper_ref'] * (1 + K1 * price_df['sigma'])
    price_df['LowerBound'] = price_df['lower_ref'] * (1 - K2 * price_df['sigma'])
    
    # 根据检查间隔生成允许的交易时间
    allowed_times = []
    start_hour, start_minute = trading_start_time  # 使用可配置的开始时间
    end_hour, end_minute = trading_end_time        # 使用可配置的结束时间
    
    current_hour, current_minute = start_hour, start_minute
    while current_hour < end_hour or (current_hour == end_hour and current_minute <= end_minute):
        # 将当前时间添加到allowed_times
        allowed_times.append(f"{current_hour:02d}:{current_minute:02d}")
        
        # 增加check_interval_minutes
        current_minute += check_interval_minutes
        if current_minute >= 60:
            current_hour += current_minute // 60
            current_minute = current_minute % 60
    
    # 始终确保trading_end_time包含在内，用于平仓
    end_time_str = f"{trading_end_time[0]:02d}:{trading_end_time[1]:02d}"
    if end_time_str not in allowed_times:
        allowed_times.append(end_time_str)
        allowed_times.sort()
    
    print(f"使用{check_interval_minutes}分钟的检查间隔")
    
    # 初始化回测变量
    capital = initial_capital
    daily_results = []
    all_trades = []
    total_transaction_fees = 0  # 跟踪总交易费用
    
    # 添加交易日期统计变量
    trading_days = set()       # 有交易的日期集合
    non_trading_days = set()   # 无交易的日期集合
    
    # 如果指定了随机生成图表的数量，随机选择交易日
    days_with_trades = []
    if random_plots > 0:
        # 先运行回测，记录有交易的日期
        for trade_date in unique_dates:
            day_data = price_df[price_df['Date'] == trade_date].copy()
            if len(day_data) < 10:  # 跳过数据不足的日期
                continue
                
            prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
            if prev_close is None:
                continue
                
            # 模拟当天交易
            simulation_result = simulate_day(day_data, prev_close, allowed_times, 100, config)
            
            # 从结果中提取交易
            trades = simulation_result
                
            if trades:  # 如果有交易
                days_with_trades.append(trade_date)
        
        # 如果有交易的日期少于请求的随机图表数量，调整随机图表数量
        random_plots = min(random_plots, len(days_with_trades))
        # 随机选择日期
        if random_plots > 0:
            random_plot_days = random.sample(days_with_trades, random_plots)
        else:
            random_plot_days = []
    else:
        random_plot_days = []
    
    # 合并指定的绘图日期和随机选择的日期
    if plot_days is None:
        plot_days = []
    all_plot_days = list(set(plot_days + random_plot_days))
    
    # 确保绘图目录存在
    if all_plot_days and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
    
    # 创建买入持有回测数据（使用原始数据，不受sigma筛选影响）
    buy_hold_data = []
    filtered_dates = price_df['Date'].unique()  # 策略交易使用的日期（经过sigma筛选）
    
    # 创建独立的买入持有数据，使用原始数据（未经过sigma筛选）
    for trade_date in unique_dates:
        # 获取当天的数据（从原始数据中）
        day_data = price_df_original[price_df_original['Date'] == trade_date].copy()
        
        # 跳过数据不足的日期
        if len(day_data) < 10:  # 任意阈值
            continue
        
        # 获取当天的开盘价和收盘价（用于计算买入持有）
        open_price = day_data['day_open'].iloc[0]
        close_price = day_data['DayClose'].iloc[0]
        
        # 存储买入持有数据
        buy_hold_data.append({
            'Date': trade_date,
            'Open': open_price,
            'Close': close_price
        })
    
    # 处理策略交易部分
    for i, trade_date in enumerate(filtered_dates):
        # 获取当天的数据
        day_data = price_df[price_df['Date'] == trade_date].copy()
        day_data = day_data.sort_values('DateTime').reset_index(drop=True)
        
        # 跳过数据不足的日期
        if len(day_data) < 10:  # 任意阈值
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
        
        # 获取前一天的收盘价
        prev_close = day_data['prev_close'].iloc[0] if not pd.isna(day_data['prev_close'].iloc[0]) else None
        
        # 将trade_date转换为字符串格式以便统一显示
        date_str = pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        
        # 获取当天的开盘价
        day_open_price = day_data['day_open'].iloc[0]
        
        # 计算仓位大小（应用杠杆）
        leveraged_capital = capital * leverage  # 应用杠杆倍数
        position_size = floor(leveraged_capital / day_open_price)
        
        # 如果资金不足，跳过当天
        if position_size <= 0:
            daily_results.append({
                'Date': trade_date,
                'capital': capital,
                'daily_return': 0
            })
            continue
                
        # 模拟当天的交易
        simulation_result = simulate_day(day_data, prev_close, allowed_times, position_size, config)
        
        # 从结果中提取交易
        trades = simulation_result
        
        # 更新交易日期统计
        if trades:  # 有交易的日期
            trading_days.add(trade_date)
        else:  # 无交易的日期
            non_trading_days.add(trade_date)
        
        # 打印每天的交易信息
        if trades and print_daily_trades:
            # 计算当天总盈亏
            day_total_pnl = sum(trade['pnl'] for trade in trades)
            
            # 创建交易方向与时间的简要信息
            trade_summary = []
            for trade in trades:
                direction = "多" if trade['side'] == 'Long' else "空"
                entry_time = trade['entry_time'].strftime('%H:%M')
                exit_time = trade['exit_time'].strftime('%H:%M')
                pnl = trade['pnl']
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                size = trade.get('position_size', position_size)
                trade_summary.append(f"{direction}({entry_time}->{exit_time}) 买:{entry_price:.2f} 卖:{exit_price:.2f} 股数:{size} 盈亏:${pnl:.2f}")
            
            # 打印单行交易日志
            trade_info = ", ".join(trade_summary)
            leverage_info = f" [杠杆{leverage}x]" if leverage != 1 else ""
            print(f"{date_str} | 交易数: {len(trades)} | 总盈亏: ${day_total_pnl:.2f}{leverage_info} | {trade_info}")
        
        # 检查是否需要为这一天生成图表
        if trade_date in all_plot_days:
            # 为当天的交易生成图表
            plot_path = os.path.join(plots_dir, f"{ticker}_trade_visualization_{trade_date}")
            
            # 添加交易类型到文件名
            sides = [trade['side'] for trade in trades]
            if 'Long' in sides and 'Short' not in sides:
                plot_path += "_Long.png"
            elif 'Short' in sides and 'Long' not in sides:
                plot_path += "_Short.png"
            elif 'Long' in sides and 'Short' in sides:
                plot_path += "_Mixed.png"
            else:
                plot_path += ".png"  # 没有交易
                
            # 生成并保存图表
            plot_trading_day(day_data, trades, save_path=plot_path)
        
        # 计算每日盈亏和交易费用
        day_pnl = 0
        day_transaction_fees = 0
        for trade in trades:
            day_pnl += trade['pnl']
            # 从每笔交易中提取交易费用
            if 'transaction_fees' not in trade:
                # 如果交易数据中没有交易费用，则计算
                trade['transaction_fees'] = max(position_size * transaction_fee_per_share * 2, 2.16)  # 买入和卖出费用，最低2.16
            day_transaction_fees += trade['transaction_fees']
        
        # 添加到总交易费用
        total_transaction_fees += day_transaction_fees
        
        # 更新资金并计算每日回报
        capital_start = capital
        capital += day_pnl
        daily_return = day_pnl / capital_start
        
        # 存储每日结果
        daily_results.append({
            'Date': trade_date,
            'capital': capital,
            'daily_return': daily_return
        })
        
        # 存储交易
        for trade in trades:
            trade['Date'] = trade_date
            all_trades.append(trade)
    
    # 创建每日结果DataFrame
    daily_df = pd.DataFrame(daily_results)
    if len(daily_df) > 0:
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df.set_index('Date', inplace=True)
    else:
        print("警告: 没有有效的交易日数据")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    
    # 创建买入持有DataFrame
    buy_hold_df = pd.DataFrame(buy_hold_data)
    
    # 检查buy_hold_data是否为空
    if not buy_hold_data:
        print("警告: 没有足够的数据来计算买入持有策略的表现")
        buy_hold_df = pd.DataFrame()  # 创建一个空的DataFrame
    else:
        buy_hold_df['Date'] = pd.to_datetime(buy_hold_df['Date'])
        buy_hold_df.set_index('Date', inplace=True)
    
    # 计算买入持有策略的表现
    if not buy_hold_df.empty:
        # 计算每日收益率
        buy_hold_df['daily_return'] = buy_hold_df['Close'] / buy_hold_df['Close'].shift(1) - 1
        
        # 计算累积资本
        buy_hold_df['capital'] = initial_capital * (1 + buy_hold_df['daily_return']).cumprod().fillna(1)
    
    # 计算月度回报
    monthly = daily_df.resample('ME').first()[['capital']].rename(columns={'capital': 'month_start'})
    monthly['month_end'] = daily_df.resample('ME').last()['capital']
    monthly['monthly_return'] = monthly['month_end'] / monthly['month_start'] - 1
    
    # 打印月度回报
    print("\n月度回报:")
    # 设置pandas显示选项以显示所有行
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # 创建格式化的月度回报显示
    monthly_display = monthly[['month_start', 'month_end', 'monthly_return']].copy()
    monthly_display['monthly_return_pct'] = monthly_display['monthly_return'] * 100
    monthly_display = monthly_display.round({'month_start': 2, 'month_end': 2, 'monthly_return_pct': 2})
    
    print(monthly_display[['month_start', 'month_end', 'monthly_return_pct']].rename(columns={
        'month_start': '月初资金',
        'month_end': '月末资金', 
        'monthly_return_pct': '月度收益率(%)'
    }))
    
    # 打印月度回报统计信息
    monthly_returns = monthly['monthly_return'].dropna()
    if len(monthly_returns) > 0:
        print(f"\n月度回报统计:")
        print(f"  平均月度收益率: {monthly_returns.mean()*100:.2f}%")
        print(f"  月度收益率标准差: {monthly_returns.std()*100:.2f}%")
        print(f"  最佳月度收益率: {monthly_returns.max()*100:.2f}%")
        print(f"  最差月度收益率: {monthly_returns.min()*100:.2f}%")
        print(f"  正收益月份: {(monthly_returns > 0).sum()}个")
        print(f"  负收益月份: {(monthly_returns < 0).sum()}个")
        print(f"  胜率: {(monthly_returns > 0).mean()*100:.1f}%")
    
    # 恢复默认显示设置
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.max_colwidth')
    
    # 计算总体表现
    total_return = capital / initial_capital - 1
    print(f"\n总回报: {total_return*100:.2f}%")
    
    # 创建交易DataFrame
    trades_df = pd.DataFrame(all_trades)
    
    # 计算策略性能指标
    print(f"\n计算策略性能指标...")
    metrics = calculate_performance_metrics(daily_df, trades_df, initial_capital, buy_hold_df=buy_hold_df)
    
    # 打印交易费用统计
    print(f"\n交易费用统计:")
    print(f"总交易费用: ${total_transaction_fees:.2f}")
    if len(trades_df) > 0:
        print(f"平均每笔交易费用: ${total_transaction_fees / len(trades_df):.2f}")
    if len(daily_df) > 0:
        print(f"平均每日交易费用: ${total_transaction_fees / len(daily_df):.2f}")
    print(f"交易费用占初始资金比例: {total_transaction_fees / initial_capital * 100:.2f}%")
    print(f"交易费用占总收益比例: {total_transaction_fees / (capital - initial_capital) * 100:.2f}%" if capital > initial_capital else "交易费用占总收益比例: N/A (无盈利)")
    
    # 打印交易日期统计
    print(f"\n交易日期统计:")
    print(f"总交易日数: {len(trading_days) + len(non_trading_days)}")
    print(f"有交易的天数: {len(trading_days)} ({len(trading_days)/(len(trading_days) + len(non_trading_days))*100:.1f}%)")
    print(f"无交易的天数: {len(non_trading_days)} ({len(non_trading_days)/(len(trading_days) + len(non_trading_days))*100:.1f}%)")
    
    # 打印简化的性能指标
    print(f"\n策略性能指标:")
    leverage_text = f" (杠杆{leverage}x)" if leverage != 1 else ""
    strategy_name = f"{ticker} 噪声空间策略{leverage_text}"
    print(f"策略: {strategy_name}")
    
    # 创建表格格式对比策略与买入持有的指标
    print("\n性能指标对比:")
    print(f"{'指标':<20} | {'策略':<15} | {f'{ticker} Buy & Hold':<15}")
    print("-" * 55)
    
    # 总回报率
    print(f"{'总回报率':<20} | {metrics['total_return']*100:>14.1f}% | {metrics['buy_hold_return']*100:>14.1f}%")
    
    # 年化收益率
    print(f"{'年化收益率':<20} | {metrics['irr']*100:>14.1f}% | {metrics['buy_hold_irr']*100:>14.1f}%")
    
    # 波动率
    print(f"{'波动率':<20} | {metrics['volatility']*100:>14.1f}% | {metrics['buy_hold_volatility']*100:>14.1f}%")
    
    # 夏普比率
    print(f"{'夏普比率':<20} | {metrics['sharpe_ratio']:>14.2f} | {metrics['buy_hold_sharpe']:>14.2f}")
    
    # 最大回撤
    print(f"{'最大回撤':<20} | {metrics['mdd']*100:>14.1f}% | {metrics['buy_hold_mdd']*100:>14.1f}%")
    
    # 打印最大回撤的详细信息
    if 'max_drawdown_start_date' in metrics and 'max_drawdown_date' in metrics:
        start_date = metrics['max_drawdown_start_date'].strftime('%Y-%m-%d')
        bottom_date = metrics['max_drawdown_date'].strftime('%Y-%m-%d')
        
        print(f"\n最大回撤详细信息:")
        print(f"  峰值日期: {start_date}")
        print(f"  最低点日期: {bottom_date}")
        
        if metrics['max_drawdown_end_date'] is not None:
            end_date = metrics['max_drawdown_end_date'].strftime('%Y-%m-%d')
            print(f"  恢复日期: {end_date}")
            
            # 计算回撤持续时间
            duration = (metrics['max_drawdown_end_date'] - metrics['max_drawdown_start_date']).days
            print(f"  回撤持续时间: {duration}天")
        else:
            print(f"  恢复日期: 尚未恢复")
            
            # 计算到目前为止的回撤持续时间
            duration = (metrics['max_drawdown_date'] - metrics['max_drawdown_start_date']).days
            print(f"  回撤持续时间: {duration}天 (仍在回撤中)")
    
    # 策略特有指标
    print(f"\n策略特有指标:")
    print(f"胜率: {metrics['hit_ratio']*100:.1f}%")
    print(f"总交易次数: {metrics['total_trades']}")
    
    # 计算做多和做空的笔数
    if len(trades_df) > 0:
        long_trades = len(trades_df[trades_df['side'] == 'Long'])
        short_trades = len(trades_df[trades_df['side'] == 'Short'])
        print(f"做多交易笔数: {long_trades}")
        print(f"做空交易笔数: {short_trades}")
    else:
        print(f"做多交易笔数: 0")
        print(f"做空交易笔数: 0")
    
    print(f"平均每日交易次数: {metrics['avg_daily_trades']:.2f}")
    
    # 打印最大单笔收益和亏损统计
    print(f"\n单笔交易统计:")
    print(f"最大单笔收益: ${metrics.get('max_single_gain', 0):.2f}")
    print(f"最大单笔亏损: ${metrics.get('max_single_loss', 0):.2f}")
    
    # 打印前10笔最大收益
    if metrics.get('top_10_gains'):
        print(f"\n前10笔最大收益:")
        print(f"{'排名':<4} | {'日期':<12} | {'方向':<6} | {'买入价':<8} | {'卖出价':<8} | {'盈亏':<10} | {'退出原因':<15}")
        print("-" * 85)
        for i, trade in enumerate(metrics['top_10_gains'], 1):
            date_str = pd.to_datetime(trade['Date']).strftime('%Y-%m-%d')
            side = '多' if trade['side'] == 'Long' else '空'
            print(f"{i:<4} | {date_str:<12} | {side:<6} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f} | ${trade['pnl']:<9.2f} | {trade['exit_reason']:<15}")
    
    # 打印前10笔最大亏损
    if metrics.get('top_10_losses'):
        print(f"\n前10笔最大亏损:")
        print(f"{'排名':<4} | {'日期':<12} | {'方向':<6} | {'买入价':<8} | {'卖出价':<8} | {'盈亏':<10} | {'退出原因':<15}")
        print("-" * 85)
        for i, trade in enumerate(metrics['top_10_losses'], 1):
            date_str = pd.to_datetime(trade['Date']).strftime('%Y-%m-%d')
            side = '多' if trade['side'] == 'Long' else '空'
            print(f"{i:<4} | {date_str:<12} | {side:<6} | ${trade['entry_price']:<7.2f} | ${trade['exit_price']:<7.2f} | ${trade['pnl']:<9.2f} | {trade['exit_reason']:<15}")
    
    # 打印策略总结
    print(f"\n" + "="*50)
    print(f"策略回测总结 - {strategy_name}")
    print(f"="*50)
    
    # 打印杠杆信息
    if leverage != 1:
        print(f"💰 资金杠杆倍数: {leverage}x")
        print(f"💵 初始资金: ${initial_capital:,.0f}")
        print(f"💸 杠杆后可用资金: ${initial_capital * leverage:,.0f}")
        print(f"-"*50)
    
    # 核心表现指标
    print(f"📈 总回报率: {metrics['total_return']*100:.1f}%")
    print(f"📊 年化收益率: {metrics['irr']*100:.1f}%")
    print(f"⚡ 夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"📉 最大回撤: {metrics['mdd']*100:.1f}%")
    if 'max_drawdown_start_date' in metrics and 'max_drawdown_date' in metrics:
        start_date = metrics['max_drawdown_start_date'].strftime('%Y-%m-%d')
        bottom_date = metrics['max_drawdown_date'].strftime('%Y-%m-%d')
        print(f"   └─ 峰值: {start_date} → 最低点: {bottom_date}")
        
        if metrics['max_drawdown_end_date'] is not None:
            end_date = metrics['max_drawdown_end_date'].strftime('%Y-%m-%d')
            duration = (metrics['max_drawdown_end_date'] - metrics['max_drawdown_start_date']).days
            print(f"   └─ 恢复: {end_date} (持续{duration}天)")
        else:
            duration = (metrics['max_drawdown_date'] - metrics['max_drawdown_start_date']).days
            print(f"   └─ 尚未恢复 (已持续{duration}天)")
    print(f"🎯 胜率: {metrics['hit_ratio']*100:.1f}% | 总交易: {metrics['total_trades']}次")
    
    print(f"="*50)
    
    return daily_df, monthly, trades_df, metrics 

def calculate_performance_metrics(daily_df, trades_df, initial_capital, risk_free_rate=0.02, trading_days_per_year=252, buy_hold_df=None):
    """
    计算策略的性能指标
    
    参数:
        daily_df: 包含每日回测结果的DataFrame
        trades_df: 包含所有交易的DataFrame
        initial_capital: 初始资金
        risk_free_rate: 无风险利率，默认为2%
        trading_days_per_year: 一年的交易日数量，默认为252
        buy_hold_df: 买入持有策略的DataFrame
        
    返回:
        包含各种性能指标的字典
    """
    metrics = {}
    
    # 确保daily_df有数据
    if len(daily_df) == 0:
        print("警告: 没有足够的数据来计算性能指标")
        # 返回默认值
        return {
            'total_return': 0, 'irr': 0, 'volatility': 0, 'sharpe_ratio': 0,
            'hit_ratio': 0, 'mdd': 0, 'buy_hold_return': 0, 'buy_hold_irr': 0,
            'buy_hold_volatility': 0, 'buy_hold_sharpe': 0, 'buy_hold_mdd': 0
        }
    
    # 1. 总回报率 (Total Return)
    final_capital = daily_df['capital'].iloc[-1]
    metrics['total_return'] = final_capital / initial_capital - 1
    
    # 2. 年化收益率 (IRR - Internal Rate of Return)
    # 获取回测的开始和结束日期
    start_date = daily_df.index[0]
    end_date = daily_df.index[-1]
    # 计算实际年数（考虑实际日历日而不仅仅是交易日）
    years = (end_date - start_date).days / 365.25
    # 如果时间跨度太短，使用交易日计算
    if years < 0.1:  # 少于约36天
        trading_days = len(daily_df)
        years = trading_days / trading_days_per_year
    
    # 计算年化收益率 (CAGR - Compound Annual Growth Rate)
    if years > 0:
        metrics['irr'] = (1 + metrics['total_return']) ** (1 / years) - 1
    else:
        metrics['irr'] = 0
    
    # 3. 波动率 (Vol - Volatility)
    # 计算日收益率的标准差，然后年化
    daily_returns = daily_df['daily_return']
    # 移除异常值（如果有）
    daily_returns = daily_returns[daily_returns.between(daily_returns.quantile(0.001), 
                                                      daily_returns.quantile(0.999))]
    metrics['volatility'] = daily_returns.std() * np.sqrt(trading_days_per_year)
    
    # 4. 夏普比率 (Sharpe Ratio)
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['irr'] - risk_free_rate) / metrics['volatility']
    else:
        metrics['sharpe_ratio'] = 0
    
    # 5. 胜率 (Hit Ratio)和交易统计
    if len(trades_df) > 0:
        winning_trades = trades_df[trades_df['pnl'] > 0]
        metrics['hit_ratio'] = len(winning_trades) / len(trades_df)
        
        # 计算平均盈利和平均亏损
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        # 计算盈亏比
        metrics['profit_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 总交易次数
        metrics['total_trades'] = len(trades_df)
        
        # 计算每日交易次数
        daily_trade_counts = trades_df.groupby('Date').size()
        metrics['avg_daily_trades'] = daily_trade_counts.mean() if len(daily_trade_counts) > 0 else 0
        metrics['max_daily_trades'] = daily_trade_counts.max() if len(daily_trade_counts) > 0 else 0
        
        # 计算每日盈亏
        daily_pnl = trades_df.groupby('Date')['pnl'].sum()
        metrics['max_daily_loss'] = daily_pnl.min() if len(daily_pnl) > 0 and daily_pnl.min() < 0 else 0
        metrics['max_daily_gain'] = daily_pnl.max() if len(daily_pnl) > 0 else 0
        
        # 计算最大单笔收益和最大单笔亏损
        # 按盈亏排序，获取前10笔最大收益
        top_gains = trades_df.nlargest(10, 'pnl')[['Date', 'side', 'entry_price', 'exit_price', 'pnl', 'exit_reason']]
        metrics['top_10_gains'] = top_gains.to_dict('records')
        
        # 获取前10笔最大亏损
        top_losses = trades_df.nsmallest(10, 'pnl')[['Date', 'side', 'entry_price', 'exit_price', 'pnl', 'exit_reason']]
        metrics['top_10_losses'] = top_losses.to_dict('records')
        
        # 最大单笔收益和亏损
        metrics['max_single_gain'] = trades_df['pnl'].max()
        metrics['max_single_loss'] = trades_df['pnl'].min()
    else:
        metrics['hit_ratio'] = 0
        metrics['profit_loss_ratio'] = 0
        metrics['total_trades'] = 0
        metrics['avg_daily_trades'] = 0
        metrics['max_daily_trades'] = 0
        metrics['max_daily_loss'] = 0
        metrics['max_daily_gain'] = 0
        metrics['top_10_gains'] = []
        metrics['top_10_losses'] = []
        metrics['max_single_gain'] = 0
        metrics['max_single_loss'] = 0
    
    # 6. 最大回撤 (MDD - Maximum Drawdown)
    # 计算每日资金的累计最大值
    daily_df['peak'] = daily_df['capital'].cummax()
    # 计算每日回撤
    daily_df['drawdown'] = (daily_df['capital'] - daily_df['peak']) / daily_df['peak']
    # 最大回撤
    metrics['mdd'] = daily_df['drawdown'].min() * -1
    
    # 找到最大回撤发生的日期
    max_drawdown_date = daily_df['drawdown'].idxmin()
    metrics['max_drawdown_date'] = max_drawdown_date
    
    # 找到最大回撤开始的日期（即达到峰值的日期）
    max_drawdown_peak = daily_df.loc[max_drawdown_date, 'peak']
    # 找到达到这个峰值的最后一个日期
    peak_dates = daily_df[daily_df['capital'] == max_drawdown_peak].index
    max_drawdown_start_date = peak_dates[peak_dates <= max_drawdown_date].max()
    metrics['max_drawdown_start_date'] = max_drawdown_start_date
    
    # 找到最大回撤结束的日期（资金重新达到峰值的日期）
    recovery_dates = daily_df[daily_df['capital'] >= max_drawdown_peak].index
    recovery_dates_after = recovery_dates[recovery_dates > max_drawdown_date]
    if len(recovery_dates_after) > 0:
        max_drawdown_end_date = recovery_dates_after.min()
        metrics['max_drawdown_end_date'] = max_drawdown_end_date
    else:
        metrics['max_drawdown_end_date'] = None  # 尚未恢复
    
    # 计算回撤持续时间
    # 找到每个回撤开始的点
    drawdown_begins = (daily_df['peak'] != daily_df['peak'].shift(1)) & (daily_df['peak'] != daily_df['capital'])
    # 找到每个回撤结束的点（资金达到新高）
    drawdown_ends = daily_df['capital'] == daily_df['peak']
    
    # 计算最长回撤持续时间（交易日）
    if drawdown_begins.any() and drawdown_ends.any():
        begin_dates = daily_df.index[drawdown_begins]
        end_dates = daily_df.index[drawdown_ends]
        
        max_duration = 0
        for begin_date in begin_dates:
            # 找到这个回撤之后的第一个结束点
            end_date = end_dates[end_dates > begin_date]
            if len(end_date) > 0:
                duration = (end_date.min() - begin_date).days
                max_duration = max(max_duration, duration)
        
        metrics['max_drawdown_duration'] = max_duration
    else:
        metrics['max_drawdown_duration'] = 0
    
    # 计算Calmar比率 (年化收益率/最大回撤)
    if metrics['mdd'] > 0:
        metrics['calmar_ratio'] = metrics['irr'] / metrics['mdd']
    else:
        metrics['calmar_ratio'] = float('inf')  # 如果没有回撤，设为无穷大
        
    # 计算曝光时间 (Exposure Time)
    if len(trades_df) > 0:
        # 计算每笔交易的持仓时间（以分钟为单位）
        trades_df['duration'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.total_seconds() / 60
        
        # 计算总交易时间（分钟）
        total_trade_minutes = trades_df['duration'].sum()
        
        # 计算回测总时间（分钟）
        # 假设每个交易日有6.5小时（390分钟）
        trading_minutes_per_day = 390
        total_backtest_minutes = len(daily_df) * trading_minutes_per_day
        
        # 计算曝光时间百分比
        metrics['exposure_time'] = total_trade_minutes / total_backtest_minutes
    else:
        metrics['exposure_time'] = 0
    
    # 计算买入持有策略的表现
    if buy_hold_df is not None and not buy_hold_df.empty:
        # 计算买入持有策略的总回报率
        if 'capital' in buy_hold_df.columns:
            final_buy_hold_capital = buy_hold_df['capital'].iloc[-1]
            metrics['buy_hold_return'] = final_buy_hold_capital / initial_capital - 1
            
            # 计算买入持有策略的年化收益率
            if years > 0:
                metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
            else:
                metrics['buy_hold_irr'] = 0
            
            # 计算买入持有策略的波动率
            if 'daily_return' in buy_hold_df.columns:
                buy_hold_returns = buy_hold_df['daily_return'].dropna()
                # 移除异常值
                buy_hold_returns = buy_hold_returns[buy_hold_returns.between(
                    buy_hold_returns.quantile(0.001), buy_hold_returns.quantile(0.999))]
                metrics['buy_hold_volatility'] = buy_hold_returns.std() * np.sqrt(trading_days_per_year)
                
                # 计算买入持有策略的夏普比率
                if metrics['buy_hold_volatility'] > 0:
                    metrics['buy_hold_sharpe'] = (metrics['buy_hold_irr'] - risk_free_rate) / metrics['buy_hold_volatility']
                else:
                    metrics['buy_hold_sharpe'] = 0
            else:
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
            
            # 计算买入持有策略的最大回撤
            if 'capital' in buy_hold_df.columns:
                buy_hold_df['peak'] = buy_hold_df['capital'].cummax()
                buy_hold_df['drawdown'] = (buy_hold_df['capital'] - buy_hold_df['peak']) / buy_hold_df['peak']
                metrics['buy_hold_mdd'] = buy_hold_df['drawdown'].min() * -1
            else:
                metrics['buy_hold_mdd'] = 0
        else:
            # 如果buy_hold_df中没有capital列，则计算起始日期和结束日期的价格变化
            if 'Close' in buy_hold_df.columns:
                start_price = buy_hold_df['Close'].iloc[0]
                end_price = buy_hold_df['Close'].iloc[-1]
                metrics['buy_hold_return'] = end_price / start_price - 1
                
                # 计算买入持有策略的年化收益率
                if years > 0:
                    metrics['buy_hold_irr'] = (1 + metrics['buy_hold_return']) ** (1 / years) - 1
                else:
                    metrics['buy_hold_irr'] = 0
                
                # 其他指标设为0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
            else:
                # 没有足够的数据计算买入持有策略的表现
                metrics['buy_hold_return'] = 0
                metrics['buy_hold_irr'] = 0
                metrics['buy_hold_volatility'] = 0
                metrics['buy_hold_sharpe'] = 0
                metrics['buy_hold_mdd'] = 0
    else:
        # 没有买入持有的数据，设置默认值
        metrics['buy_hold_return'] = 0
        metrics['buy_hold_irr'] = 0
        metrics['buy_hold_volatility'] = 0
        metrics['buy_hold_sharpe'] = 0
        metrics['buy_hold_mdd'] = 0
    
    return metrics 

def plot_specific_days(config, dates_to_plot):
    """
    为指定的日期生成交易图表
    
    参数:
        config: 配置字典，包含所有回测参数
        dates_to_plot: 要绘制的日期列表 (datetime.date 对象列表)
    """
    # 创建配置的副本并更新plot_days
    plot_config = config.copy()
    plot_config['plot_days'] = dates_to_plot
    
    # 运行回测，指定要绘制的日期
    _, _, _, _ = run_backtest(plot_config)
    
    print(f"\n已为以下日期生成图表:")
    for d in dates_to_plot:
        print(f"- {d}")

# 示例用法
if __name__ == "__main__":  
    # 创建配置字典
    config = {
        'data_path': 'okx_btc_1m.csv',
        'ticker': 'BTC',
        'initial_capital': 100000,
        'lookback_days':1,
        'start_date': date(2025, 6, 1),
        'end_date': date(2025, 6, 29),
        'check_interval_minutes': 15 ,
        'transaction_fee_per_share': 0,
        # 'transaction_fee_per_share': 0.008166,


        'trading_start_time': (9, 40),
        'trading_end_time': (15, 45),
        'max_positions_per_day': 10,
        'random_plots': 2,
        'plots_dir': 'trading_plots',
        'print_daily_trades': True,
        'print_trade_details': False,
        'K1': 1,  # 上边界sigma乘数
        'K2': 1,  # 下边界sigma乘数
        'leverage': 3  # 资金杠杆倍数，默认为1
    }
    
    # 运行回测
    daily_results, monthly_results, trades, metrics = run_backtest(config)
