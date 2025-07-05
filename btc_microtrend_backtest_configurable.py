#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class BTCMicroTrendConfig:
    """配置类，包含所有可调参数"""
    
    def __init__(self):
        # === 基础配置 ===
        self.data_path = 'btc_1m.csv'
        self.initial_capital = 100000  # 初始资金（USDT）- 使用成功版本的参数
        
        # === 数据配置 ===
        self.data_limit = None  # 使用全部数据（设为None）或指定条数
        
        # === 时间范围配置 ===
        # 可以设置为None使用默认的80/20划分，或者设置具体日期
        self.train_start_date = '2024-01-01'    # 训练开始日期，格式: '2024-01-01'
        self.train_end_date = '2024-01-31'      # 训练结束日期，格式: '2024-02-01'  
        self.test_start_date = '2024-02-01'     # 测试开始日期，格式: '2024-02-01'
        self.test_end_date = '2025-05-30'       # 测试结束日期，格式: '2024-03-01' (一个月测试)
        self.use_date_split = True     # 是否使用日期分割（True）还是比例分割（False）
        
        # === 策略参数 ===
        self.lookback = 10      # 观察历史数据的分钟数
        self.predict_ahead = 10  # 预测未来多少分钟
        
        # === 交易参数 ===
        self.buy_threshold_percentile = 75   # 买入信号阈值（百分位数）
        self.sell_threshold_percentile = 25  # 卖出信号阈值（百分位数）
        
        # === 模型参数 ===
        self.n_estimators = 30               # 随机森林树的数量
        self.max_depth = 10                  # 最大深度
        self.min_samples_split = 20          # 内部节点再划分所需最小样本数
        self.min_samples_leaf = 10           # 叶子节点最少样本数
        self.random_state = 42               # 随机种子
        
        # === 输出配置 ===
        self.verbose = True                  # 是否显示详细日志
        self.print_trades = True             # 是否打印交易详情
        self.max_trades_to_print = 50        # 最多打印多少笔交易（None表示全部打印）
        self.print_daily_pnl = True          # 是否打印每日盈亏
        self.print_daily_stats = True        # 是否打印每日交易统计（交易次数、多空单）

class BTCMicroTrendBacktest:
    """BTC微趋势交易回测系统（可配置版，保持原有成功逻辑）"""
    
    def __init__(self, config=None):
        if config is None:
            config = BTCMicroTrendConfig()
        self.config = config
        self.data_path = config.data_path
        self.lookback = config.lookback
        self.predict_ahead = config.predict_ahead
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载数据"""
        if self.config.verbose:
            print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        if self.config.verbose:
            print(f"原始数据加载完成，共 {len(self.df)} 条记录")
        
        self.df['DateTime'] = pd.to_datetime(self.df['DateTime'])
        self.df.set_index('DateTime', inplace=True)
        
        # 根据配置决定使用多少数据
        if self.config.data_limit is not None:
            self.df = self.df.tail(self.config.data_limit)
            if self.config.verbose:
                print(f"数据预处理完成，使用最近 {len(self.df)} 条记录")
        else:
            if self.config.verbose:
                print(f"数据预处理完成，使用全部 {len(self.df)} 条记录")
        
    def calculate_features(self):
        """计算技术特征"""
        if self.config.verbose:
            print("正在计算技术特征...")
        
        # 价格变化率
        if self.config.verbose:
            print("  - 计算价格变化率...")
        self.df['price_change'] = self.df['Close'].pct_change()
        
        # 价格速度（1分钟、5分钟、10分钟）
        if self.config.verbose:
            print("  - 计算价格速度...")
        self.df['velocity_1m'] = self.df['Close'].diff(1) / self.df['Close'].shift(1)
        self.df['velocity_5m'] = self.df['Close'].diff(5) / self.df['Close'].shift(5)
        self.df['velocity_10m'] = self.df['Close'].diff(10) / self.df['Close'].shift(10)
        
        # 价格加速度
        if self.config.verbose:
            print("  - 计算价格加速度...")
        self.df['acceleration'] = self.df['velocity_1m'].diff(1)
        
        # 成交量相关特征
        if self.config.verbose:
            print("  - 计算成交量特征...")
        self.df['volume_ma5'] = self.df['Volume'].rolling(5).mean()
        self.df['volume_ratio'] = self.df['Volume'] / self.df['volume_ma5']
        
        # 价格动量
        if self.config.verbose:
            print("  - 计算价格动量...")
        self.df['momentum_5m'] = (self.df['Close'] - self.df['Close'].shift(5)) / self.df['Close'].shift(5)
        self.df['momentum_10m'] = (self.df['Close'] - self.df['Close'].shift(10)) / self.df['Close'].shift(10)
        
        # 价格振幅
        if self.config.verbose:
            print("  - 计算价格振幅...")
        self.df['amplitude'] = (self.df['High'] - self.df['Low']) / self.df['Close']
        
        # 简单移动平均
        if self.config.verbose:
            print("  - 计算移动平均...")
        self.df['sma_5'] = self.df['Close'].rolling(5).mean()
        self.df['sma_10'] = self.df['Close'].rolling(10).mean()
        self.df['price_to_sma5'] = self.df['Close'] / self.df['sma_5'] - 1
        self.df['price_to_sma10'] = self.df['Close'] / self.df['sma_10'] - 1
        
        # 目标变量：10分钟后的收益率
        if self.config.verbose:
            print("  - 计算目标变量...")
        self.df['target'] = self.df['Close'].shift(-self.predict_ahead) / self.df['Close'] - 1
        
        # 删除NaN值
        original_len = len(self.df)
        self.df.dropna(inplace=True)
        if self.config.verbose:
            print(f"特征计算完成，删除NaN后剩余 {len(self.df)} 条记录 (原有 {original_len} 条)")
        
    def prepare_ml_data(self):
        """准备机器学习数据"""
        if self.config.verbose:
            print("正在准备机器学习数据...")
        
        # 选择特征（保持原有特征不变）
        features = [
            'price_change', 'velocity_1m', 'velocity_5m', 'velocity_10m',
            'acceleration', 'volume_ratio', 'momentum_5m', 'momentum_10m',
            'amplitude', 'price_to_sma5', 'price_to_sma10'
        ]
        
        # 创建特征矩阵
        X = []
        y = []
        
        if self.config.verbose:
            print("  - 构建时间序列特征...")
        for i in range(self.lookback, len(self.df)):
            # 使用过去lookback分钟的特征
            feature_window = []
            for feature in features:
                feature_values = self.df[feature].iloc[i-self.lookback:i].values
                feature_window.extend(feature_values)
            
            X.append(feature_window)
            y.append(self.df['target'].iloc[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # 获取对应的时间索引
        all_dates = self.df.index[self.lookback:]
        
        # 划分训练集和测试集
        if self.config.use_date_split and all([
            self.config.train_start_date, self.config.train_end_date,
            self.config.test_start_date, self.config.test_end_date
        ]):
            # 使用日期分割
            train_start = pd.to_datetime(self.config.train_start_date)
            train_end = pd.to_datetime(self.config.train_end_date)
            test_start = pd.to_datetime(self.config.test_start_date)
            test_end = pd.to_datetime(self.config.test_end_date)
            
            # 找到对应的索引
            train_mask = (all_dates >= train_start) & (all_dates <= train_end)
            test_mask = (all_dates >= test_start) & (all_dates <= test_end)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) == 0 or len(test_indices) == 0:
                if self.config.verbose:
                    print(f"  - 警告：指定的日期范围内没有数据，回退到80/20划分")
                # 回退到比例划分
                split_idx = int(len(X) * 0.8)
                self.X_train, self.X_test = X[:split_idx], X[split_idx:]
                self.y_train, self.y_test = y[:split_idx], y[split_idx:]
                self.test_dates = all_dates[split_idx:]
            else:
                self.X_train, self.X_test = X[train_indices], X[test_indices]
                self.y_train, self.y_test = y[train_indices], y[test_indices]
                self.test_dates = all_dates[test_indices]
                
                if self.config.verbose:
                    print(f"  - 使用日期分割:")
                    print(f"    训练期间: {train_start.date()} 到 {train_end.date()} ({len(train_indices)} 个样本)")
                    print(f"    测试期间: {test_start.date()} 到 {test_end.date()} ({len(test_indices)} 个样本)")
        else:
            # 使用默认的80/20比例划分
            split_idx = int(len(X) * 0.8)
            self.X_train, self.X_test = X[:split_idx], X[split_idx:]
            self.y_train, self.y_test = y[:split_idx], y[split_idx:]
            self.test_dates = all_dates[split_idx:]
            
            if self.config.verbose:
                print(f"  - 使用80/20比例划分:")
                print(f"    训练期间: {all_dates[0].date()} 到 {all_dates[split_idx-1].date()}")
                print(f"    测试期间: {all_dates[split_idx].date()} 到 {all_dates[-1].date()}")
        
        # 标准化特征
        if self.config.verbose:
            print("  - 标准化特征...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        if self.config.verbose:
            print(f"训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
        
        # 分析目标变量分布
        if self.config.verbose:
            print(f"目标变量统计:")
            print(f"  - 平均值: {np.mean(y):.6f}")
            print(f"  - 标准差: {np.std(y):.6f}")
            print(f"  - 最小值: {np.min(y):.6f}")
            print(f"  - 最大值: {np.max(y):.6f}")
        
    def train_model(self):
        """训练随机森林模型"""
        if self.config.verbose:
            print("开始训练随机森林模型...")
            print(f"  - 使用 {self.X_train.shape[0]} 个训练样本，{self.X_train.shape[1]} 个特征")
        
        # 使用配置的模型参数
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        if self.config.verbose:
            print("  - 正在训练模型...")
        self.model.fit(self.X_train, self.y_train)
        if self.config.verbose:
            print("  - 模型训练完成")
        
        # 评估模型
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        
        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        if self.config.verbose:
            print(f"训练集MSE: {train_mse:.6f}")
            print(f"测试集MSE: {test_mse:.6f}")
            print(f"测试集R²: {test_r2:.4f}")
        
        # 分析预测分布
        if self.config.verbose:
            print(f"预测值统计:")
            print(f"  - 平均值: {np.mean(test_pred):.6f}")
            print(f"  - 标准差: {np.std(test_pred):.6f}")
            print(f"  - 最小值: {np.min(test_pred):.6f}")
            print(f"  - 最大值: {np.max(test_pred):.6f}")
        
    def backtest(self):
        """执行回测（保持原有逻辑不变）"""
        if self.config.verbose:
            print("\n开始回测...")
        
        # 预测
        predictions = self.model.predict(self.X_test)
        
        # 动态设置交易阈值（基于预测值的分位数）
        buy_threshold = np.percentile(predictions, self.config.buy_threshold_percentile)
        sell_threshold = np.percentile(predictions, self.config.sell_threshold_percentile)
        
        if self.config.verbose:
            print(f"动态交易阈值:")
            print(f"  - 买入阈值: {buy_threshold:.6f} ({buy_threshold*100:.3f}%)")
            print(f"  - 卖出阈值: {sell_threshold:.6f} ({sell_threshold*100:.3f}%)")
        
        signals = np.zeros_like(predictions)
        signals[predictions > buy_threshold] = 1  # 买入信号
        signals[predictions < sell_threshold] = -1  # 卖出信号
        
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        if self.config.verbose:
            print(f"生成信号统计: 买入信号 {buy_signals} 个, 卖出信号 {sell_signals} 个")
        
        # 计算收益（修正为全仓买入卖出逻辑）
        actual_returns = self.y_test
        position = 0  # 0=空仓, 1=满仓
        portfolio_value = self.config.initial_capital  # 使用实际初始资金
        portfolio_values = []
        trades = []
        daily_returns = []
        daily_pnl = {}  # 每日盈亏统计
        daily_trades = {}  # 每日交易次数统计
        daily_long_short = {}  # 每日多空单统计
        
        if self.config.verbose:
            print(f"开始交易模拟...")
        
        for i in range(len(signals)):
            current_date = self.test_dates[i].date()
            current_price = self.df['Close'].loc[self.test_dates[i]]
            
            if signals[i] == 1 and position == 0:  # 买入信号且当前空仓
                position = 1
                buy_price = current_price
                # 计算买入份额（全仓买入）
                shares = portfolio_value / buy_price
                trades.append(('买入', self.test_dates[i], buy_price, predictions[i], shares, portfolio_value))
                
                # 统计每日交易次数和多空单
                if current_date not in daily_trades:
                    daily_trades[current_date] = 0
                if current_date not in daily_long_short:
                    daily_long_short[current_date] = {'long': 0, 'short': 0}
                daily_trades[current_date] += 1
                daily_long_short[current_date]['long'] += 1
                    
            elif signals[i] == -1 and position == 1:  # 卖出信号且当前满仓
                position = 0
                sell_price = current_price
                # 找到对应的买入交易
                last_buy = None
                for trade in reversed(trades):
                    if trade[0] == '买入':
                        last_buy = trade
                        break
                
                if last_buy:
                    buy_price = last_buy[2]
                    shares = last_buy[4]
                    buy_value = last_buy[5]
                    
                    # 计算卖出价值和盈亏
                    sell_value = shares * sell_price
                    pnl = sell_value - buy_value
                    pnl_percent = (sell_value / buy_value - 1) * 100
                    
                    # 更新投资组合价值
                    portfolio_value = sell_value
                    
                    trades.append(('卖出', self.test_dates[i], sell_price, pnl_percent/100, shares, sell_value, pnl))
                    
                    # 记录每日盈亏
                    if current_date not in daily_pnl:
                        daily_pnl[current_date] = 0
                    daily_pnl[current_date] += pnl
                    
                    # 统计每日交易次数和多空单
                    if current_date not in daily_trades:
                        daily_trades[current_date] = 0
                    if current_date not in daily_long_short:
                        daily_long_short[current_date] = {'long': 0, 'short': 0}
                    daily_trades[current_date] += 1
                    daily_long_short[current_date]['short'] += 1
            
            portfolio_values.append(portfolio_value)
            
        # 如果最后还持有仓位，按最后价格计算
        if position == 1:
            last_price = self.df['Close'].loc[self.test_dates[-1]]
            last_buy = None
            for trade in reversed(trades):
                if trade[0] == '买入':
                    last_buy = trade
                    break
            if last_buy:
                shares = last_buy[4]
                portfolio_value = shares * last_price
        
        # 计算策略表现
        total_return = (portfolio_value / self.config.initial_capital - 1) * 100
        
        # 计算买入持有收益
        test_start_idx = self.df.index.get_loc(self.test_dates[0])
        test_end_idx = self.df.index.get_loc(self.test_dates[-1])
        buy_hold_return = (self.df['Close'].iloc[test_end_idx] / self.df['Close'].iloc[test_start_idx] - 1) * 100
        
        # 计算每日收益率
        daily_returns = []
        prev_value = self.config.initial_capital
        for value in portfolio_values:
            if prev_value > 0:
                daily_returns.append((value / prev_value - 1))
            else:
                daily_returns.append(0)
            prev_value = value
        
        # 计算夏普比率（分钟级别）
        returns_series = pd.Series(daily_returns)
        if returns_series.std() > 0:
            sharpe_ratio = np.sqrt(252 * 24 * 60) * returns_series.mean() / returns_series.std()
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        cumulative = pd.Series(portfolio_values)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # 计算胜率（基于卖出交易的收益）
        sell_trades = [t for t in trades if t[0] == '卖出']
        winning_trades = [t for t in sell_trades if len(t) > 6 and t[6] > 0]  # t[6]是盈亏金额
        losing_trades = [t for t in sell_trades if len(t) > 6 and t[6] <= 0]
        total_trades = len(sell_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # 统计买入和卖出数量
        buy_trades = [t for t in trades if t[0] == '买入']
        
        # 计算总的多空单数量
        total_long = sum(daily_ls.get('long', 0) for daily_ls in daily_long_short.values())
        total_short = sum(daily_ls.get('short', 0) for daily_ls in daily_long_short.values())
        
        if self.config.verbose:
            print(f"交易统计: 买入 {len(buy_trades)} 次, 卖出 {len(sell_trades)} 次")
            print(f"多空统计: 多单 {total_long} 次, 空单 {total_short} 次")
            
            # 打印交易详情
            if self.config.print_trades and len(trades) > 0:
                print(f"\n=== 详细交易记录 ===")
                
                # 确定要打印多少笔交易
                trades_to_show = len(sell_trades)  # 以完整交易对数为准
                if self.config.max_trades_to_print is not None:
                    trades_to_show = min(trades_to_show, self.config.max_trades_to_print)
                
                print(f"显示前 {trades_to_show} 笔完整交易（总共 {len(sell_trades)} 笔）：")
                
                current_position = 0
                printed_trades = 0
                
                for i, trade in enumerate(trades):
                    if printed_trades >= trades_to_show:
                        break
                        
                    if trade[0] == '买入':
                        current_position += 1
                        print(f"第{current_position}笔交易:")
                        print(f"  买入: {trade[1].strftime('%Y-%m-%d %H:%M')} 价格: ${trade[2]:.2f} 预测收益: {trade[3]*100:.3f}%")
                        print(f"       份额: {trade[4]:.6f} 投入金额: ${trade[5]:.2f}")
                    elif trade[0] == '卖出':
                        actual_return = trade[3] * 100
                        profit_loss = "盈利" if trade[3] > 0 else "亏损"
                        pnl_amount = trade[6]  # 盈亏金额
                        shares = trade[4]      # 份额
                        sell_value = trade[5]  # 卖出总价值
                        print(f"  卖出: {trade[1].strftime('%Y-%m-%d %H:%M')} 价格: ${trade[2]:.2f} 实际收益: {actual_return:.3f}% ({profit_loss})")
                        print(f"       份额: {shares:.6f} 卖出金额: ${sell_value:.2f} 盈亏: ${pnl_amount:+.2f}")
                        print(f"") # 空行分隔每笔完整交易
                        printed_trades += 1
                
                if len(sell_trades) > trades_to_show:
                    print(f"... 还有 {len(sell_trades) - trades_to_show} 笔交易未显示")
                    print(f"如需查看全部交易，请设置 config.max_trades_to_print = None")
            
            # 打印每日交易统计
            if self.config.print_daily_stats and (daily_pnl or daily_trades):
                print(f"\n=== 每日交易统计 ===")
                total_pnl = 0
                total_trades_count = 0
                total_long = 0
                total_short = 0
                
                # 获取所有有交易的日期
                all_trade_dates = set(daily_pnl.keys()) | set(daily_trades.keys())
                
                for date in sorted(all_trade_dates):
                    daily_amount = daily_pnl.get(date, 0)
                    daily_trade_count = daily_trades.get(date, 0)
                    daily_ls = daily_long_short.get(date, {'long': 0, 'short': 0})
                    
                    total_pnl += daily_amount
                    total_trades_count += daily_trade_count
                    total_long += daily_ls['long']
                    total_short += daily_ls['short']
                    
                    print(f"{date}: 交易{daily_trade_count:2d}次 (多{daily_ls['long']:2d}/空{daily_ls['short']:2d}) "
                          f"盈亏${daily_amount:+8.2f} (累计${total_pnl:+10.2f})")
                
                print(f"\n总计: 交易{total_trades_count}次 (多单{total_long}次/空单{total_short}次) 总盈亏${total_pnl:+.2f}")
                print(f"日均交易: {total_trades_count/len(all_trade_dates):.1f}次/天")
        
        # 打印回测结果
        if self.config.verbose:
            print("\n=== 回测结果 ===")
            print(f"初始资金: ${self.config.initial_capital:,.2f}")
            print(f"最终资金: ${portfolio_value:,.2f}")
            print(f"策略总收益: {total_return:.2f}%")
            print(f"买入持有收益: {buy_hold_return:.2f}%")
            print(f"超额收益: {total_return - buy_hold_return:.2f}%")
            print(f"夏普比率: {sharpe_ratio:.2f}")
            print(f"最大回撤: {max_drawdown:.2f}%")
            print(f"总交易次数: {total_trades}")
            print(f"胜率: {win_rate:.2f}%")
            print(f"预测准确率: {np.mean(np.sign(predictions) == np.sign(actual_returns)) * 100:.2f}%")
        
        # 绘制回测结果
        self.plot_backtest_results(portfolio_values, predictions, actual_returns, trades)
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'win_rate': win_rate
        }
    
    def plot_backtest_results(self, portfolio_values, predictions, actual_returns, trades):
        """绘制回测结果"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 1. 投资组合价值
        axes[0].plot(self.test_dates, portfolio_values, label='策略收益', color='blue', linewidth=2)
        axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        axes[0].set_title(f'投资组合价值 (初始资金: ${self.config.initial_capital:,})', fontsize=14)
        axes[0].set_ylabel('相对价值')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 预测vs实际
        axes[1].scatter(self.test_dates, predictions * 100, alpha=0.3, s=1, label='预测收益率', color='orange')
        axes[1].scatter(self.test_dates, actual_returns * 100, alpha=0.3, s=1, label='实际收益率', color='blue')
        axes[1].set_title('预测收益率 vs 实际收益率 (%)', fontsize=14)
        axes[1].set_ylabel('收益率 (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 预测误差分布
        errors = (predictions - actual_returns) * 100
        axes[2].hist(errors, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[2].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[2].set_title('预测误差分布', fontsize=14)
        axes[2].set_xlabel('预测误差 (%)')
        axes[2].set_ylabel('频数')
        axes[2].grid(True, alpha=0.3)
        
        # 4. 价格和交易信号
        test_prices = self.df['Close'].loc[self.test_dates]
        axes[3].plot(self.test_dates, test_prices, label='BTC价格', color='black', linewidth=1)
        
        # 标记买卖点
        buy_trades = [t for t in trades if t[0] == '买入']
        sell_trades = [t for t in trades if t[0] == '卖出']
        
        if buy_trades:
            buy_dates = [t[1] for t in buy_trades]
            buy_prices = [t[2] for t in buy_trades]  # t[2]是买入价格
            axes[3].scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='买入', zorder=5)
        
        if sell_trades:
            sell_dates = [t[1] for t in sell_trades]
            sell_prices = [t[2] for t in sell_trades]  # t[2]是卖出价格
            axes[3].scatter(sell_dates, sell_prices, color='red', marker='v', s=100, label='卖出', zorder=5)
        
        axes[3].set_title('BTC价格和交易信号', fontsize=14)
        axes[3].set_ylabel('价格 (USDT)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('backtest_results_configurable.png', dpi=150)
        if self.config.verbose:
            print("\n回测结果已保存到 backtest_results_configurable.png")
        
    def run(self):
        """运行完整的回测流程"""
        # 1. 加载数据
        self.load_data()
        
        # 2. 计算特征
        self.calculate_features()
        
        # 3. 准备机器学习数据
        self.prepare_ml_data()
        
        # 4. 训练模型
        self.train_model()
        
        # 5. 执行回测
        results = self.backtest()
        
        return results


def get_parameter_suggestions():
    """获取参数调优建议"""
    return """
    🎯 参数调优建议：
    
    === 基础参数 ===
    • initial_capital: 10000-100000 (初始资金)
    • data_limit: 20000-50000 (数据量，影响训练时间)
    
    === 时间范围配置 ===
    • use_date_split: True/False (是否使用日期分割)
    • 日期分割模式 (use_date_split=True):
      - train_start_date: '2024-01-01' (训练开始日期)
      - train_end_date: '2024-02-15' (训练结束日期)
      - test_start_date: '2024-02-16' (测试开始日期)
      - test_end_date: '2024-03-01' (测试结束日期)
    • 比例分割模式 (use_date_split=False): 自动80/20划分
    
    === 策略参数 ===
    • lookback: 5-20 (观察历史时间窗口)
      - 5-10: 短期反应快，噪音多
      - 10-15: 平衡选择
      - 15-20: 稳定但滞后
    
    • predict_ahead: 5-20 (预测未来时间)
      - 与lookback保持相近或相等
    
    === 交易阈值 ===
    • buy_threshold_percentile: 70-85 (买入阈值)
    • sell_threshold_percentile: 15-30 (卖出阈值)
      - 阈值差距大: 交易少但质量高
      - 阈值差距小: 交易多但噪音大
    
    === 模型参数 ===
    • n_estimators: 20-100 (树的数量)
    • max_depth: 8-20 (最大深度)
    • min_samples_split: 10-30
    • min_samples_leaf: 5-15
    
    === 优化建议 ===
    1. 保持原有成功的特征工程
    2. 重点调整交易阈值
    3. 适度调整模型复杂度
    4. 关注夏普比率和回撤平衡
    """


def create_date_split_config():
    """创建使用日期分割的配置示例"""
    config = BTCMicroTrendConfig()
    
    # 启用日期分割
    config.use_date_split = True
    
    # 设置训练和测试的时间范围（需要根据实际数据调整）
    config.train_start_date = '2024-01-01'  # 训练开始日期
    config.train_end_date = '2024-02-15'    # 训练结束日期
    config.test_start_date = '2024-02-16'   # 测试开始日期  
    config.test_end_date = '2024-03-01'     # 测试结束日期
    
    return config


if __name__ == "__main__":
    print(get_parameter_suggestions())
    
    # 直接使用配置类的设置（会读取你在配置类中的设置）
    config = BTCMicroTrendConfig()
    
    if config.use_date_split:
        print(f"\n使用日期分割配置:")
        print(f"  - 训练期间: {config.train_start_date} 到 {config.train_end_date}")
        print(f"  - 测试期间: {config.test_start_date} 到 {config.test_end_date}")
    else:
        print(f"\n使用默认配置（80/20比例划分）:")
    
    print(f"  - 初始资金: ${config.initial_capital:,}")
    if config.data_limit is not None:
        print(f"  - 数据量: {config.data_limit:,} 条")
    else:
        print(f"  - 数据量: 全部数据")
    print(f"  - 观察窗口: {config.lookback} 分钟")
    print(f"  - 预测时间: {config.predict_ahead} 分钟")
    print(f"  - 买入阈值: {config.buy_threshold_percentile}%")
    print(f"  - 卖出阈值: {config.sell_threshold_percentile}%")
    print(f"  - 随机森林参数: n_estimators={config.n_estimators}, max_depth={config.max_depth}")
    
    # 运行回测
    backtest = BTCMicroTrendBacktest(config)
    results = backtest.run()
    
    print("\n回测完成！") 