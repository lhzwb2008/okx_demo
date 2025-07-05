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
        self.test_end_date = '2024-02-29'       # 测试结束日期，格式: '2024-03-01' (一个月测试)
        self.use_date_split = True     # 是否使用日期分割（True）还是比例分割（False）
        
        # === 策略参数（优化后）===
        self.lookback = 30              # 增加观察窗口，获取更多历史信息
        self.predict_ahead = 10         # 保持10分钟预测
        
        # === 交易参数（简化版）===
        self.buy_threshold_percentile = 80   # 买入阈值
        self.sell_threshold_percentile = 20   # 卖出阈值
        
        # === 风险管理参数（简化版）===
        # 不使用止损和超时平仓，保持简单的开平仓逻辑
        
        # === 合约交易费用参数（欧易OKX）===
        self.enable_futures_trading = True   # 启用合约交易模式
        self.taker_fee_rate = 0.0005        # 吃单手续费率 0.05%
        self.funding_rate = 0.0001          # 资金费率 0.01%（每8小时）
        self.funding_interval_hours = 8     # 资金费率收取间隔（小时）
        
        # === 模型参数（优化后）===
        self.n_estimators = 50              # 增加树的数量，提高预测准确性
        self.max_depth = 10                  # 增加深度，捕捉更复杂的模式
        self.min_samples_split = 50          # 增加分割要求，减少过拟合
        self.min_samples_leaf = 20           # 增加叶子节点要求，提高泛化能力
        self.random_state = 42               # 随机种子
        
        # === 输出配置 ===
        self.verbose = True                  # 开启详细日志
        self.print_trades = True             # 开启交易详情
        self.max_trades_to_print = 50        # 打印前50笔交易
        self.print_daily_pnl = False         # 关闭每日盈亏
        self.print_daily_stats = False       # 关闭每日交易统计
        self.print_fee_details = True        # 开启费用明细
        self.print_win_rate_only = False     # 显示完整信息

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
        # 简化的交易信号
        strong_buy = predictions > buy_threshold
        strong_sell = predictions < sell_threshold
        
        signals[strong_buy] = 1  # 买入信号
        signals[strong_sell] = -1  # 卖出信号
        
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        if self.config.verbose:
            print(f"生成信号统计: 买入信号 {buy_signals} 个, 卖出信号 {sell_signals} 个")
        
        # 计算收益（修正为全仓买入卖出逻辑）
        actual_returns = self.y_test
        position = 0  # -1=空仓, 0=无仓, 1=多仓
        portfolio_value = self.config.initial_capital  # 使用实际初始资金
        portfolio_values = []
        trades = []
        daily_returns = []
        daily_pnl = {}  # 每日盈亏统计
        daily_trades = {}  # 每日交易次数统计
        daily_long_short = {}  # 每日多空单统计
        
        # 记录持仓信息
        entry_price = 0
        entry_shares = 0
        entry_value = 0
        entry_type = None  # 'long' 或 'short'
        entry_time_idx = 0  # 记录入场时间索引
        
        # 费用统计
        total_trading_fees = 0  # 总交易手续费
        total_funding_fees = 0  # 总资金费
        daily_trading_fees = {}  # 每日交易手续费
        daily_funding_fees = {}  # 每日资金费
        actual_funding_fees_paid = 0  # 实际支付的资金费
        
        if self.config.verbose:
            print(f"开始交易模拟...")
        
        for i in range(len(signals)):
            current_date = self.test_dates[i].date()
            current_price = self.df['Close'].loc[self.test_dates[i]]
            
            # 简化版：不使用止损和超时平仓，只根据信号开平仓
            
            # 多头开仓：买入信号且当前无仓
            if signals[i] == 1 and position == 0:
                # 计算开仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = portfolio_value * self.config.taker_fee_rate
                    portfolio_value -= trading_fee  # 扣除手续费
                    total_trading_fees += trading_fee
                    
                    if current_date not in daily_trading_fees:
                        daily_trading_fees[current_date] = 0
                    daily_trading_fees[current_date] += trading_fee
                
                position = 1
                entry_price = current_price
                entry_value = portfolio_value
                entry_shares = portfolio_value / current_price  # 买入份额
                entry_type = 'long'
                entry_time_idx = i  # 记录入场时间
                
                trades.append(('开多', self.test_dates[i], entry_price, predictions[i], entry_shares, entry_value, 0, trading_fee if self.config.enable_futures_trading else 0))
                
                # 统计每日交易
                if current_date not in daily_trades:
                    daily_trades[current_date] = 0
                if current_date not in daily_long_short:
                    daily_long_short[current_date] = {'long': 0, 'short': 0}
                daily_trades[current_date] += 1
                daily_long_short[current_date]['long'] += 1
                    
            # 空头开仓：卖出信号且当前无仓
            elif signals[i] == -1 and position == 0:
                # 计算开仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = portfolio_value * self.config.taker_fee_rate
                    portfolio_value -= trading_fee  # 扣除手续费
                    total_trading_fees += trading_fee
                    
                    if current_date not in daily_trading_fees:
                        daily_trading_fees[current_date] = 0
                    daily_trading_fees[current_date] += trading_fee
                
                position = -1
                entry_price = current_price
                entry_value = portfolio_value
                entry_shares = portfolio_value / current_price  # 做空份额
                entry_type = 'short'
                entry_time_idx = i  # 记录入场时间
                
                trades.append(('开空', self.test_dates[i], entry_price, predictions[i], entry_shares, entry_value, 0, trading_fee if self.config.enable_futures_trading else 0))
                
                # 统计每日交易
                if current_date not in daily_trades:
                    daily_trades[current_date] = 0
                if current_date not in daily_long_short:
                    daily_long_short[current_date] = {'long': 0, 'short': 0}
                daily_trades[current_date] += 1
                daily_long_short[current_date]['short'] += 1
                
            # 多头平仓：卖出信号且当前持多仓
            elif signals[i] == -1 and position == 1:
                # 计算多头收益
                exit_value = entry_shares * current_price
                
                # 计算平仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = exit_value * self.config.taker_fee_rate
                    exit_value -= trading_fee  # 扣除手续费
                    total_trading_fees += trading_fee
                    
                    if current_date not in daily_trading_fees:
                        daily_trading_fees[current_date] = 0
                    daily_trading_fees[current_date] += trading_fee
                
                pnl = exit_value - entry_value
                pnl_percent = (exit_value / entry_value - 1) * 100
                
                # 更新投资组合价值
                portfolio_value = exit_value
                
                trades.append(('平多', self.test_dates[i], current_price, pnl_percent/100, entry_shares, exit_value, pnl, trading_fee if self.config.enable_futures_trading else 0))
                
                # 记录每日盈亏
                if current_date not in daily_pnl:
                    daily_pnl[current_date] = 0
                daily_pnl[current_date] += pnl
                
                # 重置仓位
                position = 0
                entry_price = 0
                entry_shares = 0
                entry_value = 0
                entry_type = None
                
            # 空头平仓：买入信号且当前持空仓
            elif signals[i] == 1 and position == -1:
                # 计算空头收益（价格下跌时盈利）
                price_change_ratio = current_price / entry_price
                exit_value = entry_value * (2 - price_change_ratio)  # 空头收益计算
                
                # 计算平仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = exit_value * self.config.taker_fee_rate
                    exit_value -= trading_fee  # 扣除手续费
                    total_trading_fees += trading_fee
                    
                    if current_date not in daily_trading_fees:
                        daily_trading_fees[current_date] = 0
                    daily_trading_fees[current_date] += trading_fee
                
                pnl = exit_value - entry_value
                pnl_percent = (exit_value / entry_value - 1) * 100
                
                # 更新投资组合价值
                portfolio_value = exit_value
                
                trades.append(('平空', self.test_dates[i], current_price, pnl_percent/100, entry_shares, exit_value, pnl, trading_fee if self.config.enable_futures_trading else 0))
                
                # 记录每日盈亏
                if current_date not in daily_pnl:
                    daily_pnl[current_date] = 0
                daily_pnl[current_date] += pnl
                
                # 重置仓位
                position = 0
                entry_price = 0
                entry_shares = 0
                entry_value = 0
                entry_type = None
            
            # 计算资金费（如果持仓）
            if self.config.enable_futures_trading and position != 0:
                # 计算实际持仓时间（分钟）
                holding_minutes = i - entry_time_idx
                
                # 检查是否跨过资金费收取时间点（每8小时）
                # 获取入场和当前时间
                entry_time = self.test_dates[entry_time_idx]
                current_time = self.test_dates[i]
                
                # 计算跨过的8小时时间点数量
                entry_hour = entry_time.hour
                current_hour = current_time.hour
                days_diff = (current_time.date() - entry_time.date()).days
                
                # 资金费收取时间点：0:00, 8:00, 16:00 (UTC+8)
                funding_times = [0, 8, 16]
                
                # 计算是否刚好跨过资金费时间点
                for funding_hour in funding_times:
                    # 检查是否在这一分钟刚好跨过资金费时间
                    if i > entry_time_idx:
                        prev_time = self.test_dates[i-1]
                        if prev_time.hour < funding_hour <= current_hour or (days_diff > 0 and current_hour == funding_hour and current_time.minute == 0):
                            # 计算资金费
                            funding_fee = portfolio_value * self.config.funding_rate
                            
                            # 多头支付资金费，空头收取资金费（假设资金费率为正）
                            if position == 1:
                                portfolio_value -= funding_fee
                                total_funding_fees += funding_fee
                                actual_funding_fees_paid += funding_fee
                            else:  # position == -1
                                portfolio_value += funding_fee
                                total_funding_fees -= funding_fee
                                actual_funding_fees_paid -= funding_fee
                            
                            if current_date not in daily_funding_fees:
                                daily_funding_fees[current_date] = 0
                            daily_funding_fees[current_date] += funding_fee if position == 1 else -funding_fee
            
            portfolio_values.append(portfolio_value)
            
        # 如果最后还持有仓位，按最后价格平仓
        if position != 0:
            last_price = self.df['Close'].loc[self.test_dates[-1]]
            last_date = self.test_dates[-1].date()
            
            if position == 1:  # 持有多仓
                exit_value = entry_shares * last_price
                
                # 计算平仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = exit_value * self.config.taker_fee_rate
                    exit_value -= trading_fee
                    total_trading_fees += trading_fee
                    
                    if last_date not in daily_trading_fees:
                        daily_trading_fees[last_date] = 0
                    daily_trading_fees[last_date] += trading_fee
                
                pnl = exit_value - entry_value
                pnl_percent = (exit_value / entry_value - 1) * 100
                portfolio_value = exit_value
                trades.append(('平多(结束)', self.test_dates[-1], last_price, pnl_percent/100, entry_shares, exit_value, pnl, trading_fee if self.config.enable_futures_trading else 0))
                
            elif position == -1:  # 持有空仓
                price_change_ratio = last_price / entry_price
                exit_value = entry_value * (2 - price_change_ratio)
                
                # 计算平仓手续费
                if self.config.enable_futures_trading:
                    trading_fee = exit_value * self.config.taker_fee_rate
                    exit_value -= trading_fee
                    total_trading_fees += trading_fee
                    
                    if last_date not in daily_trading_fees:
                        daily_trading_fees[last_date] = 0
                    daily_trading_fees[last_date] += trading_fee
                
                pnl = exit_value - entry_value
                pnl_percent = (exit_value / entry_value - 1) * 100
                portfolio_value = exit_value
                trades.append(('平空(结束)', self.test_dates[-1], last_price, pnl_percent/100, entry_shares, exit_value, pnl, trading_fee if self.config.enable_futures_trading else 0))
            
            # 记录最后的盈亏
            if last_date not in daily_pnl:
                daily_pnl[last_date] = 0
            daily_pnl[last_date] += pnl
        
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
        
        # 计算胜率（基于平仓交易的收益）
        close_trades = [t for t in trades if t[0] in ['平多', '平空', '平多(结束)', '平空(结束)']]
        winning_trades = [t for t in close_trades if len(t) > 6 and t[6] > 0]  # t[6]是盈亏金额
        losing_trades = [t for t in close_trades if len(t) > 6 and t[6] <= 0]
        total_trades = len(close_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # 统计开仓和平仓数量
        long_open_trades = [t for t in trades if t[0] == '开多']
        short_open_trades = [t for t in trades if t[0] == '开空']
        long_close_trades = [t for t in trades if t[0] in ['平多', '平多(结束)']]
        short_close_trades = [t for t in trades if t[0] in ['平空', '平空(结束)']]
        
        # 计算总的多空单数量
        total_long = sum(daily_ls.get('long', 0) for daily_ls in daily_long_short.values())
        total_short = sum(daily_ls.get('short', 0) for daily_ls in daily_long_short.values())
        
        if self.config.verbose:
            print(f"交易统计: 开多 {len(long_open_trades)} 次, 开空 {len(short_open_trades)} 次")
            print(f"平仓统计: 平多 {len(long_close_trades)} 次, 平空 {len(short_close_trades)} 次")
            print(f"多空统计: 多单 {total_long} 次, 空单 {total_short} 次")
            
            # 打印交易详情
            if self.config.print_trades and len(trades) > 0:
                print(f"\n=== 详细交易记录 ===")
                
                # 确定要打印多少笔交易
                trades_to_show = len(close_trades)  # 以完整交易对数为准
                if self.config.max_trades_to_print is not None:
                    trades_to_show = min(trades_to_show, self.config.max_trades_to_print)
                
                print(f"显示前 {trades_to_show} 笔完整交易（总共 {len(close_trades)} 笔）：")
                
                current_position = 0
                printed_trades = 0
                
                for i, trade in enumerate(trades):
                    if printed_trades >= trades_to_show:
                        break
                        
                    if trade[0] in ['开多', '开空']:
                        current_position += 1
                        trade_type = "多单" if trade[0] == '开多' else "空单"
                        print(f"第{current_position}笔交易 ({trade_type}):")
                        print(f"  📈 {trade[0]}: {trade[1].strftime('%Y-%m-%d %H:%M')} 价格: ${trade[2]:.2f} 预测收益: {trade[3]*100:.3f}%")
                        print(f"       💰 投入金额: ${trade[5]:.2f} | 交易量: {trade[4]:.6f} BTC")
                        if self.config.enable_futures_trading and len(trade) > 7:
                            print(f"       💸 开仓手续费: ${trade[7]:.2f} ({trade[7]/trade[5]*100:.3f}%)")
                        
                        # 存储开仓信息用于计算总费用
                        entry_info = {
                            'entry_price': trade[2],
                            'entry_amount': trade[5],
                            'entry_fee': trade[7] if self.config.enable_futures_trading and len(trade) > 7 else 0,
                            'shares': trade[4]
                        }
                        
                    elif trade[0] in ['平多', '平空', '平多(结束)', '平空(结束)']:
                        actual_return = trade[3] * 100
                        profit_loss = "🟢 盈利" if trade[3] > 0 else "🔴 亏损"
                        pnl_amount = trade[6]  # 盈亏金额
                        shares = trade[4]      # 份额
                        exit_value = trade[5]  # 平仓总价值
                        exit_price = trade[2]  # 平仓价格
                        
                        # 计算总手续费
                        exit_fee = trade[7] if self.config.enable_futures_trading and len(trade) > 7 else 0
                        total_fees = entry_info.get('entry_fee', 0) + exit_fee
                        
                        # 计算价格变化
                        price_change = (exit_price - entry_info.get('entry_price', 0)) / entry_info.get('entry_price', 1) * 100
                        
                        print(f"  📉 {trade[0]}: {trade[1].strftime('%Y-%m-%d %H:%M')} 价格: ${exit_price:.2f} 实际收益: {actual_return:.3f}% ({profit_loss})")
                        print(f"       💰 平仓金额: ${exit_value:.2f} | 交易量: {shares:.6f} BTC")
                        print(f"       📊 价格变化: {price_change:+.3f}% | 盈亏: ${pnl_amount:+.2f}")
                        
                        if self.config.enable_futures_trading:
                            print(f"       💸 平仓手续费: ${exit_fee:.2f} ({exit_fee/exit_value*100:.3f}%)")
                            print(f"       💸 总手续费: ${total_fees:.2f} | 净收益: ${pnl_amount:+.2f}")
                            
                        # 计算收益率统计
                        gross_return = (exit_value + entry_info.get('entry_fee', 0) + exit_fee) / entry_info.get('entry_amount', 1) - 1
                        net_return = pnl_amount / entry_info.get('entry_amount', 1)
                        
                        print(f"       📈 毛收益率: {gross_return*100:+.3f}% | 净收益率: {net_return*100:+.3f}%")
                        print(f"") # 空行分隔每笔完整交易
                        printed_trades += 1
                
                if len(close_trades) > trades_to_show:
                    print(f"... 还有 {len(close_trades) - trades_to_show} 笔交易未显示")
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
                    
                    output_str = f"{date}: 交易{daily_trade_count:2d}次 (多{daily_ls['long']:2d}/空{daily_ls['short']:2d}) "
                    output_str += f"盈亏${daily_amount:+8.2f} (累计${total_pnl:+10.2f})"
                    
                    # 添加每日费用信息
                    if self.config.enable_futures_trading:
                        daily_tfee = daily_trading_fees.get(date, 0)
                        daily_ffee = daily_funding_fees.get(date, 0)
                        if daily_tfee > 0 or daily_ffee != 0:
                            output_str += f" | 手续费${daily_tfee:.2f} 资金费${daily_ffee:+.2f}"
                    
                    print(output_str)
                
                print(f"\n总计: 交易{total_trades_count}次 (多单{total_long}次/空单{total_short}次) 总盈亏${total_pnl:+.2f}")
                print(f"日均交易: {total_trades_count/len(all_trade_dates):.1f}次/天")
        
        # 只打印胜率相关信息
        if hasattr(self.config, 'print_win_rate_only') and self.config.print_win_rate_only:
            print(f"胜率: {win_rate:.2f}% | 交易次数: {total_trades} | 净收益: {total_return:.2f}%")
        elif self.config.verbose:
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
            
            # 打印费用明细
            if self.config.enable_futures_trading and self.config.print_fee_details:
                print("\n=== 费用明细（欧易合约）===")
                print(f"吃单费率: {self.config.taker_fee_rate*100:.3f}%")
                print(f"资金费率: {self.config.funding_rate*100:.3f}% (每{self.config.funding_interval_hours}小时)")
                print(f"总交易手续费: ${total_trading_fees:,.2f} ({total_trading_fees/self.config.initial_capital*100:.2f}%)")
                print(f"总资金费: ${total_funding_fees:,.2f} ({total_funding_fees/self.config.initial_capital*100:.2f}%)")
                print(f"总费用: ${total_trading_fees + total_funding_fees:,.2f} ({(total_trading_fees + total_funding_fees)/self.config.initial_capital*100:.2f}%)")
                
                # 计算净收益
                gross_pnl = portfolio_value - self.config.initial_capital + total_trading_fees + total_funding_fees
                net_pnl = portfolio_value - self.config.initial_capital
                print(f"\n毛利润(未扣费): ${gross_pnl:,.2f} ({gross_pnl/self.config.initial_capital*100:.2f}%)")
                print(f"净利润(已扣费): ${net_pnl:,.2f} ({net_pnl/self.config.initial_capital*100:.2f}%)")
        
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
        
        # 标记开仓和平仓点
        long_open = [t for t in trades if t[0] == '开多']
        short_open = [t for t in trades if t[0] == '开空']
        long_close = [t for t in trades if t[0] in ['平多', '平多(结束)']]
        short_close = [t for t in trades if t[0] in ['平空', '平空(结束)']]
        
        if long_open:
            dates = [t[1] for t in long_open]
            prices = [t[2] for t in long_open]
            axes[3].scatter(dates, prices, color='green', marker='^', s=100, label='开多', zorder=5)
        
        if short_open:
            dates = [t[1] for t in short_open]
            prices = [t[2] for t in short_open]
            axes[3].scatter(dates, prices, color='red', marker='v', s=100, label='开空', zorder=5)
            
        if long_close:
            dates = [t[1] for t in long_close]
            prices = [t[2] for t in long_close]
            axes[3].scatter(dates, prices, color='lightgreen', marker='x', s=100, label='平多', zorder=5)
            
        if short_close:
            dates = [t[1] for t in short_close]
            prices = [t[2] for t in short_close]
            axes[3].scatter(dates, prices, color='lightcoral', marker='x', s=100, label='平空', zorder=5)
        
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
    # === 胜率优化参数建议 ===
    # 可以手动调整以下参数来优化胜率：
    # 
    # 交易阈值（影响交易频率和质量）：
    # - buy_threshold_percentile: 75, 80, 85, 90, 95  # 买入阈值，越高交易越少但质量越好
    # - sell_threshold_percentile: 25, 20, 15, 10, 5   # 卖出阈值，与买入阈值对称
    # 
    # 已移除最小收益过滤，使用简化的交易逻辑
    # 
    # 观察窗口（影响预测准确性）：
    # - lookback: 10, 15, 20, 25  # 观察历史时间窗口
    # - predict_ahead: 10, 15, 20  # 预测未来时间
    # 
    # 模型参数（影响预测能力）：
    # - n_estimators: 50, 100, 150  # 随机森林树的数量
    # - max_depth: 10, 15, 20  # 树的最大深度
    
    # 创建配置并运行单次回测
    config = BTCMicroTrendConfig()
    
    # 可以在这里手动调整参数，例如：
    # config.buy_threshold_percentile = 85
    # config.sell_threshold_percentile = 15
    
    print("=== 当前参数配置 ===")
    print(f"买入阈值: {config.buy_threshold_percentile}%")
    print(f"卖出阈值: {config.sell_threshold_percentile}%")
    print(f"观察窗口: {config.lookback}分钟")
    print(f"预测时间: {config.predict_ahead}分钟")
    print(f"随机森林: {config.n_estimators}棵树, 深度{config.max_depth}")
    print(f"简化版本: 无止损、无超时平仓、无最小收益过滤")
    print()
    
    # 运行回测
    backtest = BTCMicroTrendBacktest(config)
    results = backtest.run()
    
    print("回测完成！") 