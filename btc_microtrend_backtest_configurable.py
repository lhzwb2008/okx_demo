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
        self.initial_capital = 10000  # 初始资金（USDT）
        
        # === 数据配置 ===
        self.data_limit = 30000  # 使用数据条数
        
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
        
        # 只使用最近的数据以加快处理速度
        self.df = self.df.tail(self.config.data_limit)
        if self.config.verbose:
            print(f"数据预处理完成，使用最近 {len(self.df)} 条记录")
        
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
        
        # 划分训练集和测试集
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        # 标准化特征
        if self.config.verbose:
            print("  - 标准化特征...")
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 保存对应的时间索引
        self.test_dates = self.df.index[self.lookback + split_idx:]
        
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
        
        # 计算收益（保持原有逻辑）
        actual_returns = self.y_test
        position = 0
        portfolio_value = 1.0
        portfolio_values = []
        trades = []
        daily_returns = []
        
        for i in range(len(signals)):
            daily_return = 0
            
            if signals[i] == 1 and position == 0:  # 买入
                position = 1
                trades.append(('买入', self.test_dates[i], predictions[i]))
            elif signals[i] == -1 and position == 1:  # 卖出
                position = 0
                daily_return = actual_returns[i]
                portfolio_value *= (1 + actual_returns[i])
                trades.append(('卖出', self.test_dates[i], actual_returns[i]))
            elif position == 1:  # 持仓
                daily_return = actual_returns[i]
                portfolio_value *= (1 + actual_returns[i])
            
            portfolio_values.append(portfolio_value)
            daily_returns.append(daily_return)
        
        # 计算策略表现
        total_return = (portfolio_value - 1) * 100
        
        # 计算买入持有收益
        test_start_idx = self.df.index.get_loc(self.test_dates[0])
        test_end_idx = self.df.index.get_loc(self.test_dates[-1])
        buy_hold_return = (self.df['Close'].iloc[test_end_idx] / self.df['Close'].iloc[test_start_idx] - 1) * 100
        
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
        
        # 计算胜率
        winning_trades = [t for t in trades if t[0] == '卖出' and t[2] > 0]
        losing_trades = [t for t in trades if t[0] == '卖出' and t[2] <= 0]
        total_trades = len(winning_trades) + len(losing_trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        # 打印回测结果
        if self.config.verbose:
            print("\n=== 回测结果 ===")
            print(f"初始资金: ${self.config.initial_capital:,.2f}")
            print(f"最终资金: ${self.config.initial_capital * portfolio_value:,.2f}")
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
        buy_dates = [t[1] for t in trades if t[0] == '买入']
        sell_dates = [t[1] for t in trades if t[0] == '卖出']
        
        if buy_dates:
            buy_prices = self.df['Close'].loc[buy_dates]
            axes[3].scatter(buy_dates, buy_prices, color='green', marker='^', s=100, label='买入', zorder=5)
        
        if sell_dates:
            sell_prices = self.df['Close'].loc[sell_dates]
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


if __name__ == "__main__":
    print(get_parameter_suggestions())
    
    # 使用默认配置（与原optimized版本相同）
    config = BTCMicroTrendConfig()
    
    # 你可以在这里修改参数，例如：
    # config.initial_capital = 50000
    # config.lookback = 15
    # config.buy_threshold_percentile = 80
    
    # 运行回测
    backtest = BTCMicroTrendBacktest(config)
    results = backtest.run()
    
    print("\n回测完成！") 