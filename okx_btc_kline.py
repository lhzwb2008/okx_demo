import requests
import pandas as pd  # pyright: ignore
import json
import time
from datetime import datetime, timedelta, timezone
import hmac
import hashlib
import base64

# ==================== 配置区域 ====================
# 时间范围配置 (格式: YYYY-MM-DD)
START_DATE = "2025-06-01"  # 开始日期
END_DATE = "2025-06-29"    # 结束日期

# API密钥配置
API_KEY = "9e3982ed-4c73-4436-8cf9-6f790b99fbfc"
SECRET_KEY = "FE1CF3A502A9EE9D52CD60346DE4DD8D"
PASSPHRASE = "Hello@2025"

# 交易对配置
SYMBOL = "BTC-USDT"  # 交易对
BAR_SIZE = "1m"      # K线周期: 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M

# 文件输出配置
OUTPUT_FILENAME = "okx_btc_1m.csv"
# ================================================

class OKXClient:
    """OKX API客户端"""
    
    def __init__(self, api_key=None, secret_key=None, passphrase=None, is_demo=False):
        """
        初始化OKX客户端
        
        参数:
            api_key: API密钥
            secret_key: 密钥
            passphrase: 密码短语
            is_demo: 是否为模拟环境
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        
        # 设置基础URL
        if is_demo:
            self.base_url = "https://www.okx.com"  # 模拟环境
        else:
            self.base_url = "https://www.okx.com"  # 正式环境
            
        self.headers = {
            'Content-Type': 'application/json',
            'OK-ACCESS-KEY': self.api_key or '',
            'OK-ACCESS-SIGN': '',
            'OK-ACCESS-TIMESTAMP': '',
            'OK-ACCESS-PASSPHRASE': self.passphrase or ''
        }
    
    def _generate_signature(self, timestamp, method, request_path, body=''):
        """生成签名"""
        if not self.secret_key:
            return ''
            
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf-8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def _make_request(self, method, endpoint, params=None, data=None):
        """发送HTTP请求"""
        url = self.base_url + endpoint
        
        # 生成时间戳
        timestamp = datetime.now(timezone.utc).isoformat()[:-3] + 'Z'
        
        # 准备请求体
        if method == 'GET':
            query_string = '&'.join([f"{k}={v}" for k, v in (params or {}).items()])
            if query_string:
                url += '?' + query_string
            body = ''
        else:
            body = json.dumps(data) if data else ''
        
        # 生成签名
        signature = self._generate_signature(timestamp, method, endpoint + ('?' + query_string if method == 'GET' and params else ''), body)
        
        # 设置请求头
        headers = self.headers.copy()
        headers['OK-ACCESS-SIGN'] = signature
        headers['OK-ACCESS-TIMESTAMP'] = timestamp
        
        # 发送请求
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            else:
                response = requests.post(url, headers=headers, data=body, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")
            return None
    
    def get_instruments(self, inst_type='SPOT'):
        """
        获取交易产品基础信息
        
        参数:
            inst_type: 产品类型 (SPOT, SWAP, FUTURES, OPTION)
        """
        endpoint = '/api/v5/public/instruments'
        params = {'instType': inst_type}
        
        return self._make_request('GET', endpoint, params)
    
    def get_kline_data(self, inst_id='BTC-USDT', bar='1m', limit=100, after=None, before=None):
        """
        获取K线数据
        
        参数:
            inst_id: 产品ID，如 BTC-USDT
            bar: K线周期 (1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M, 3M, 6M, 1Y)
            limit: 返回结果的数量，最大值为300，默认返回100条
            after: 请求此时间戳之后（更旧的数据）的分页内容，传的值为对应接口的ts
            before: 请求此时间戳之前（更新的数据）的分页内容，传的值为对应接口的ts
        """
        endpoint = '/api/v5/market/candles'
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        if after:
            params['after'] = str(after)
        if before:
            params['before'] = str(before)
        
        return self._make_request('GET', endpoint, params)
    
    def get_ticker(self, inst_id='BTC-USDT'):
        """
        获取单个产品行情信息
        
        参数:
            inst_id: 产品ID
        """
        endpoint = '/api/v5/market/ticker'
        params = {'instId': inst_id}
        
        return self._make_request('GET', endpoint, params)
    
    def get_account_balance(self, ccy=None):
        """
        获取账户余额
        
        参数:
            ccy: 币种，如 BTC。支持多币种查询，币种之间半角逗号分隔
        """
        endpoint = '/api/v5/account/balance'
        params = {}
        if ccy:
            params['ccy'] = ccy
            
        return self._make_request('GET', endpoint, params)
    
    def get_history_index_candles(self, inst_id='BTC-USD', bar='1m', limit=100, after=None, before=None):
        """
        获取指数历史K线数据
        
        参数:
            inst_id: 现货指数，如BTC-USD
            bar: 时间粒度，默认值1m
            limit: 分页返回的结果集数量，最大为100，不填默认返回100条
            after: 请求此时间戳之前（更旧的数据）的分页内容
            before: 请求此时间戳之后（更新的数据）的分页内容
        """
        endpoint = '/api/v5/market/history-index-candles'
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        if after:
            params['after'] = str(after)
        if before:
            params['before'] = str(before)
        
        return self._make_request('GET', endpoint, params)
    
    def get_mark_price_candles(self, inst_id='BTC-USD-SWAP', bar='1m', limit=100, after=None, before=None):
        """
        获取标记价格K线数据
        
        参数:
            inst_id: 产品ID，如BTC-USD-SWAP
            bar: 时间粒度，默认值1m
            limit: 分页返回的结果集数量，最大为100，不填默认返回100条
            after: 请求此时间戳之前（更旧的数据）的分页内容
            before: 请求此时间戳之后（更新的数据）的分页内容
        """
        endpoint = '/api/v5/market/mark-price-candles'
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        if after:
            params['after'] = str(after)
        if before:
            params['before'] = str(before)
        
        return self._make_request('GET', endpoint, params)
    
    def get_history_mark_price_candles(self, inst_id='BTC-USD-SWAP', bar='1m', limit=100, after=None, before=None):
        """
        获取标记价格历史K线数据
        
        参数:
            inst_id: 产品ID，如BTC-USD-SWAP
            bar: 时间粒度，默认值1m
            limit: 分页返回的结果集数量，最大为100，不填默认返回100条
            after: 请求此时间戳之前（更旧的数据）的分页内容
            before: 请求此时间戳之后（更新的数据）的分页内容
        """
        endpoint = '/api/v5/market/history-mark-price-candles'
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': str(limit)
        }
        
        if after:
            params['after'] = str(after)
        if before:
            params['before'] = str(before)
        
        return self._make_request('GET', endpoint, params)

def fetch_btc_kline_data(client, days=7, save_to_csv=True):
    """
    获取比特币1分钟K线数据
    
    参数:
        client: OKX客户端实例
        days: 获取多少天的数据
        save_to_csv: 是否保存到CSV文件
    """
    print(f"开始获取比特币过去{days}天的1分钟K线数据...")
    
    all_data = []
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # 由于OKX API限制每次最多返回300条数据，需要分批获取
    current_time = end_time
    batch_count = 0
    
    while current_time > start_time:
        batch_count += 1
        print(f"正在获取第{batch_count}批数据...")
        
        # 获取K线数据
        result = client.get_kline_data(
            inst_id='BTC-USDT',
            bar='1m',
            limit=300,
            before=int(current_time.timestamp() * 1000)
        )
        
        if not result or result.get('code') != '0':
            print(f"获取数据失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print("没有更多数据")
            break
        
        # 处理数据
        for item in data:
            # OKX返回的数据格式: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # 只保留指定时间范围内的数据
            if dt < start_time:
                break
                
            kline_data = {
                'DateTime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Open': float(item[1]),
                'High': float(item[2]),
                'Low': float(item[3]),
                'Close': float(item[4]),
                'Volume': float(item[5])
            }
            all_data.append(kline_data)
        
        # 更新时间戳，获取更早的数据
        if data:
            current_time = datetime.fromtimestamp(int(data[-1][0]) / 1000)
        
        # 避免请求过于频繁
        time.sleep(0.1)
        
        # 如果已经获取到足够早的数据，停止
        if current_time <= start_time:
            break
    
    # 按时间排序（从早到晚）
    all_data.sort(key=lambda x: x['DateTime'])
    
    # 过滤掉超出时间范围的数据
    filtered_data = [item for item in all_data if item['DateTime'] >= start_time.strftime('%Y-%m-%d %H:%M:%S')]
    
    print(f"共获取到 {len(filtered_data)} 条K线数据")
    
    if save_to_csv and filtered_data:
        # 转换为DataFrame
        df = pd.DataFrame(filtered_data)
        
        # 保存到CSV
        filename = f'btc_1m_kline_{days}days.csv'
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")
        
        # 显示数据统计
        print(f"\n数据统计:")
        print(f"时间范围: {df['DateTime'].min()} 到 {df['DateTime'].max()}")
        print(f"最高价: ${df['High'].max():.2f}")
        print(f"最低价: ${df['Low'].min():.2f}")
        print(f"最新价: ${df['Close'].iloc[-1]:.2f}")
        print(f"总成交量: {df['Volume'].sum():.2f} BTC")
        
        return df
    
    return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

def fetch_btc_index_history_data(client, days=30, save_to_csv=True):
    """
    获取比特币指数历史K线数据
    
    参数:
        client: OKX客户端实例
        days: 获取多少天的数据
        save_to_csv: 是否保存到CSV文件
    """
    print(f"开始获取比特币指数过去{days}天的1分钟历史K线数据...")
    
    all_data = []
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # 由于OKX API限制每次最多返回100条数据，需要分批获取
    current_time = end_time
    batch_count = 0
    
    while current_time > start_time:
        batch_count += 1
        print(f"正在获取第{batch_count}批指数数据...")
        
        # 获取指数历史K线数据
        result = client.get_history_index_candles(
            inst_id='BTC-USD',
            bar='1m',
            limit=100,
            before=int(current_time.timestamp() * 1000)
        )
        
        if not result or result.get('code') != '0':
            print(f"获取指数数据失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print("没有更多指数数据")
            break
        
        # 处理数据
        for item in data:
            # OKX返回的指数数据格式: [ts, o, h, l, c, confirm]
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # 只保留指定时间范围内的数据
            if dt < start_time:
                break
                
            index_data = {
                'timestamp': timestamp,
                'datetime': dt,
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'confirm': item[5]
            }
            all_data.append(index_data)
        
        # 更新时间戳，获取更早的数据
        if data:
            current_time = datetime.fromtimestamp(int(data[-1][0]) / 1000)
        
        # 避免请求过于频繁
        time.sleep(0.1)
        
        # 如果已经获取到足够早的数据，停止
        if current_time <= start_time:
            break
    
    # 按时间排序（从早到晚）
    all_data.sort(key=lambda x: x['timestamp'])
    
    # 过滤掉超出时间范围的数据
    filtered_data = [item for item in all_data if item['datetime'] >= start_time]
    
    print(f"共获取到 {len(filtered_data)} 条指数K线数据")
    
    if save_to_csv and filtered_data:
        # 转换为DataFrame
        df = pd.DataFrame(filtered_data)
        
        # 保存到CSV
        filename = f'btc_index_1m_history_{days}days.csv'
        df.to_csv(filename, index=False)
        print(f"指数数据已保存到 {filename}")
        
        # 显示数据统计
        print(f"\n指数数据统计:")
        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        print(f"最高价: ${df['high'].max():.2f}")
        print(f"最低价: ${df['low'].min():.2f}")
        print(f"最新价: ${df['close'].iloc[-1]:.2f}")
        
        return df
    
    return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

def fetch_mark_price_data(client, days=7, save_to_csv=True):
    """
    获取比特币标记价格K线数据
    
    参数:
        client: OKX客户端实例
        days: 获取多少天的数据
        save_to_csv: 是否保存到CSV文件
    """
    print(f"开始获取比特币标记价格过去{days}天的1分钟K线数据...")
    
    all_data = []
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # 由于OKX API限制每次最多返回100条数据，需要分批获取
    current_time = end_time
    batch_count = 0
    
    while current_time > start_time:
        batch_count += 1
        print(f"正在获取第{batch_count}批标记价格数据...")
        
        # 获取标记价格K线数据
        result = client.get_mark_price_candles(
            inst_id='BTC-USD-SWAP',
            bar='1m',
            limit=100,
            before=int(current_time.timestamp() * 1000)
        )
        
        if not result or result.get('code') != '0':
            print(f"获取标记价格数据失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print("没有更多标记价格数据")
            break
        
        # 处理数据
        for item in data:
            # OKX返回的标记价格数据格式: [ts, o, h, l, c, confirm]
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # 只保留指定时间范围内的数据
            if dt < start_time:
                break
                
            mark_price_data = {
                'timestamp': timestamp,
                'datetime': dt,
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'confirm': item[5]
            }
            all_data.append(mark_price_data)
        
        # 更新时间戳，获取更早的数据
        if data:
            current_time = datetime.fromtimestamp(int(data[-1][0]) / 1000)
        
        # 避免请求过于频繁
        time.sleep(0.1)
        
        # 如果已经获取到足够早的数据，停止
        if current_time <= start_time:
            break
    
    # 按时间排序（从早到晚）
    all_data.sort(key=lambda x: x['timestamp'])
    
    # 过滤掉超出时间范围的数据
    filtered_data = [item for item in all_data if item['datetime'] >= start_time]
    
    print(f"共获取到 {len(filtered_data)} 条标记价格K线数据")
    
    if save_to_csv and filtered_data:
        # 转换为DataFrame
        df = pd.DataFrame(filtered_data)
        
        # 保存到CSV
        filename = f'btc_mark_price_1m_{days}days.csv'
        df.to_csv(filename, index=False)
        print(f"标记价格数据已保存到 {filename}")
        
        # 显示数据统计
        print(f"\n标记价格数据统计:")
        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        print(f"最高价: ${df['high'].max():.2f}")
        print(f"最低价: ${df['low'].min():.2f}")
        print(f"最新价: ${df['close'].iloc[-1]:.2f}")
        
        return df
    
    return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

def fetch_mark_price_history_data(client, days=30, save_to_csv=True):
    """
    获取比特币标记价格历史K线数据
    
    参数:
        client: OKX客户端实例
        days: 获取多少天的数据
        save_to_csv: 是否保存到CSV文件
    """
    print(f"开始获取比特币标记价格历史过去{days}天的1分钟K线数据...")
    
    all_data = []
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # 由于OKX API限制每次最多返回100条数据，需要分批获取
    current_time = end_time
    batch_count = 0
    
    while current_time > start_time:
        batch_count += 1
        print(f"正在获取第{batch_count}批标记价格历史数据...")
        
        # 获取标记价格历史K线数据
        result = client.get_history_mark_price_candles(
            inst_id='BTC-USD-SWAP',
            bar='1m',
            limit=100,
            before=int(current_time.timestamp() * 1000)
        )
        
        if not result or result.get('code') != '0':
            print(f"获取标记价格历史数据失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print("没有更多标记价格历史数据")
            break
        
        # 处理数据
        for item in data:
            # OKX返回的标记价格历史数据格式: [ts, o, h, l, c, confirm]
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # 只保留指定时间范围内的数据
            if dt < start_time:
                break
                
            mark_price_history_data = {
                'timestamp': timestamp,
                'datetime': dt,
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'confirm': item[5]
            }
            all_data.append(mark_price_history_data)
        
        # 更新时间戳，获取更早的数据
        if data:
            current_time = datetime.fromtimestamp(int(data[-1][0]) / 1000)
        
        # 避免请求过于频繁
        time.sleep(0.1)
        
        # 如果已经获取到足够早的数据，停止
        if current_time <= start_time:
            break
    
    # 按时间排序（从早到晚）
    all_data.sort(key=lambda x: x['timestamp'])
    
    # 过滤掉超出时间范围的数据
    filtered_data = [item for item in all_data if item['datetime'] >= start_time]
    
    print(f"共获取到 {len(filtered_data)} 条标记价格历史K线数据")
    
    if save_to_csv and filtered_data:
        # 转换为DataFrame
        df = pd.DataFrame(filtered_data)
        
        # 保存到CSV
        filename = f'btc_mark_price_history_1m_{days}days.csv'
        df.to_csv(filename, index=False)
        print(f"标记价格历史数据已保存到 {filename}")
        
        # 显示数据统计
        print(f"\n标记价格历史数据统计:")
        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        print(f"最高价: ${df['high'].max():.2f}")
        print(f"最低价: ${df['low'].min():.2f}")
        print(f"最新价: ${df['close'].iloc[-1]:.2f}")
        
        return df
    
    return pd.DataFrame(filtered_data) if filtered_data else pd.DataFrame()

def get_btc_current_price(client):
    """获取比特币当前价格"""
    result = client.get_ticker('BTC-USDT')
    
    if result and result.get('code') == '0':
        data = result.get('data', [])
        if data:
            ticker = data[0]
            price = float(ticker['last'])
            change_24h = float(ticker['sodUtc0'])
            change_pct = float(ticker['sodUtc8'])
            
            print(f"比特币当前价格: ${price:.2f}")
            print(f"24小时涨跌额: ${change_24h:.2f}")
            print(f"24小时涨跌幅: {change_pct:.2f}%")
            
            return price
    
    print("获取价格失败")
    return None

def main():
    """主函数"""
    print("=== OKX 比特币K线数据获取工具 ===\n")
    
    # 使用配置的API密钥创建客户端
    client = OKXClient(api_key=API_KEY, secret_key=SECRET_KEY, passphrase=PASSPHRASE)
    
    print(f"获取时间范围: {START_DATE} 到 {END_DATE}")
    print(f"正在获取{SYMBOL} 1分钟K线数据...\n")
    
    # 获取K线数据
    df = fetch_btc_kline_data_by_date_range(client, START_DATE, END_DATE)
    
    if not df.empty:
        print(f"\n数据预览 (前5条):")
        print(df.head().to_string(index=False))
        
        print(f"\n数据预览 (后5条):")
        print(df.tail().to_string(index=False))
    else:
        print("未获取到任何数据")

def fetch_btc_kline_data_by_date_range(client, start_date, end_date, save_to_csv=True):
    """
    根据指定日期范围获取比特币1分钟K线数据
    按天循环获取，确保每天数据完整
    
    参数:
        client: OKX客户端实例
        start_date: 开始日期，格式: "YYYY-MM-DD"
        end_date: 结束日期，格式: "YYYY-MM-DD"
        save_to_csv: 是否保存到CSV文件
    """
    print(f"开始获取比特币从 {start_date} 到 {end_date} 的1分钟K线数据...")
    
    # 解析日期
    try:
        start_time = datetime.strptime(start_date, "%Y-%m-%d")
        end_time = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        print(f"日期格式错误: {e}")
        return pd.DataFrame()
    
    print(f"时间范围: {start_time.date()} 到 {end_time.date()}")
    
    all_data = []
    current_date = start_time
    
    # 按天循环获取数据
    while current_date <= end_time:
        print(f"\n正在获取 {current_date.strftime('%Y-%m-%d')} 的数据...")
        
        # 获取单天数据
        day_data = fetch_single_day_data(client, current_date)
        
        if day_data:
            all_data.extend(day_data)
            print(f"✓ {current_date.strftime('%Y-%m-%d')} 获取到 {len(day_data)} 条数据")
        else:
            print(f"✗ {current_date.strftime('%Y-%m-%d')} 未获取到数据")
        
        # 移动到下一天
        current_date += timedelta(days=1)
        
        # 避免请求过于频繁
        time.sleep(0.2)
    
    print(f"\n总共获取到 {len(all_data)} 条K线数据")
    
    if save_to_csv and all_data:
        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        
        # 按时间排序
        df = df.sort_values('DateTime').reset_index(drop=True)
        
        # 保存到CSV
        df.to_csv(OUTPUT_FILENAME, index=False)
        print(f"数据已保存到 {OUTPUT_FILENAME}")
        
        # 显示数据统计
        print(f"\n数据统计:")
        print(f"时间范围: {df['DateTime'].min()} 到 {df['DateTime'].max()}")
        print(f"数据条数: {len(df)} 条")
        print(f"最高价: ${df['High'].max():.2f}")
        print(f"最低价: ${df['Low'].min():.2f}")
        print(f"开盘价: ${df['Open'].iloc[0]:.2f}")
        print(f"收盘价: ${df['Close'].iloc[-1]:.2f}")
        print(f"总成交量: {df['Volume'].sum():.4f} BTC")
        
        return df
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def fetch_single_day_data(client, target_date):
    """
    获取单天的1分钟K线数据
    一天有1440分钟，需要分批获取（每次最多300条）
    
    参数:
        client: OKX客户端实例
        target_date: 目标日期（datetime对象）
    
    返回:
        list: 该天的K线数据列表
    """
    # 计算该天的开始和结束时间戳（毫秒）
    day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    print(f"  目标日期: {target_date.strftime('%Y-%m-%d')}")
    print(f"  目标时间范围: {day_start} - {day_end}")
    
    day_data = []
    batch_count = 0
    current_before = None  # 用于分页的时间戳
    
    # 一天最多1440条数据，每次获取300条，最多需要5次
    while batch_count < 10:  # 增加批次限制
        batch_count += 1
        
        # 获取K线数据
        if current_before is None:
            # 第一次获取最新数据
            result = client.get_kline_data(
                inst_id=SYMBOL,
                bar=BAR_SIZE,
                limit=300
            )
        else:
            # 后续获取，使用before参数获取更早的数据
            result = client.get_kline_data(
                inst_id=SYMBOL,
                bar=BAR_SIZE,
                limit=300,
                before=current_before
            )
        
        if not result or result.get('code') != '0':
            print(f"  批次{batch_count}获取失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print(f"  批次{batch_count}无数据")
            break
        
        # print(f"  批次{batch_count}原始数据条数: {len(data)}")
        # if data:
        #     first_time = datetime.fromtimestamp(int(data[0][0]) / 1000)
        #     last_time = datetime.fromtimestamp(int(data[-1][0]) / 1000)
        #     print(f"  批次{batch_count}时间范围: {first_time} - {last_time}")
        
        batch_valid_data = []
        earliest_timestamp = None
        found_target_date = False
        
        # 创建已存在时间的集合，避免重复
        existing_times = {item['DateTime'] for item in day_data}
        
        for item in data:
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # 记录本批次最早的时间戳
            if earliest_timestamp is None or timestamp < earliest_timestamp:
                earliest_timestamp = timestamp
            
            # 只保留目标日期内且不重复的数据
            if day_start <= dt <= day_end and dt_str not in existing_times:
                found_target_date = True
                kline_data = {
                    'DateTime': dt_str,
                    'Open': float(item[1]),
                    'High': float(item[2]),
                    'Low': float(item[3]),
                    'Close': float(item[4]),
                    'Volume': float(item[5])
                }
                batch_valid_data.append(kline_data)
                existing_times.add(dt_str)  # 添加到已存在集合中
            
            # 如果数据时间早于目标日期，停止处理
            elif dt < day_start:
                print(f"  已获取到{target_date.strftime('%Y-%m-%d')}之前的数据，停止获取")
                break
        
        day_data.extend(batch_valid_data)
        print(f"  批次{batch_count}: 获取{len(batch_valid_data)}条有效数据")
        
        # 更新分页时间戳
        current_before = earliest_timestamp
        
        # 如果已经获取到目标日期之前的数据，停止
        if earliest_timestamp and datetime.fromtimestamp(earliest_timestamp / 1000) < day_start:
            print(f"  已到达目标日期之前，停止获取")
            break
        
        # 如果本批次没有有效数据且已经找到过目标日期的数据，停止
        if not batch_valid_data and found_target_date:
            print(f"  已获取完目标日期的所有数据")
            break
        
        # 避免请求过于频繁
        time.sleep(0.1)
    
    # 按时间排序（从早到晚）
    day_data.sort(key=lambda x: x['DateTime'])
    
    return day_data

def fetch_index_history_simple(start_date=None, end_date=None):
    """
    简单获取指定日期范围的比特币指数历史K线数据
    
    参数:
        start_date: 开始日期，格式: "YYYY-MM-DD"，如果为None则使用配置区域的START_DATE
        end_date: 结束日期，格式: "YYYY-MM-DD"，如果为None则使用配置区域的END_DATE
    """
    # 如果没有传入参数，使用配置区域的日期
    if start_date is None:
        start_date = START_DATE
    if end_date is None:
        end_date = END_DATE
    
    print(f"=== 获取比特币指数历史K线数据 ===")
    print(f"时间范围: {start_date} 到 {end_date}\n")
    
    # 创建客户端
    client = OKXClient()
    
    # 解析日期并转换为时间戳
    start_time = datetime.strptime(start_date, "%Y-%m-%d")
    end_time = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    start_timestamp_ms = int(start_time.timestamp() * 1000)
    end_timestamp_ms = int(end_time.timestamp() * 1000)
    
    print(f"开始时间戳: {start_timestamp_ms} ({start_time})")
    print(f"结束时间戳: {end_timestamp_ms} ({end_time})\n")
    
    all_data = []
    batch_count = 0
    current_after = None  # 用于分页
    
    while True:
        batch_count += 1
        
        # 获取数据
        if current_after is None:
            # 第一次请求，不带参数，获取最新数据
            result = client.get_history_index_candles(
                inst_id='BTC-USD',
                bar='1m',
                limit=100
            )
        else:
            # 后续请求，使用after参数获取更早的数据
            result = client.get_history_index_candles(
                inst_id='BTC-USD',
                bar='1m',
                limit=100,
                after=current_after
            )
        
        if not result or result.get('code') != '0':
            print(f"批次{batch_count}获取失败: {result}")
            break
        
        data = result.get('data', [])
        if not data:
            print(f"批次{batch_count}无数据")
            break
        
        # 处理数据
        batch_data = []
        earliest_timestamp = None
        
        # 打印第一条数据的时间信息（调试用）
        if batch_count == 1 and data:
            first_ts = int(data[0][0])
            first_dt = datetime.fromtimestamp(first_ts / 1000)
            print(f"第一批数据的第一条时间: {first_dt}")
        
        for item in data:
            timestamp = int(item[0])
            dt = datetime.fromtimestamp(timestamp / 1000)
            
            # 记录最早的时间戳
            if earliest_timestamp is None or timestamp < earliest_timestamp:
                earliest_timestamp = timestamp
            
            # 只保留时间范围内的数据
            if start_timestamp_ms <= timestamp <= end_timestamp_ms:
                kline_data = {
                    'Timestamp': timestamp,
                    'DateTime': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': float(item[1]),
                    'High': float(item[2]),
                    'Low': float(item[3]),
                    'Close': float(item[4]),
                    'Confirm': item[5]
                }
                batch_data.append(kline_data)
        
        if batch_data:
            all_data.extend(batch_data)
            print(f"批次{batch_count}: 获取{len(batch_data)}条有效数据，累计{len(all_data)}条")
        else:
            print(f"批次{batch_count}: 无有效数据")
        
        # 更新after参数为本批次最早的时间戳
        current_after = earliest_timestamp
        
        # 如果已经获取到开始时间之前的数据，停止
        if earliest_timestamp and earliest_timestamp < start_timestamp_ms:
            print(f"已获取到开始时间之前的数据，停止")
            break
        
        # 避免请求过快
        time.sleep(0.2)
    
    # 按时间排序
    all_data.sort(key=lambda x: x['Timestamp'])
    
    print(f"\n总共获取到 {len(all_data)} 条数据")
    
    if all_data:
        # 转换为DataFrame
        df = pd.DataFrame(all_data)
        
        # 保存到CSV
        filename = "okx_btc_1m.csv"
        df.to_csv(filename, index=False)
        print(f"数据已保存到 {filename}")
        
        # 显示统计信息
        print(f"\n数据统计:")
        print(f"时间范围: {df['DateTime'].min()} 到 {df['DateTime'].max()}")
        print(f"数据条数: {len(df)} 条")
        print(f"预期条数: 约{int((end_timestamp_ms - start_timestamp_ms) / 60000)}条 (3天×24小时×60分钟)")
        print(f"最高价: ${df['High'].max():.2f}")
        print(f"最低价: ${df['Low'].min():.2f}")
        print(f"开盘价: ${df['Open'].iloc[0]:.2f}")
        print(f"收盘价: ${df['Close'].iloc[-1]:.2f}")
        
        # 显示前后几条数据
        print(f"\n前5条数据:")
        print(df.head()[['DateTime', 'Open', 'High', 'Low', 'Close']].to_string(index=False))
        
        print(f"\n后5条数据:")
        print(df.tail()[['DateTime', 'Open', 'High', 'Low', 'Close']].to_string(index=False))
        
        return df
    
    return pd.DataFrame()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "spot":
        # 获取现货K线数据
        main()
    else:
        # 默认获取指数历史数据
        fetch_index_history_simple() 