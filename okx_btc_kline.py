import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import hmac
import hashlib
import base64

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
        timestamp = datetime.utcnow().isoformat()[:-3] + 'Z'
        
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
                'timestamp': timestamp,
                'datetime': dt,
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5]),
                'volume_ccy': float(item[6]),
                'volume_ccy_quote': float(item[7]),
                'confirm': item[8]
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
    all_data.sort(key=lambda x: x['timestamp'])
    
    # 过滤掉超出时间范围的数据
    filtered_data = [item for item in all_data if item['datetime'] >= start_time]
    
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
        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
        print(f"最高价: ${df['high'].max():.2f}")
        print(f"最低价: ${df['low'].min():.2f}")
        print(f"最新价: ${df['close'].iloc[-1]:.2f}")
        print(f"总成交量: {df['volume'].sum():.2f} BTC")
        
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
    
    # 使用提供的API密钥创建客户端
    api_key = "a16ccaed-3adc-4a37-8484-4d9dd0cb940c"
    secret_key = "8EE34D5B82E1ED5FA441021E20FEC08C"
    passphrase = ""  # 如果有passphrase请填写
    
    client = OKXClient(api_key=api_key, secret_key=secret_key, passphrase=passphrase)
    
    # 获取当前价格
    print("1. 获取比特币当前价格:")
    get_btc_current_price(client)
    
    print("\n" + "="*50 + "\n")
    
    # 获取产品信息
    print("2. 获取BTC-USDT产品信息:")
    instruments = client.get_instruments('SPOT')
    if instruments and instruments.get('code') == '0':
        btc_instruments = [inst for inst in instruments.get('data', []) if inst['instId'] == 'BTC-USDT']
        if btc_instruments:
            inst = btc_instruments[0]
            print(f"产品ID: {inst['instId']}")
            print(f"基础货币: {inst['baseCcy']}")
            print(f"计价货币: {inst['quoteCcy']}")
            print(f"最小下单量: {inst['minSz']}")
            print(f"价格精度: {inst['tickSz']}")
            print(f"数量精度: {inst['lotSz']}")
            print(f"产品状态: {inst['state']}")
    
    print("\n" + "="*50 + "\n")
    
    # 选择数据类型
    print("3. 选择要获取的数据类型:")
    print("   1) 现货K线数据 (BTC-USDT)")
    print("   2) 指数历史K线数据 (BTC-USD)")
    print("   3) 标记价格K线数据 (BTC-USD-SWAP)")
    print("   4) 标记价格历史K线数据 (BTC-USD-SWAP)")
    
    try:
        choice = int(input("请选择 (1-4): ") or "1")
    except ValueError:
        choice = 1
        print("输入无效，使用默认选项1")
    
    try:
        days = int(input("请输入要获取多少天的数据 (1-30): ") or "7")
        days = max(1, min(days, 30))  # 限制在1-30天之间
    except ValueError:
        days = 7
        print("输入无效，使用默认值7天")
    
    df = pd.DataFrame()
    
    if choice == 1:
        print(f"\n获取现货K线数据 (BTC-USDT):")
        df = fetch_btc_kline_data(client, days=days)
    elif choice == 2:
        print(f"\n获取指数历史K线数据 (BTC-USD):")
        df = fetch_btc_index_history_data(client, days=days)
    elif choice == 3:
        print(f"\n获取标记价格K线数据 (BTC-USD-SWAP):")
        df = fetch_mark_price_data(client, days=days)
    elif choice == 4:
        print(f"\n获取标记价格历史K线数据 (BTC-USD-SWAP):")
        df = fetch_mark_price_history_data(client, days=days)
    
    if not df.empty:
        print(f"\n数据预览 (前5条):")
        print(df.head().to_string(index=False))
        
        print(f"\n数据预览 (后5条):")
        print(df.tail().to_string(index=False))

if __name__ == "__main__":
    main() 