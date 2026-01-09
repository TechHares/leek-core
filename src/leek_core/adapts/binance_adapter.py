#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
币安 API 桥接类
整合所有币安 API 接口调用，提供统一的接口访问
"""

import time
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None
    BinanceAPIException = Exception

from leek_core.models import TimeFrame, TradeInsType, data
from leek_core.utils import get_logger, retry, rate_limit

logger = get_logger(__name__)


class BinanceAdapter:
    """
    币安 API 适配器类
    整合交易、市场数据等 API 接口，提供统一的适配接口
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", 
                 testnet: bool = False, proxy: Optional[str] = None):
        """
        初始化币安适配器类
        
        Args:
            api_key: API密钥（可选，仅公开市场数据不需要）
            secret_key: 密钥（可选，仅公开市场数据不需要）
            testnet: 是否使用测试网
            proxy: 代理设置
        """
        if Client is None:
            raise ImportError("python-binance 库未安装，请运行: pip install python-binance")
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.proxy = proxy
        
        # 初始化 API 客户端
        # 使用位置参数传递 api_key 和 api_secret，确保正确设置请求头
        requests_params = None
        if proxy:
            requests_params = {'proxies': {'https': proxy, 'http': proxy}}
        
        # 允许无 API 密钥初始化（仅用于公开市场数据）
        self.client = Client(api_key=api_key or "", api_secret=secret_key or "", requests_params=requests_params, testnet=testnet)
    
    # ==================== Market Data API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="market_data")  # 币安限速：1200次/分钟
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        获取订单簿数据
        
        Args:
            symbol: 交易对符号（如 BTCUSDT）
            limit: 订单簿深度
            
        Returns:
            Dict: 订单簿数据
        """
        try:
            orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                "code": "0",
                "data": [{
                    "bids": [[bid[0], bid[1]] for bid in orderbook.get("bids", [])],
                    "asks": [[ask[0], ask[1]] for ask in orderbook.get("asks", [])]
                }]
            }
        except BinanceAPIException as e:
            logger.error(f"获取订单簿失败: {e}")
            return {"code": str(getattr(e, 'status_code', '-1')), "msg": str(e)}
        except Exception as e:
            logger.error(f"获取订单簿异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="market_data")
    def get_exchange_info(self, symbol: str = None) -> Dict:
        """
        获取交易对信息
        
        Args:
            symbol: 交易对符号（可选，不传则返回所有交易对）
            
        Returns:
            Dict: 交易对信息
        """
        try:
            if symbol:
                exchange_info = self.client.get_exchange_info()
                # 从所有交易对中筛选指定的交易对
                for s in exchange_info.get("symbols", []):
                    if s["symbol"] == symbol:
                        return {"code": "0", "data": s}
                return {"code": "-1", "msg": f"交易对 {symbol} 不存在"}
            else:
                exchange_info = self.client.get_exchange_info()
                return {"code": "0", "data": exchange_info}
        except BinanceAPIException as e:
            logger.error(f"获取交易对信息失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"获取交易对信息异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Trade API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="trade")  # 币安限速：1200次/分钟
    def place_order(self, symbol: str, side: str, order_type: str, 
                   quantity: str = None, price: str = None, 
                   client_order_id: str = None, **kwargs) -> Dict:
        """
        下单
        
        Args:
            symbol: 交易对符号
            side: 订单方向 (BUY/SELL)
            order_type: 订单类型 (LIMIT/MARKET)
            quantity: 委托数量
            price: 委托价格（限价单必填）
            client_order_id: 客户端订单ID (可选)
            **kwargs: 其他可选参数
            
        Returns:
            Dict: 下单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
            }
            
            if order_type == "LIMIT":
                if not price:
                    raise ValueError("限价单必须提供价格")
                params["price"] = price
                params["timeInForce"] = kwargs.get("timeInForce", "GTC")  # Good Till Cancel
            
            if quantity:
                params["quantity"] = quantity
            
            if client_order_id:
                params["newClientOrderId"] = client_order_id
            
            # 添加其他参数
            params.update(kwargs)
            logger.info(f"下单参数: {params}")
            result = self.client.create_order(**params)
            return {
                "code": "0",
                "data": [result]
            }
        except BinanceAPIException as e:
            logger.error(f"下单失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"下单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="trade")
    def cancel_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Dict:
        """
        撤单
        
        Args:
            symbol: 交易对符号
            order_id: 订单ID (可选，与client_order_id二选一)
            client_order_id: 客户端订单ID (可选，与order_id二选一)
            
        Returns:
            Dict: 撤单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"symbol": symbol}
            if order_id:
                params["orderId"] = order_id
            if client_order_id:
                params["origClientOrderId"] = client_order_id
            
            result = self.client.cancel_order(**params)
            return {
                "code": "0",
                "data": [result]
            }
        except BinanceAPIException as e:
            logger.error(f"撤单失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"撤单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="trade")
    def get_order(self, symbol: str, order_id: str = None, client_order_id: str = None) -> Dict:
        """
        查询单个订单详情
        
        Args:
            symbol: 交易对符号
            order_id: 订单ID (可选，与client_order_id二选一)
            client_order_id: 客户端订单ID (可选，与order_id二选一)
            
        Returns:
            Dict: 订单信息
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"symbol": symbol}
            if order_id:
                params["orderId"] = order_id
            if client_order_id:
                params["origClientOrderId"] = client_order_id
            
            result = self.client.get_order(**params)
            return {
                "code": "0",
                "data": [result]
            }
        except BinanceAPIException as e:
            logger.error(f"查询订单失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"查询订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="trade")
    def get_open_orders(self, symbol: str = None) -> Dict:
        """
        查询待成交订单列表
        
        Args:
            symbol: 交易对符号 (可选，不传则返回所有交易对的待成交订单)
            
        Returns:
            Dict: 订单列表
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            if symbol:
                orders = self.client.get_open_orders(symbol=symbol)
            else:
                orders = self.client.get_open_orders()
            
            return {
                "code": "0",
                "data": orders if isinstance(orders, list) else [orders]
            }
        except BinanceAPIException as e:
            logger.error(f"查询待成交订单失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"查询待成交订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="trade")
    def get_my_trades(self, symbol: str, order_id: str = None, limit: int = 100) -> Dict:
        """
        查询个人成交记录
        
        Args:
            symbol: 交易对符号
            order_id: 订单ID (可选，用于查询指定订单的成交记录)
            limit: 返回数量限制
            
        Returns:
            Dict: 成交记录列表，包含 commission（手续费）等字段
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"symbol": symbol, "limit": limit}
            if order_id:
                params["orderId"] = order_id
            
            trades = self.client.get_my_trades(**params)
            
            return {
                "code": "0",
                "data": trades if isinstance(trades, list) else [trades]
            }
        except BinanceAPIException as e:
            logger.error(f"查询成交记录失败: {e}")
            return {"code": str(e.status_code), "msg": str(e)}
        except Exception as e:
            logger.error(f"查询成交记录异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def build_symbol(base_asset: str, quote_currency: str) -> str:
        """
        构建币安交易对符号
        
        Args:
            base_asset: 基础资产（如 BTC）
            quote_currency: 计价货币（如 USDT）
            
        Returns:
            str: 交易对符号（如 BTCUSDT）
        """
        return f"{base_asset}{quote_currency}"
    
    @staticmethod
    def get_binance_timeframe(timeframe: Union[TimeFrame, str]) -> Optional[str]:
        """
        获取币安格式的时间周期字符串
        
        Args:
            timeframe: 时间周期
            
        Returns:
            str: 币安格式的时间周期
        """
        # 币安 TimeFrame 映射
        BINANCE_TIMEFRAME_MAP = {
            TimeFrame.M1: "1m", TimeFrame.M3: "3m", TimeFrame.M5: "5m",
            TimeFrame.M15: "15m", TimeFrame.M30: "30m", TimeFrame.H1: "1h",
            TimeFrame.H2: "2h", TimeFrame.H4: "4h", TimeFrame.H6: "6h",
            TimeFrame.H12: "12h", TimeFrame.D1: "1d", TimeFrame.W1: "1w",
            TimeFrame.MON1: "1M",
        }
        
        if isinstance(timeframe, str):
            try:
                tf_enum = TimeFrame(timeframe)
                return BINANCE_TIMEFRAME_MAP.get(tf_enum)
            except ValueError:
                # 如果已经是币安格式，直接返回
                if timeframe in ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1M"]:
                    return timeframe
                logger.error(f"无效的时间周期字符串: {timeframe}")
                return None
        else:
            return BINANCE_TIMEFRAME_MAP.get(timeframe)
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="market_data")
    def get_klines(self, symbol: str, interval: str, start_time: int = None, 
                   end_time: int = None, limit: int = 500) -> Dict:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号（如 BTCUSDT）
            interval: 时间周期（如 1m, 5m, 1h, 1d）
            start_time: 开始时间（毫秒时间戳，可选）
            end_time: 结束时间（毫秒时间戳，可选）
            limit: 返回数量限制，默认500，最大1000
            
        Returns:
            Dict: K线数据列表，格式为:
                {
                    "code": "0",
                    "data": [
                        [open_time, open, high, low, close, volume, close_time, 
                         quote_volume, trades, taker_buy_base, taker_buy_quote, ignore],
                        ...
                    ]
                }
        """
        try:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(limit, 1000)  # 币安最大限制1000
            }
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time
            
            klines = self.client.get_klines(**params)
            return {
                "code": "0",
                "data": klines
            }
        except BinanceAPIException as e:
            logger.error(f"获取K线数据失败: {e}")
            return {"code": str(getattr(e, 'status_code', '-1')), "msg": str(e)}
        except Exception as e:
            logger.error(f"获取K线数据异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="market_data")
    def get_all_symbols(self, quote_currency: str = "USDT") -> List[str]:
        """
        获取所有可用的交易对基础资产列表
        
        Args:
            quote_currency: 计价货币筛选（如 USDT, BTC），默认为 USDT
            
        Returns:
            List[str]: 基础资产符号列表（如 ["BTC", "ETH", ...]）
        """
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = set()
            for s in exchange_info.get("symbols", []):
                # 只获取指定计价货币且状态为 TRADING 的交易对
                if s.get("quoteAsset") == quote_currency and s.get("status") == "TRADING":
                    symbols.add(s.get("baseAsset"))
            return sorted(list(symbols))
        except BinanceAPIException as e:
            logger.error(f"获取交易对列表失败: {e}")
            return []
        except Exception as e:
            logger.error(f"获取交易对列表异常: {e}")
            return []
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=1200, time_window=60.0, group="market_data")
    def get_ticker_price(self, symbol: str = None) -> Dict:
        """
        获取最新价格
        
        Args:
            symbol: 交易对符号（可选，不传则返回所有交易对）
            
        Returns:
            Dict: 价格数据
        """
        try:
            if symbol:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return {"code": "0", "data": [ticker]}
            else:
                tickers = self.client.get_all_tickers()
                return {"code": "0", "data": tickers}
        except BinanceAPIException as e:
            logger.error(f"获取价格失败: {e}")
            return {"code": str(getattr(e, 'status_code', '-1')), "msg": str(e)}
        except Exception as e:
            logger.error(f"获取价格异常: {e}")
            return {"code": "-1", "msg": str(e)}

    def check_api_available(self) -> bool:
        """
        检查 API 是否可用
        
        Returns:
            bool: API 是否可用
        """
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"API 检查失败: {e}")
            return False
