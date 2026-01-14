#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gate.io API 桥接类
整合所有 Gate.io API 接口调用，提供统一的接口访问
直接使用 HTTP 请求，不使用 SDK
"""

import hashlib
import hmac
import json
import time
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlencode

import requests

from leek_core.models import TimeFrame, TradeInsType
from leek_core.utils import get_logger, retry, rate_limit

logger = get_logger(__name__)


class GateAdapter:
    """
    Gate.io API 适配器类
    整合交易、市场数据等 API 接口，提供统一的适配接口
    直接使用 HTTP 请求，不使用 SDK
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", 
                 testnet: bool = False, proxy: Optional[str] = None):
        """
        初始化 Gate.io 适配器类
        
        Args:
            api_key: API密钥
            secret_key: 密钥
            testnet: 是否使用测试网
            proxy: 代理设置
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.proxy = proxy
        
        # API 基础 URL
        if testnet:
            self.base_url = "https://fx-api-testnet.gateio.ws"
        else:
            self.base_url = "https://api.gateio.ws"
        
        # 设置代理
        self.session = requests.Session()
        if proxy:
            self.session.proxies = {
                'http': proxy,
                'https': proxy
            }
    
    def _sign(self, method: str, path: str, query_string: str, payload_string: str, timestamp: str) -> str:
        """
        Gate.io API 签名算法
        签名字符串 = Request Method + "\n" + Request URL + "\n" + Query String + "\n" + HexEncode(SHA512(Request Payload)) + "\n" + Timestamp
        
        Args:
            method: HTTP 方法（大写）
            path: 请求路径（不包含服务地址）
            query_string: 查询字符串（URL 参数，不进行 URL 编码）
            payload_string: 请求体字符串（JSON）
            timestamp: 时间戳字符串
            
        Returns:
            str: 签名字符串（hex）
        """
        # 计算请求体的 SHA512 哈希值
        payload_hash = hashlib.sha512(payload_string.encode('utf-8')).hexdigest()
        
        # 构建签名字符串
        # Request Method + "\n" + Request URL + "\n" + Query String + "\n" + HexEncode(SHA512(Request Payload)) + "\n" + Timestamp
        message = f"{method}\n{path}\n{query_string}\n{payload_hash}\n{timestamp}"
        
        # 使用 HMAC-SHA512 签名
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        logger.debug(f"Gate.io 签名: method={method}, path={path}, query_string={query_string}, payload_hash={payload_hash}, timestamp={timestamp}, signature={signature}")
        return signature
    
    def _request(self, method: str, path: str, params: Dict = None, data: Dict = None) -> Dict:
        """
        发送 HTTP 请求
        
        Args:
            method: HTTP 方法 (GET, POST, DELETE)
            path: API 路径
            params: URL 参数
            data: 请求体数据
            
        Returns:
            Dict: API 响应
        """
        url = f"{self.base_url}{path}"
        
        # 构建查询字符串（不进行 URL 编码，保持原始格式）
        # Gate.io 要求 Query String 不使用 URL 编码，顺序要和实际请求一致
        query_string = ""
        if params:
            # 使用 & 连接参数，不进行 URL 编码
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        
        # 构建请求体字符串
        payload_string = ""
        if data:
            payload_string = json.dumps(data, separators=(',', ':'))
        
        # 生成时间戳
        timestamp = str(int(time.time()))
        
        # 生成签名（传入 method 和 path）
        signature = self._sign(method.upper(), path, query_string, payload_string, timestamp)
        
        # 构建请求头
        headers = {
            'KEY': self.api_key,
            'Timestamp': timestamp,
            'SIGN': signature,
            'Content-Type': 'application/json'
        }
        
        # 发送请求
        try:
            if method == 'GET':
                response = self.session.get(url, params=params, headers=headers, timeout=10)
            elif method == 'POST':
                # POST 请求需要使用 data=payload_string 而不是 json=data
                # 确保发送的 JSON 字符串和签名的 JSON 字符串完全一致
                if payload_string:
                    response = self.session.post(url, params=params, data=payload_string, headers=headers, timeout=10)
                else:
                    response = self.session.post(url, params=params, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params, headers=headers, timeout=10)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")
            
            # 解析响应（先尝试解析 JSON，不管状态码）
            try:
                result = response.json()
            except json.JSONDecodeError:
                result = None
            
            # 检查 HTTP 状态码
            if not response.ok:
                # 尝试从响应中获取详细错误信息
                if result and isinstance(result, dict):
                    error_msg = result.get("message") or result.get("label") or str(result)
                    logger.error(f"Gate.io API 错误: status={response.status_code}, response={result}")
                    return {"code": str(response.status_code), "msg": error_msg}
                else:
                    logger.error(f"Gate.io API 错误: status={response.status_code}, response={response.text}")
                    return {"code": str(response.status_code), "msg": response.text}
            
            # Gate.io API 返回格式检查
            # 成功响应通常是订单对象（包含 id, text, status 等字段）或订单列表
            if isinstance(result, dict):
                # 如果有 label 字段且为 "SUCCESS"，表示成功（某些接口）
                if result.get("label") == "SUCCESS":
                    return {"code": "0", "data": result.get("id") if "id" in result else result}
                # 如果有 label 字段且不为 "SUCCESS"，表示失败
                elif "label" in result:
                    return {"code": "-1", "msg": result.get("message", result.get("label", "未知错误"))}
                # 如果有 id 字段，表示是订单对象
                elif "id" in result:
                    return {"code": "0", "data": result}
                # 如果有 message 字段，可能是错误信息
                elif "message" in result:
                    return {"code": "-1", "msg": result.get("message")}
                # 其他情况，当作成功处理
                else:
                    return {"code": "0", "data": result}
            elif isinstance(result, list):
                return {"code": "0", "data": result}
            else:
                return {"code": "0", "data": result}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP 请求失败: {e}")
            return {"code": "-1", "msg": str(e)}
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}")
            return {"code": "-1", "msg": f"JSON 解析失败: {e}"}
        except Exception as e:
            logger.error(f"请求异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Market Data API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="market_data")  # Gate.io 限速：100次/秒
    def get_orderbook(self, currency_pair: str, limit: int = 20, interval: str = "0") -> Dict:
        """
        获取订单簿数据
        
        Args:
            currency_pair: 交易对符号（如 BTC_USDT）
            limit: 订单簿深度
            interval: 价格精度间隔（0 表示不聚合）
            
        Returns:
            Dict: 订单簿数据
        """
        try:
            params = {
                "currency_pair": currency_pair,
                "limit": limit,
                "interval": interval
            }
            result = self._request("GET", "/api/v4/spot/order_book", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", {})
                return {
                    "code": "0",
                    "data": [{
                        "bids": data.get("bids", []),
                        "asks": data.get("asks", [])
                    }]
                }
            return result
        except Exception as e:
            logger.error(f"获取订单簿异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="market_data")
    def get_currency_pairs(self, currency_pair: str = None) -> Dict:
        """
        获取交易对信息
        
        Args:
            currency_pair: 交易对符号（可选，不传则返回所有交易对）
            
        Returns:
            Dict: 交易对信息
        """
        try:
            params = {}
            if currency_pair:
                params["currency_pair"] = currency_pair
            
            result = self._request("GET", "/api/v4/spot/currency_pairs", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                if currency_pair and isinstance(data, list) and len(data) > 0:
                    # 返回单个交易对信息
                    return {"code": "0", "data": data[0]}
                elif currency_pair:
                    # 未找到指定交易对
                    return {"code": "-1", "msg": f"交易对 {currency_pair} 不存在"}
                else:
                    # 返回所有交易对
                    return {"code": "0", "data": data}
            return result
        except Exception as e:
            logger.error(f"获取交易对信息异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Trade API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="trade")  # Gate.io 限速：100次/秒
    def place_order(self, currency_pair: str, side: str, amount: str = None, 
                   price: str = None, text: str = None, order_type: str = None, **kwargs) -> Dict:
        """
        下单
        
        Args:
            currency_pair: 交易对符号
            side: 订单方向 (buy/sell)
            amount: 委托数量
            price: 委托价格（限价单必填）
            text: 客户端订单ID (可选，通过 text 参数传递)
            order_type: 订单类型 (limit/market)，如果不传则根据 price 是否为空判断
            **kwargs: 其他可选参数
            
        Returns:
            Dict: 下单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            data = {
                "currency_pair": currency_pair,
                "side": side,
                "account": "spot",  # 指定账户类型为现货
            }
            
            if amount:
                data["amount"] = amount
            
            # 根据 order_type 参数或 price 是否为空来判断订单类型
            is_limit_order = order_type == "limit" if order_type else bool(price)
            
            if is_limit_order:
                if price:
                    data["price"] = price
                data["type"] = "limit"
                data["time_in_force"] = "gtc"  # Good Till Cancelled（仅限价单支持）
            else:
                # 市价单需要设置 time_in_force 为 ioc（Immediate or Cancel）
                data["type"] = "market"
                data["time_in_force"] = "ioc"  # 市价单使用 IOC
            
            if text:
                # Gate.io 的 text 参数需要以 "t-" 开头
                if not str(text).startswith("t-"):
                    data["text"] = f"t-{text}"
                else:
                    data["text"] = text
            
            # 添加其他参数（排除 order_type，因为 Gate.io 使用 type）
            for k, v in kwargs.items():
                if k != "order_type":
                    data[k] = v
            
            logger.info(f"Gate.io 下单请求数据: {data}")
            result = self._request("POST", "/api/v4/spot/orders", data=data)
            
            if result.get("code") == "0":
                # Gate.io 返回的订单信息
                order_data = result.get("data", {})
                if isinstance(order_data, dict):
                    return {"code": "0", "data": [order_data]}
                else:
                    return {"code": "0", "data": [{"id": order_data}]}
            return result
        except Exception as e:
            logger.error(f"下单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="trade")
    def cancel_order(self, currency_pair: str, order_id: str = None, text: str = None) -> Dict:
        """
        撤单
        
        Args:
            currency_pair: 交易对符号
            order_id: 订单ID (可选，与text二选一)
            text: 客户端订单ID (可选，与order_id二选一)
            
        Returns:
            Dict: 撤单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"currency_pair": currency_pair}
            if order_id:
                params["order_id"] = order_id
            if text:
                params["text"] = text
            
            result = self._request("DELETE", "/api/v4/spot/orders", params=params)
            
            if result.get("code") == "0":
                order_data = result.get("data", {})
                if isinstance(order_data, dict):
                    return {"code": "0", "data": [order_data]}
                else:
                    return {"code": "0", "data": [{"id": order_data}]}
            return result
        except Exception as e:
            logger.error(f"撤单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="trade")
    def get_order(self, currency_pair: str, order_id: str = None, text: str = None) -> Dict:
        """
        查询单个订单详情
        
        Args:
            currency_pair: 交易对符号
            order_id: 订单ID (可选，与text二选一)
            text: 客户端订单ID (可选，与order_id二选一)
            
        Returns:
            Dict: 订单信息
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"currency_pair": currency_pair}
            if order_id:
                params["order_id"] = order_id
            if text:
                params["text"] = text
            
            result = self._request("GET", "/api/v4/spot/orders", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                if isinstance(data, list) and len(data) > 0:
                    return {"code": "0", "data": data}
                elif isinstance(data, dict):
                    return {"code": "0", "data": [data]}
                else:
                    return {"code": "-1", "msg": "订单不存在"}
            return result
        except Exception as e:
            logger.error(f"查询订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="trade")
    def get_open_orders(self, currency_pair: str = None, page: int = 1, limit: int = 100) -> Dict:
        """
        查询待成交订单列表
        
        Args:
            currency_pair: 交易对符号 (可选，不传则返回所有交易对的待成交订单)
            page: 页码
            limit: 每页数量
            
        Returns:
            Dict: 订单列表
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {
                "status": "open",
                "page": page,
                "limit": limit
            }
            if currency_pair:
                params["currency_pair"] = currency_pair
            
            result = self._request("GET", "/api/v4/spot/orders", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                return {"code": "0", "data": data if isinstance(data, list) else [data]}
            return result
        except Exception as e:
            logger.error(f"查询待成交订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Futures Trade API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_contract(self, settle: str, contract: str) -> Dict:
        """
        获取合约信息
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (如 BTC_USDT)
            
        Returns:
            Dict: 合约信息
        """
        try:
            result = self._request("GET", f"/api/v4/futures/{settle}/contracts/{contract}")
            if result.get("code") == "0":
                return {"code": "0", "data": result.get("data", {})}
            return result
        except Exception as e:
            logger.error(f"获取合约信息异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_orderbook(self, settle: str, contract: str, limit: int = 20) -> Dict:
        """
        获取合约订单簿
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (如 BTC_USDT)
            limit: 订单簿深度
            
        Returns:
            Dict: 订单簿数据
        """
        try:
            params = {"contract": contract, "limit": limit}
            result = self._request("GET", f"/api/v4/futures/{settle}/order_book", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", {})
                return {
                    "code": "0",
                    "data": [{
                        "bids": data.get("bids", []),
                        "asks": data.get("asks", [])
                    }]
                }
            return result
        except Exception as e:
            logger.error(f"获取合约订单簿异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def update_futures_leverage(self, settle: str, contract: str, leverage: int, cross_leverage_limit: int = None) -> Dict:
        """
        更新合约仓位杠杆
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (如 BTC_USDT)
            leverage: 杠杆倍数，0 表示全仓，正数表示逐仓杠杆倍数
            cross_leverage_limit: 全仓杠杆倍数限制（仅当 leverage=0 时使用）
            
        Returns:
            Dict: 仓位信息
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"leverage": str(leverage)}
            if cross_leverage_limit is not None:
                params["cross_leverage_limit"] = str(cross_leverage_limit)
            
            logger.info(f"Gate.io 更新合约杠杆: contract={contract}, leverage={leverage}")
            result = self._request("POST", f"/api/v4/futures/{settle}/positions/{contract}/leverage", params=params)
            
            if result.get("code") == "0":
                return {"code": "0", "data": result.get("data", {})}
            return result
        except Exception as e:
            logger.error(f"更新合约杠杆异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def place_futures_order(self, settle: str, contract: str, size: int, price: str = "0",
                           tif: str = "gtc", text: str = None, close: bool = False,
                           reduce_only: bool = False, **kwargs) -> Dict:
        """
        合约下单
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (如 BTC_USDT)
            size: 交易数量，正数为买入，负数为卖出，平仓设置为 0
            price: 委托价，价格为 0 且 tif 为 ioc 代表市价委托
            tif: Time in force 策略 (gtc/ioc/poc/fok)，市价单只支持 ioc
            text: 客户端订单ID
            close: 设置为 True 时执行平仓操作，size 应设置为 0
            reduce_only: 设置为 True 时，为只减仓委托
            **kwargs: 其他可选参数
            
        Returns:
            Dict: 下单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            data = {
                "contract": contract,
                "size": size,
                "price": price,
                "tif": tif,
            }
            
            if close:
                data["close"] = close
            
            if reduce_only:
                data["reduce_only"] = reduce_only
            
            if text:
                # Gate.io 的 text 参数需要以 "t-" 开头
                if not str(text).startswith("t-"):
                    data["text"] = f"t-{text}"
                else:
                    data["text"] = text
            
            # 添加其他参数
            for k, v in kwargs.items():
                if k not in data:
                    data[k] = v
            
            logger.info(f"Gate.io 合约下单请求数据: {data}")
            result = self._request("POST", f"/api/v4/futures/{settle}/orders", data=data)
            
            if result.get("code") == "0":
                order_data = result.get("data", {})
                if isinstance(order_data, dict):
                    return {"code": "0", "data": [order_data]}
                else:
                    return {"code": "0", "data": [{"id": order_data}]}
            return result
        except Exception as e:
            logger.error(f"合约下单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def cancel_futures_order(self, settle: str, order_id: str) -> Dict:
        """
        合约撤单
        
        Args:
            settle: 结算货币 (usdt/btc)
            order_id: 订单ID
            
        Returns:
            Dict: 撤单结果
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            result = self._request("DELETE", f"/api/v4/futures/{settle}/orders/{order_id}")
            
            if result.get("code") == "0":
                order_data = result.get("data", {})
                if isinstance(order_data, dict):
                    return {"code": "0", "data": [order_data]}
                else:
                    return {"code": "0", "data": [{"id": order_data}]}
            return result
        except Exception as e:
            logger.error(f"合约撤单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_order(self, settle: str, order_id: str) -> Dict:
        """
        查询合约订单详情
        
        Args:
            settle: 结算货币 (usdt/btc)
            order_id: 订单ID
            
        Returns:
            Dict: 订单信息
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            result = self._request("GET", f"/api/v4/futures/{settle}/orders/{order_id}")
            
            if result.get("code") == "0":
                data = result.get("data", {})
                if isinstance(data, dict):
                    return {"code": "0", "data": [data]}
                else:
                    return {"code": "0", "data": [data]}
            return result
        except Exception as e:
            logger.error(f"查询合约订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_open_orders(self, settle: str, contract: str = None, status: str = "open") -> Dict:
        """
        查询合约待成交订单
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (可选)
            status: 订单状态 (open/finished)
            
        Returns:
            Dict: 订单列表
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"status": status}
            if contract:
                params["contract"] = contract
            
            result = self._request("GET", f"/api/v4/futures/{settle}/orders", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                return {"code": "0", "data": data if isinstance(data, list) else [data]}
            return result
        except Exception as e:
            logger.error(f"查询合约待成交订单异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_my_trades(self, settle: str, contract: str = None, 
                              order_id: str = None, limit: int = 100) -> Dict:
        """
        查询合约个人成交记录
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (可选)
            order_id: 订单ID (可选，用于查询指定订单的成交记录)
            limit: 返回数量限制
            
        Returns:
            Dict: 成交记录列表，包含 fee（手续费）等字段
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"limit": limit}
            if contract:
                params["contract"] = contract
            if order_id:
                params["order"] = order_id
            
            result = self._request("GET", f"/api/v4/futures/{settle}/my_trades", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                return {"code": "0", "data": data if isinstance(data, list) else [data]}
            return result
        except Exception as e:
            logger.error(f"查询合约成交记录异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_position_close(self, settle: str, contract: str = None, limit: int = 100) -> Dict:
        """
        查询平仓历史记录
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (可选)
            limit: 返回数量限制
            
        Returns:
            Dict: 平仓历史列表，包含 pnl（总盈亏）、pnl_pnl（平仓盈亏）、pnl_fee（手续费）等字段
            示例：
            {
                "time": 1546487347,
                "pnl": "0.00013",
                "pnl_pnl": "0.00011",
                "pnl_fund": "0.00001",
                "pnl_fee": "0.00001",
                "side": "long",
                "contract": "BTC_USDT",
                "text": "web",
                "max_size": "100",
                "accum_size": "100",
                "first_open_time": 1546487347,
                "long_price": "2026.87",
                "short_price": "2544.4"
            }
        """
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Trade API未初始化，请提供API密钥")
        
        try:
            params = {"limit": limit}
            if contract:
                params["contract"] = contract
            
            result = self._request("GET", f"/api/v4/futures/{settle}/position_close", params=params)
            
            if result.get("code") == "0":
                data = result.get("data", [])
                return {"code": "0", "data": data if isinstance(data, list) else [data]}
            return result
        except Exception as e:
            logger.error(f"查询平仓历史异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Futures Market Data API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_contracts(self, settle: str = "usdt") -> Dict:
        """
        获取所有合约列表
        
        Args:
            settle: 结算货币 (usdt/btc)
            
        Returns:
            Dict: 合约列表
        """
        try:
            result = self._request("GET", f"/api/v4/futures/{settle}/contracts")
            if result.get("code") == "0":
                return {"code": "0", "data": result.get("data", [])}
            return result
        except Exception as e:
            logger.error(f"获取合约列表异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=100, time_window=1.0, group="futures")
    def get_futures_candlesticks(self, settle: str, contract: str, interval: str = "1m",
                                  from_time: int = None, to_time: int = None, 
                                  limit: int = 100) -> Dict:
        """
        获取合约K线数据
        
        Args:
            settle: 结算货币 (usdt/btc)
            contract: 合约标识 (如 BTC_USDT)
            interval: K线间隔 (10s, 30s, 1m, 5m, 15m, 30m, 1h, 4h, 8h, 1d, 7d, 30d)
            from_time: 起始时间戳（秒）
            to_time: 结束时间戳（秒）
            limit: 返回数量限制（默认100，最大2000）
            
        Returns:
            Dict: K线数据列表，每条数据包含 [t, v, c, h, l, o, sum]
                  t: 时间戳（秒）
                  v: 成交量（张数）
                  c: 收盘价
                  h: 最高价
                  l: 最低价
                  o: 开盘价
                  sum: 成交额（计价货币）
        """
        try:
            params = {
                "contract": contract,
                "interval": interval,
            }
            # Gate.io API: limit 和 from/to 不能同时存在
            # 如果指定了时间范围，则不使用 limit
            if from_time is not None or to_time is not None:
                if from_time is not None:
                    params["from"] = from_time
                if to_time is not None:
                    params["to"] = to_time
            else:
                params["limit"] = limit
            
            result = self._request("GET", f"/api/v4/futures/{settle}/candlesticks", params=params)
            
            if result.get("code") == "0":
                return {"code": "0", "data": result.get("data", [])}
            return result
        except Exception as e:
            logger.error(f"获取合约K线数据异常: {e}")
            return {"code": "-1", "msg": str(e)}
    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def build_symbol(base_asset: str, quote_currency: str) -> str:
        """
        构建 Gate.io 交易对符号
        
        Args:
            base_asset: 基础资产（如 BTC）
            quote_currency: 计价货币（如 USDT）
            
        Returns:
            str: 交易对符号（如 BTC_USDT）
        """
        return f"{base_asset}_{quote_currency}"
    
    def check_api_available(self) -> bool:
        """
        检查 API 是否可用
        
        Returns:
            bool: API 是否可用
        """
        try:
            result = self.get_currency_pairs()
            return result.get("code") == "0"
        except Exception as e:
            logger.error(f"API 检查失败: {e}")
            return False
