#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OKX API桥接类
整合所有OKX API接口调用，提供统一的接口访问
"""

import time
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal

from okx.MarketData import MarketAPI
from okx.PublicData import PublicAPI
from okx.Account import AccountAPI
from okx.utils import sign

from leek_core.models import TimeFrame, TradeInsType, Field, FieldType, ChoiceType
from leek_core.utils import get_logger, retry, rate_limit

logger = get_logger(__name__)


# OKX API限速规则 - 接口维度限速


class OkxAdapter:
    """
    OKX API适配器类
    整合MarketData、PublicData、Account等API接口，提供统一的适配接口
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", 
                 domain: str = "https://www.okx.com", flag: str = "0", debug: bool = False, 
                 proxy: Optional[str] = None):
        """
        初始化OKX适配器类
        
        Args:
            api_key: API密钥
            secret_key: 密钥
            passphrase: 密码短语
            domain: API域名
            flag: 环境标识 (0: 实盘, 1: 模拟盘)
            debug: 是否开启调试模式
            proxy: 代理设置
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.domain = domain
        self.flag = flag
        self.debug = debug
        self.proxy = proxy
        
        # 初始化API客户端
        self.market_api = MarketAPI(
            domain=domain, 
            flag=flag, 
            debug=debug, 
            proxy=proxy
        )
        
        self.public_api = PublicAPI(
            domain=domain, 
            flag=flag, 
            debug=debug, 
            proxy=proxy
        )
        
        if api_key and secret_key and passphrase:
            self.account_api = AccountAPI(
                api_key=api_key,
                api_secret_key=secret_key,
                passphrase=passphrase,
                domain=domain,
                flag=flag,
                debug=debug,
                proxy=proxy
            )
        else:
            self.account_api = None
    
    # ==================== Market Data API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=20, time_window=2.0, group="market_data")
    def get_history_candlesticks(self, inst_id: str, bar: str, limit: int = 100, 
                                before: str = "", after: str = "") -> Dict:
        """
        获取历史K线数据
        
        Args:
            inst_id: 产品ID
            bar: K线周期
            limit: 返回结果的数量限制
            before: 开始时间戳
            after: 结束时间戳
            
        Returns:
            Dict: K线数据
        """
        return self.market_api.get_history_candlesticks(
            instId=inst_id, 
            bar=bar, 
            limit=limit, 
            before=before, 
            after=after
        )
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=40, time_window=2.0, group="market_data")  # 产品深度接口：40次/2s
    def get_orderbook(self, inst_id: str, sz: int = 20) -> Dict:
        """
        获取订单簿数据
        
        Args:
            inst_id: 产品ID
            sz: 订单簿深度
            
        Returns:
            Dict: 订单簿数据
        """
        return self.market_api.get_orderbook(instId=inst_id, sz=sz)
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=20, time_window=2.0, group="market_data")  # 产品行情信息：20次/2s
    def get_tickers(self, inst_type: str = "SPOT") -> Dict:
        """
        获取产品行情信息
        
        Args:
            inst_type: 产品类型
            
        Returns:
            Dict: 产品行情数据
        """
        return self.market_api.get_tickers(instType=inst_type)
    

    
    # ==================== Public Data API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=20, time_window=2.0, group="public_data")  # 交易产品基础信息：20次/2s
    def get_instruments(self, inst_type: str, inst_id: str = None) -> Dict:
        """
        获取产品基础信息
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID (可选)
            
        Returns:
            Dict: 产品信息
        """
        params = {"instType": inst_type}
        if inst_id:
            params["instId"] = inst_id
        return self.public_api.get_instruments(**params)
    
    @retry(max_retries=3, retry_interval=0.5)
    @rate_limit(max_requests=20, time_window=2.0, group="public_data")
    def get_mark_price(self, inst_type: str, inst_id: str) -> Dict:
        """
        获取标记价格
        
        Args:
            inst_type: 产品类型
            inst_id: 产品ID
            
        Returns:
            Dict: 标记价格数据
        """
        return self.public_api.get_mark_price(instType=inst_type, instId=inst_id)
    

    
    # ==================== Account API ====================
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=6, time_window=1.0, group="account_data")  # 账户API限速更严格
    def get_account_balance(self, ccy: str = None) -> Dict:
        """
        获取账户余额
        
        Args:
            ccy: 币种 (可选)
            
        Returns:
            Dict: 账户余额数据
        """
        if not self.account_api:
            raise RuntimeError("Account API未初始化，请提供API密钥")
        
        params = {}
        if ccy:
            params["ccy"] = ccy
        return self.account_api.get_account_balance(**params)
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=6, time_window=1.0, group="account_data")  # 账户API限速更严格
    def get_positions(self, inst_type: str = None, inst_id: str = None) -> Dict:
        """
        获取持仓信息
        
        Args:
            inst_type: 产品类型 (可选)
            inst_id: 产品ID (可选)
            
        Returns:
            Dict: 持仓数据
        """
        if not self.account_api:
            raise RuntimeError("Account API未初始化，请提供API密钥")
        
        params = {}
        if inst_type:
            params["instType"] = inst_type
        if inst_id:
            params["instId"] = inst_id
        return self.account_api.get_positions(**params)
    
    @retry(max_retries=3, retry_interval=0.1)
    @rate_limit(max_requests=20, time_window=2.0, group="account_data")  # 账户API限速更严格
    def set_leverage(self, lever: str, mgn_mode: str, inst_id: str = None,
                    ccy: str = None, pos_side: str = None) -> Dict:
        """
        设置杠杆倍数
        
        Args:
            lever: 杠杆倍数
            mgn_mode: 保证金模式
            inst_id: 产品ID (可选)
            ccy: 保证金币种 (可选)
            pos_side: 持仓方向 (可选)
            
        Returns:
            Dict: 设置结果
        """
        if not self.account_api:
            raise RuntimeError("Account API未初始化，请提供API密钥")
        
        params = {
            "lever": lever,
            "mgnMode": mgn_mode
        }
        if inst_id:
            params["instId"] = inst_id
        if ccy:
            params["ccy"] = ccy
        if pos_side:
            params["posSide"] = pos_side
        return self.account_api.set_leverage(**params)
    

    

    

    
    # ==================== Helper Methods ====================
    
    @staticmethod
    def build_inst_id(symbol: str, ins_type: Union[TradeInsType, int], quote_currency: str) -> str:
        """
        构建产品ID
        
        Args:
            symbol: 交易对符号
            ins_type: 交易品种类型
            quote_currency: 计价币种
            
        Returns:
            str: 产品ID
        """
        if not isinstance(ins_type, TradeInsType):
            ins_type = TradeInsType(ins_type)
            
        if ins_type == TradeInsType.SWAP or ins_type == TradeInsType.FUTURES:
            return f"{symbol}-{quote_currency}-SWAP"
        elif ins_type == TradeInsType.OPTION:
            return f"{symbol}-{quote_currency}-OPTION"
        elif ins_type == TradeInsType.SPOT or ins_type == TradeInsType.MARGIN:
            return f"{symbol}-{quote_currency}"
        else:
            raise ValueError(f"不支持的交易类型: {ins_type}")
    
    @staticmethod
    def get_okx_timeframe(timeframe: Union[TimeFrame, str]) -> Optional[str]:
        """
        获取OKX格式的时间周期字符串
        
        Args:
            timeframe: 时间周期
            
        Returns:
            str: OKX格式的时间周期
        """
        # OKX TimeFrame映射
        OKX_TIMEFRAME_MAP = {
            TimeFrame.M1: "1m", TimeFrame.M3: "3m", TimeFrame.M5: "5m",
            TimeFrame.M15: "15m", TimeFrame.M30: "30m", TimeFrame.H1: "1H",
            TimeFrame.H2: "2H", TimeFrame.H4: "4H", TimeFrame.H6: "6H",
            TimeFrame.H12: "12H", TimeFrame.D1: "1D", TimeFrame.W1: "1W",
            TimeFrame.MON1: "1M",
        }
        
        if isinstance(timeframe, str):
            try:
                tf_enum = TimeFrame(timeframe)
                return OKX_TIMEFRAME_MAP.get(tf_enum)
            except ValueError:
                # 如果已经是OKX格式，直接返回
                if timeframe in ["1m", "3m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D", "1W", "1M"]:
                    return timeframe
                logger.error(f"无效的时间周期字符串: {timeframe}")
                return None
        else:
            return OKX_TIMEFRAME_MAP.get(timeframe)
    
    def check_api_available(self) -> bool:
        """
        检查API是否可用
        
        Returns:
            bool: API是否可用
        """
        try:
            result = self.get_tickers(inst_type="SPOT")
            return result and result.get("code") == "0"
        except Exception as e:
            logger.error(f"API检查失败: {e}")
            return False 