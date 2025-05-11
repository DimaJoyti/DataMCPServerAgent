"""
n8n integration module for the Fetch.ai Advanced Crypto Trading System.

This module provides integration with n8n workflows through webhooks and API calls.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests
from dotenv import load_dotenv
from uagents import Agent, Context, Model, Protocol

from .base_agent import BaseAgent, BaseAgentState
from .trading_system import AdvancedCryptoTradingSystem, TradeRecommendation, TradingSignal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebhookType(str, Enum):
    """Types of webhooks."""
    
    RECOMMENDATION = "recommendation"
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    RISK = "risk"
    REGULATORY = "regulatory"
    MACRO = "macro"
    LEARNING = "learning"

class N8nWebhook(Model):
    """Model for an n8n webhook."""
    
    name: str
    type: WebhookType
    url: str
    headers: Dict[str, str] = {}
    active: bool = True

class N8nWorkflow(Model):
    """Model for an n8n workflow."""
    
    id: str
    name: str
    description: Optional[str] = None
    active: bool = True
    webhooks: List[N8nWebhook] = []

class N8nIntegrationState(BaseAgentState):
    """State model for the n8n Integration."""
    
    n8n_base_url: str = "http://localhost:5678"
    n8n_api_key: Optional[str] = None
    workflows: List[N8nWorkflow] = []
    webhooks: List[N8nWebhook] = []
    last_sent_data: Dict[str, Any] = {}

class N8nIntegration(BaseAgent):
    """Integration with n8n workflows."""
    
    def __init__(
        self,
        name: str = "n8n_integration",
        seed: Optional[str] = None,
        port: Optional[int] = None,
        endpoint: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        n8n_base_url: Optional[str] = None,
        n8n_api_key: Optional[str] = None,
        trading_system: Optional[AdvancedCryptoTradingSystem] = None
    ):
        """Initialize the n8n Integration.
        
        Args:
            name: Name of the agent
            seed: Seed for deterministic address generation
            port: Port for the agent server
            endpoint: Endpoint for the agent server
            logger: Logger instance
            n8n_base_url: Base URL for n8n
            n8n_api_key: API key for n8n
            trading_system: Trading system to integrate with
        """
        super().__init__(name, seed, port, endpoint, logger)
        
        # Initialize state
        self.state = N8nIntegrationState()
        
        # Set n8n configuration
        if n8n_base_url:
            self.state.n8n_base_url = n8n_base_url
        if n8n_api_key:
            self.state.n8n_api_key = n8n_api_key
        
        # Store trading system
        self.trading_system = trading_system
        
        # Initialize default webhooks
        self._initialize_default_webhooks()
        
        # Register handlers
        self._register_handlers()
    
    def _initialize_default_webhooks(self):
        """Initialize default webhooks."""
        # Add default webhooks
        self.state.webhooks = [
            N8nWebhook(
                name="Trading Recommendations",
                type=WebhookType.RECOMMENDATION,
                url=f"{self.state.n8n_base_url}/webhook/trading-recommendations"
            ),
            N8nWebhook(
                name="Market Data",
                type=WebhookType.MARKET_DATA,
                url=f"{self.state.n8n_base_url}/webhook/market-data"
            ),
            N8nWebhook(
                name="Sentiment Analysis",
                type=WebhookType.SENTIMENT,
                url=f"{self.state.n8n_base_url}/webhook/sentiment-analysis"
            ),
            N8nWebhook(
                name="Technical Analysis",
                type=WebhookType.TECHNICAL,
                url=f"{self.state.n8n_base_url}/webhook/technical-analysis"
            ),
            N8nWebhook(
                name="Risk Assessment",
                type=WebhookType.RISK,
                url=f"{self.state.n8n_base_url}/webhook/risk-assessment"
            ),
            N8nWebhook(
                name="Regulatory Compliance",
                type=WebhookType.REGULATORY,
                url=f"{self.state.n8n_base_url}/webhook/regulatory-compliance"
            ),
            N8nWebhook(
                name="Macro Correlation",
                type=WebhookType.MACRO,
                url=f"{self.state.n8n_base_url}/webhook/macro-correlation"
            ),
            N8nWebhook(
                name="Learning Optimization",
                type=WebhookType.LEARNING,
                url=f"{self.state.n8n_base_url}/webhook/learning-optimization"
            )
        ]
    
    def _register_handlers(self):
        """Register handlers for the agent."""
        
        @self.agent.on_interval(period=60.0)
        async def send_trading_recommendations(ctx: Context):
            """Send trading recommendations to n8n."""
            if not self.trading_system:
                ctx.logger.warning("No trading system available")
                return
            
            # Get recent recommendations
            recommendations = self.trading_system.state.recent_recommendations
            
            if not recommendations:
                ctx.logger.info("No recent recommendations to send")
                return
            
            # Find webhook
            webhook = next(
                (w for w in self.state.webhooks if w.type == WebhookType.RECOMMENDATION and w.active),
                None
            )
            
            if not webhook:
                ctx.logger.warning("No active webhook for recommendations")
                return
            
            # Send recommendations
            await self._send_to_webhook(
                webhook,
                {
                    "recommendations": [rec.dict() for rec in recommendations],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        @self.agent.on_interval(period=300.0)
        async def send_market_data(ctx: Context):
            """Send market data to n8n."""
            if not self.trading_system:
                ctx.logger.warning("No trading system available")
                return
            
            # Find webhook
            webhook = next(
                (w for w in self.state.webhooks if w.type == WebhookType.MARKET_DATA and w.active),
                None
            )
            
            if not webhook:
                ctx.logger.warning("No active webhook for market data")
                return
            
            # Get market data
            market_data = {}
            for symbol in self.trading_system.state.symbols_to_track:
                try:
                    ticker = await self.trading_system.exchange.fetch_ticker(symbol)
                    market_data[symbol] = {
                        "last": ticker["last"],
                        "bid": ticker["bid"],
                        "ask": ticker["ask"],
                        "volume": ticker["volume"],
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    ctx.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            
            # Send market data
            await self._send_to_webhook(
                webhook,
                {
                    "market_data": market_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    async def _send_to_webhook(self, webhook: N8nWebhook, data: Dict[str, Any]) -> bool:
        """Send data to an n8n webhook.
        
        Args:
            webhook: Webhook to send to
            data: Data to send
            
        Returns:
            Success status
        """
        try:
            # Store last sent data
            self.state.last_sent_data[webhook.type] = {
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send data
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    json=data,
                    headers=webhook.headers
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Successfully sent data to {webhook.name} webhook")
                        return True
                    else:
                        self.logger.error(
                            f"Error sending data to {webhook.name} webhook: "
                            f"Status {response.status}"
                        )
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error sending data to {webhook.name} webhook: {str(e)}")
            return False
    
    async def fetch_n8n_workflows(self) -> List[N8nWorkflow]:
        """Fetch workflows from n8n.
        
        Returns:
            List of workflows
        """
        if not self.state.n8n_api_key:
            self.logger.warning("No n8n API key available")
            return []
        
        try:
            # Fetch workflows
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.state.n8n_base_url}/api/v1/workflows",
                    headers={
                        "X-N8N-API-KEY": self.state.n8n_api_key
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convert to workflows
                        workflows = []
                        for item in data["data"]:
                            workflow = N8nWorkflow(
                                id=item["id"],
                                name=item["name"],
                                description=item.get("description"),
                                active=item.get("active", True)
                            )
                            workflows.append(workflow)
                        
                        # Update state
                        self.state.workflows = workflows
                        
                        self.logger.info(f"Fetched {len(workflows)} workflows from n8n")
                        return workflows
                    else:
                        self.logger.error(
                            f"Error fetching workflows from n8n: "
                            f"Status {response.status}"
                        )
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error fetching workflows from n8n: {str(e)}")
            return []
    
    async def trigger_workflow(self, workflow_id: str, data: Dict[str, Any]) -> bool:
        """Trigger an n8n workflow.
        
        Args:
            workflow_id: ID of the workflow to trigger
            data: Data to send
            
        Returns:
            Success status
        """
        if not self.state.n8n_api_key:
            self.logger.warning("No n8n API key available")
            return False
        
        try:
            # Trigger workflow
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.state.n8n_base_url}/api/v1/workflows/{workflow_id}/trigger",
                    json=data,
                    headers={
                        "X-N8N-API-KEY": self.state.n8n_api_key
                    }
                ) as response:
                    if response.status == 200:
                        self.logger.info(f"Successfully triggered workflow {workflow_id}")
                        return True
                    else:
                        self.logger.error(
                            f"Error triggering workflow {workflow_id}: "
                            f"Status {response.status}"
                        )
                        return False
                        
        except Exception as e:
            self.logger.error(f"Error triggering workflow {workflow_id}: {str(e)}")
            return False

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Get n8n configuration from environment variables
    n8n_base_url = os.getenv('N8N_BASE_URL', 'http://localhost:5678')
    n8n_api_key = os.getenv('N8N_API_KEY')
    
    # Create trading system
    trading_system = AdvancedCryptoTradingSystem(
        name="n8n_trading_system",
        exchange_id="binance",
        api_key=os.getenv('EXCHANGE_API_KEY'),
        api_secret=os.getenv('EXCHANGE_API_SECRET')
    )
    
    # Create n8n integration
    n8n_integration = N8nIntegration(
        name="n8n_integration",
        n8n_base_url=n8n_base_url,
        n8n_api_key=n8n_api_key,
        trading_system=trading_system
    )
    
    # Start agents
    await asyncio.gather(
        trading_system.start_all_agents(),
        n8n_integration.run_async()
    )

if __name__ == "__main__":
    asyncio.run(main())
