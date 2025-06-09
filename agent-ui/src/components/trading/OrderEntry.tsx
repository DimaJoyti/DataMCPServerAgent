'use client'

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  ShoppingCart,
  TrendingDown,
  Target,
  Clock,
  AlertTriangle,
  CheckCircle,
  Send,
  Calculator
} from 'lucide-react'
import { useCreateOrder, useOrders } from '@/services/hooks/useTrading'
import { CreateOrderRequest } from '@/services/api/types'

interface OrderForm {
  symbol: string
  side: 'buy' | 'sell'
  orderType: 'market' | 'limit' | 'stop' | 'stop-limit'
  quantity: string
  price: string
  stopPrice: string
  timeInForce: 'day' | 'gtc' | 'ioc' | 'fok'
  algorithm?: 'twap' | 'vwap' | 'iceberg'
}

interface RecentOrder {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: string
  quantity: number
  price?: number
  status: 'pending' | 'filled' | 'cancelled' | 'rejected'
  timestamp: Date
}

export const OrderEntry: React.FC = () => {
  const [orderForm, setOrderForm] = useState<OrderForm>({
    symbol: 'BTC/USD',
    side: 'buy',
    orderType: 'market',
    quantity: '',
    price: '',
    stopPrice: '',
    timeInForce: 'day'
  })

  const [estimatedCost, setEstimatedCost] = useState<number | null>(null)

  // Use trading hooks
  const { submitOrder, loading: orderLoading, error: orderError } = useCreateOrder()
  const { orders: recentOrders } = useOrders({}, { limit: 10 })

  const symbols = ['BTC/USD', 'ETH/USD', 'AAPL', 'TSLA', 'SPY', 'QQQ', 'EUR/USD', 'GBP/USD']

  const handleInputChange = (field: keyof OrderForm, value: string) => {
    setOrderForm(prev => ({ ...prev, [field]: value }))
    
    // Calculate estimated cost for limit orders
    if ((field === 'quantity' || field === 'price') && orderForm.orderType === 'limit') {
      const qty = parseFloat(field === 'quantity' ? value : orderForm.quantity)
      const price = parseFloat(field === 'price' ? value : orderForm.price)
      if (!isNaN(qty) && !isNaN(price)) {
        setEstimatedCost(qty * price)
      }
    }
  }

  const handleSubmitOrder = async () => {
    // Validate form
    if (!orderForm.quantity || parseFloat(orderForm.quantity) <= 0) {
      alert('Please enter a valid quantity')
      return
    }

    if ((orderForm.orderType === 'limit' || orderForm.orderType === 'stop-limit') &&
        (!orderForm.price || parseFloat(orderForm.price) <= 0)) {
      alert('Please enter a valid price')
      return
    }

    try {
      // Create order request
      const orderRequest: CreateOrderRequest = {
        symbol: orderForm.symbol,
        side: orderForm.side as 'buy' | 'sell',
        type: orderForm.orderType as 'market' | 'limit' | 'stop' | 'stop-limit',
        quantity: parseFloat(orderForm.quantity),
        price: orderForm.price ? parseFloat(orderForm.price) : undefined,
        stopPrice: orderForm.stopPrice ? parseFloat(orderForm.stopPrice) : undefined,
        timeInForce: orderForm.timeInForce as 'day' | 'gtc' | 'ioc' | 'fok'
      }

      // Submit order via API
      await submitOrder(orderRequest)

      // Reset form on success
      setOrderForm(prev => ({
        ...prev,
        quantity: '',
        price: '',
        stopPrice: ''
      }))
      setEstimatedCost(null)

    } catch (error) {
      console.error('Failed to submit order:', error)
      alert('Failed to submit order. Please try again.')
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-green-500'
      case 'pending': return 'text-yellow-500'
      case 'cancelled': return 'text-gray-500'
      case 'rejected': return 'text-red-500'
      default: return 'text-gray-500'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled': return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'pending': return <Clock className="h-4 w-4 text-yellow-500" />
      case 'cancelled': return <Target className="h-4 w-4 text-gray-500" />
      case 'rejected': return <AlertTriangle className="h-4 w-4 text-red-500" />
      default: return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  return (
    <Card className="h-full bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm border-0 shadow-lg">
      <CardHeader className="pb-4">
        <CardTitle className="text-xl font-bold flex items-center space-x-2">
          <ShoppingCart className="h-5 w-5" />
          <span>Order Entry</span>
        </CardTitle>
        <CardDescription>Place and manage trading orders</CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <Tabs defaultValue="simple" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="simple">Simple Order</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="simple" className="space-y-4">
            {/* Symbol Selection */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Symbol</label>
                <Select value={orderForm.symbol} onValueChange={(value) => handleInputChange('symbol', value)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {symbols.map(symbol => (
                      <SelectItem key={symbol} value={symbol}>
                        {symbol}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Side</label>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant={orderForm.side === 'buy' ? 'default' : 'outline'}
                    onClick={() => handleInputChange('side', 'buy')}
                    className={orderForm.side === 'buy' ? 'bg-green-500 hover:bg-green-600' : ''}
                  >
                    Buy
                  </Button>
                  <Button
                    variant={orderForm.side === 'sell' ? 'default' : 'outline'}
                    onClick={() => handleInputChange('side', 'sell')}
                    className={orderForm.side === 'sell' ? 'bg-red-500 hover:bg-red-600' : ''}
                  >
                    Sell
                  </Button>
                </div>
              </div>
            </div>

            {/* Order Type */}
            <div>
              <label className="text-sm font-medium mb-2 block">Order Type</label>
              <Select value={orderForm.orderType} onValueChange={(value) => handleInputChange('orderType', value)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="market">Market</SelectItem>
                  <SelectItem value="limit">Limit</SelectItem>
                  <SelectItem value="stop">Stop</SelectItem>
                  <SelectItem value="stop-limit">Stop Limit</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Quantity and Price */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Quantity</label>
                <input
                  type="number"
                  step="0.001"
                  placeholder="0.000"
                  value={orderForm.quantity}
                  onChange={(e) => handleInputChange('quantity', e.target.value)}
                  className="w-full px-3 py-2 border border-border rounded-md bg-background"
                />
              </div>

              {(orderForm.orderType === 'limit' || orderForm.orderType === 'stop-limit') && (
                <div>
                  <label className="text-sm font-medium mb-2 block">Price</label>
                  <input
                    type="number"
                    step="0.01"
                    placeholder="0.00"
                    value={orderForm.price}
                    onChange={(e) => handleInputChange('price', e.target.value)}
                    className="w-full px-3 py-2 border border-border rounded-md bg-background"
                  />
                </div>
              )}
            </div>

            {/* Estimated Cost */}
            {estimatedCost && (
              <div className="flex items-center space-x-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <Calculator className="h-4 w-4 text-blue-500" />
                <span className="text-sm">
                  Estimated Cost: <span className="font-medium">${estimatedCost.toFixed(2)}</span>
                </span>
              </div>
            )}

            {/* Submit Button */}
            <Button 
              onClick={handleSubmitOrder}
              className={`w-full ${
                orderForm.side === 'buy' 
                  ? 'bg-green-500 hover:bg-green-600' 
                  : 'bg-red-500 hover:bg-red-600'
              }`}
            >
              <Send className="h-4 w-4 mr-2" />
              {orderForm.side === 'buy' ? 'Place Buy Order' : 'Place Sell Order'}
            </Button>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-4">
            <div className="text-center py-8 text-muted-foreground">
              <Target className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>Advanced order features coming soon...</p>
              <p className="text-sm">TWAP, VWAP, Iceberg orders, and more</p>
            </div>
          </TabsContent>
        </Tabs>

        {/* Recent Orders */}
        <div className="space-y-4">
          <h4 className="font-semibold">Recent Orders</h4>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {recentOrders.map((order) => (
              <div key={order.orderId} className="flex items-center justify-between p-3 bg-muted/50 rounded-md">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(order.status)}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-medium">{order.symbol}</span>
                      <Badge variant={order.side === 'buy' ? 'default' : 'destructive'} className="text-xs">
                        {order.side.toUpperCase()}
                      </Badge>
                      <span className="text-sm text-muted-foreground">{order.type}</span>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Qty: {order.quantity} {order.price && `@ $${order.price.toFixed(2)}`}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${getStatusColor(order.status)} capitalize`}>
                    {order.status}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {new Date(order.updatedAt).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
