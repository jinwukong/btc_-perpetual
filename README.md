# btc_-perpetual
这个项目主要在研究如何利用 Binance 永续合约（futures）的 1 分钟级别数据与资金费率，结合一些改进过的 CTA 情绪因子与常见技术指标，构建针对短周期（5 分钟、15 分钟、30 分钟）的预测模型，并附带一个基础的强化学习示例环境。我之前在做自研的量化项目时（比如 [这份](https://github.com/jinwukong/-vnpy-CTA-) 里也有类似思路），想把 CTA 策略里的 T+0、T+1 逻辑映射到没有 OI 数据的环境下，这里就尝试了通过 `fundingRate`、`buy_sell_diff` 等来模拟持仓延续和日内投机的差异。
