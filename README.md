# btc_-perpetual
这个项目主要在研究如何利用 Binance 永续合约（futures）的 1 分钟级别数据与资金费率，结合一些改进过的 CTA 情绪因子与常见技术指标，构建针对短周期（5 分钟、15 分钟、30 分钟）的预测模型，并附带一个基础的强化学习示例环境。我之前在做自研的量化项目时（比如 [这份](https://github.com/jinwukong/-vnpy-CTA-) 里也有类似思路），想把 CTA 策略里的 T+0、T+1 逻辑映射到没有 OI 数据的环境下，这里就尝试了通过 `fundingRate`、`buy_sell_diff` 等来模拟持仓延续和日内投机的差异。

```markdown
# BTCUSDT 永续合约量化项目

**项目概要**  
这个项目主要在研究如何利用 Binance 永续合约（futures）的 1 分钟级别数据与资金费率，结合一些改进过的 CTA 情绪因子与常见技术指标，构建针对短周期（5 分钟、15 分钟、30 分钟）的预测模型，并附带一个基础的强化学习示例环境。我之前在做自研的量化项目时（比如 [这份](https://github.com/jinwukong/-vnpy-CTA-) 里也有类似思路），想把 CTA 策略里的 T+0、T+1 逻辑映射到没有 OI 数据的环境下，这里就尝试了通过 `fundingRate`、`buy_sell_diff` 等来模拟持仓延续和日内投机的差异。以下是我从头到尾的操作记录，以及一些我自己写的说明文字。

---

## 专业简述

1. **数据拉取**：  
   - 通过 `futures_historical_klines` 获取过去一年的 BTCUSDT 1m K 线，字段包含 `open, high, low, close, volume, taker_buy_base_volume`。  
   - 用 `fapi/v1/fundingRate` 获取资金费率，每 8 小时一条，然后**前向填充**到每分钟，这样每根 K 线就都有一个对应的资金费率值。  
   - 数据比较大，就分段循环拉取，每次只拿 1500 条，直到覆盖整年，然后写进 CSV。  
   - 之所以使用永续合约 API 而非现货，是因为永续合约能直接获取 taker 买入量，用来区分主动买卖，还能得到资金费率(这对短线情绪影响很大)。

2. **特征工程**：  
   - 计算了常见技术指标：CFO(Chande Forecast Oscillator)、PWMA(帕斯卡加权移动平均)、RVI(Relative Vigor Index)；  
   - 做一些基础的滚动统计，如 30 分钟对数收益波动率 `roll_vol_30`；  
   - 最终做了一个**CTA 情绪因子**：把短线(taker_ratio)和中线(资金费率×买卖差的滚动)组合，并在波动率高或资金费率极端的时候动态调整它的权重；再做个 DOV 替代 `abs(Δ buy_sell_diff)/Δ volume` 用来捕捉突变。  
   - 把结果存成 `df_cta_optimized.csv` 等文件，便于后续训练和强化学习。

3. **机器学习 (LightGBM + Optuna)**：  
   - 构造 5 分钟、15 分钟、30 分钟后的涨跌(二分类)标签，用 `TimeSeriesSplit` 避免未来数据泄漏；  
   - 用 Optuna 做超参数搜索来找到比较好的 `learning_rate, num_leaves, subsample` 等；  
   - 最后拿到三个 LGB 模型，对应 3 个时间尺度，AUC 大约 0.52~0.53，作为一个初步基线可接受。

4. **强化学习示例**（可选）：  
   - 写了一个小的 Gym 环境，每一步是 1 分钟 K 线，动作可做多/空/平；  
   - 在 reward 中考虑了资金费率(多头要付费)，主要演示 RL 在加密短线上能否学到一套策略；  
   - 没有对撮合深度或滑点做精确模拟，仅供参考。

后面若要接到 freqtrade 里，只需要在 freqtrade 的策略文件中：  
- **bot_start** 里加载模型 pkl；  
- **populate_indicators** 里实现与此项目相同的特征工程；  
- **populate_entry_trend** 用 `model.predict_proba(...)` 来判断何时买。  
这样就能在 freqtrade 的回测/实盘中跑出绩效曲线。

---

## 更详细的开发记录

**1. 关于“futures_historical_klines”和“fundingRate”API**  
我发现：  
- `futures_historical_klines` 每次只能取 1500 条，我就写了一个 while 循环，用 `startTime` ~ `endTime` 分段取，每日或每 8h+1 ms 递增；  
- `fundingRate` 数据量更少，但同样 `limit=1000`，就同理循环到结束，**最后一条**的 `fundingTime` +1 毫秒继续。  
- 我对 K 线的话，就 `last_open_time = data[-1][0] + 60*1000`，因为知道下一根 K 线肯定在下一个完整分钟，这在语义上也很明确。  
- 时区的话，Binance 是 UTC，若我用上海时区就可能出现 2 月 29 日闰年的 Bug，所以全部以 UTC 计。

**2. 我在注释里写了好多原因**  
例如“为什么k线+1分钟但funding只+1毫秒”，以及“是否把 8 小时的资金费率做线性插值”等等，这些其实都是我在实际操作中遇到的思考。结论是**前向填充**更简单且足够准确，因为资金费率结算本就离散，没必要对它做三次样条插值或均匀摊到每分钟。  
不过有些策略研究会只在资金费率结算那刻收取一次费用，这就会让资金费率序列绝大多数时间是 0，只有结算分钟非 0；对建模不方便。但对回测撮合也许更真实，所以看场景需求。

**3. CTA 因子的想法**  
在我之前一个 [vnpy-CTA](https://github.com/jinwukong/-vnpy-CTA-) 项目里，我通过 OI (Open Interest) 把 T+0 和 T+1 交易者分开，这次因为没拿到全年的 OI，就只能用 taker 交易量与资金费率的滚动均值来代替。  
- 短线：taker_ratio 高 => 短线投机意愿强；  
- 中线：资金费率 + buy_sell_diff 若为正，说明多头持仓在加强 (相当于 OI 在涨)；若为负则空头占主导  
- 波动率越大，就更信赖短线；波动率小，依赖中线；二者同向再放大信号；资金费率若特别正 => 说明多头过热，稍微减一下 CTA signal，特别负 => 说明空头极端，也可能反弹，就稍微加一下。  
- DOV：虽然 OI 的 ΔOI / ΔVolume 是常见 CTA 公式，但这里改成 `abs(Δ buy_sell_diff)/Δ volume` 再 clip(10)；我反复测试后觉得 clip 10~20 都可以，防止某些极端分钟 spike。

**4. “多因子 + 短期涨跌” 的 LightGBM**  
把 CFO/PWMA/RVI/CTA 因子等全部扔进特征，再加了 close, volume, buy_sell_diff 等基础列，然后建立 (5min/15min/30min后价格是否上涨) 三份 label；  
为了确保没有未来泄漏，我使用**TimeSeriesSplit** 进行滚动切分：  
```
train: 0~(0.6*N), test: (0.6*N+1)~(0.7*N)
train: 0~(0.7*N), test: (0.7*N+1)~(0.8*N)
...
```
Optuna 做了30~88次 trial，自动搜索了 `learning_rate, num_leaves, subsample, colsample_bytree` 等。  
最后 AUC 大约 0.52~0.53(不算太好，但起码证明了短期里的一点可预测性)。因为行情涨跌带有较多噪音，我平时也就把 0.52+ 当一个 baseline，然后后面可能通过订单簿大单数据或链上指标再提高一点点。

**5. RL 环境**  
这个我只是想尝试 “每分钟 step，一步 reward = 净值变化 + 资金费扣除(多头) - (空头)。” Action= [-1,1] 表示做多/空的仓位大小。  
在 stable-baselines3 里，用 PPO 跑一下看看 reward 曲线。但毕竟没模拟滑点/深度，所以不一定实用。  
可以继续扩展，比如 keypoint:  
- 8 小时一次把累计资金费扣一次  
- 或者每分钟把资金费 /480分摊  
- 结合 CTA 因子当观测，也许 RL 会学到相似策略

---

## 后面我准备干什么

- **接入 freqtrade**：  
  打算把训练好的 3 个 LightGBM 模型放到 freqtrade 的回测里，通过 `_populate_entry_trend()` 里的 `model.predict_proba(X_)` 来决定买卖，再加一个 `custom_stake_amount()` 调仓大小。这样可以看更真实的回测统计，比如最大回撤、年化收益、交易笔数等。  
- **更多因子**：  
  挖掘订单簿挂单不平衡(前 5 档深度买卖量对比)、社交媒体（Twitter）情绪、链上指标(地址活跃度)、大额成交占比等。  
- **多任务**：  
  把 5/15/30min 三个目标合并成一个模型(MultiOutputClassifier) 或者 LSTM/Transformer 方式一次性预测各时点概率。  
- **回测 / 实盘**：  
  先在 freqtrade 用回测跑多币种看看波动适应度，再考虑实盘交易(要注意下单限频/盘口流动性/交易费等现实问题)。  
  也可以在别的 vn.py 引擎或我自己写的撮合中把 CTA 因子 + LightGBM 模型结合。

**反正现阶段也能看出**：  
1. 我确实把 Binance 永续合约 1 分钟数据拿来做了很多自定义处理；  
2. 在因子思路上既有 CFO/RVI 这种常规技术指标，也有 CTA-style(短线+中线+波动率权重)；  
3. 训练结果 AUC 不到 0.53 其实不算高，但至少建立了一个框架。后面若再做集成或多因子，就可能稳步提升。

---

## 结语

- **免责声明**：一切代码仅用于研究或参考，实盘盈亏责任自负。  
- **推荐**：如果你和我一样想做快节奏的短线，其实更高频(秒级)或多时段多数据(小时线 / 日线)也可以融合。但越高频需要越多微观数据(逐笔或订单簿)，这部分可用开源 Kaiko / Cryptowatch / CryptoTick 等 data 付费才能拿全量。  
- **欢迎大家**：Issue/PR都可以，我把这项目开源到 GitHub，希望能说明我在量化 & Python 代码方面的能力。如果公司本身有内部回测框架也很正常，我只要能熟练开发就行了嘛。  

**作者：** 高鸿晋 (假名)  
**写于：** 2024.3.1

*PS. 如果想进一步交流，请随时给我发邮件或提 issue，非常欢迎讨论策略思路！*

```
