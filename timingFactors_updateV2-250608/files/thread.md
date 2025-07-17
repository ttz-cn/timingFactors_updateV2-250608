# Note

- 首先需要用因子业绩评价的模块，对每个信号因子进行重新测量和评价，并记录每个因子的构造方式和尝试的结果
  - :timer_clock: 20241223-20250103回测了除量价因子外的因子。目前已回测因子中，绝大部分都找到了可以跟上指数并且胜率不低于0.5的信号提取方式。但目前的信号获取方式显然不是最优方式，且参数也有根据数据结构和特性进一步优化的空间，目前还没有进行上述操作。

- 月频数据区间是20160101-20241131，日频和周频因子数据区间是20120101-20141131（数据公布时间、特征提取都会影响实际回测区间）

# Testing

##  **PPI:全部工业品:当月同比**

```python
factor = pd.read_excel('./db/ppi.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102091014665](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102091014665.png)

![image-20250102091033979](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102091033979.png)

##  **CPI:当月同比**

```python
factor = pd.read_excel('./db/cpi.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102091644658](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102091644658.png)

![image-20250102091731759](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102091731759.png)

##  **核心CPI：当月同比**

```python
factor = pd.read_excel('./db/core_cpi_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102092654171](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102092654171.png)

![image-20250102092707055](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102092707055.png)

## **PPI:全部工业品:当月同比-CPI:当月同比**

```python
factor_1 = pd.read_excel('./db/ppi.xlsx').set_index('Unnamed: 0').dropna()
factor_2 = pd.read_excel('./db/cpi.xlsx').set_index('Unnamed: 0').dropna()
factor = factor_1 - factor_2
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241226174328168](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241226174328168.png)

![image-20241226174339498](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241226174339498.png)

## **PMI：新出口订单**

```python
factor = pd.read_excel('./db/pmi_neo.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

- 用原始数据产生的动量要远优于mom、yoy处理后

![image-20250102093910500](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102093910500.png)

![image-20250102093920383](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102093920383.png)

## **PMI：生产**

- 回测结果表明，mom效果总是好于yoy效果，因此下列涉及类似操作均采用mom

```python
factor = pd.read_excel('./db/pmi_p.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.pct_change(1).dropna()
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250117183226366](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250117183226366.png)

![image-20250102094504085](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102094504085.png)

## **服务业PMI**

```python
factor = pd.read_excel('./db/pmi_sv.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

- 我们发现，对服务业原始数据采用动量策略的效果要远好于yoy或mom，这一点和制造业pmi是不同的。暂时的猜测是和服务业pmi的滞后性有关，后续待进一步理论分析。

![image-20250102100005223](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100005223.png)

![image-20250102100018037](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100018037.png)

## **综合PMI**

```python
factor = pd.read_excel('./db/pmi_coi.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.pct_change(1).dropna()
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102100818780](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100818780.png)

![image-20250102100826899](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100826899.png)

## **制造业PMI**

```python
factor = pd.read_excel('./db/pmi.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.pct_change(1).dropna()
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102100518800](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100518800.png)

![image-20250102100526873](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102100526873.png)

## **PMI:新订单**

````python
factor = pd.read_excel('./db/pmi_no.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.pct_change(1).dropna()
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
````

- 用mom是合理的，全称是制造业PMI：新订单

![image-20250102101028325](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102101028325.png)

![image-20250102101037595](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102101037595.png)

## **BCI**

```pyth
factor = pd.read_excel('./db/bci.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.pct_change(1).dropna()
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

- 仍然mom是合理的，从传导上应该是：经济预期变化 → 企业决策调整(BCI) → 生产变化(制造业PMI) → 收入改善 → 服务需求变化(服务业PMI)

![image-20250102101528670](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102101528670.png)

![image-20250102101537302](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102101537302.png)

## **中国:社会消费品零售总额:当月同比**

```python
factor = pd.read_excel('./db/rs_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241227110644026](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227110644026.png)

![image-20241227110654518](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227110654518.png)

## **中国:房地产开发投资完成额:累计同比**

```python
factor = pd.read_excel('./db/rei_ytd.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241227111257676](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111257676.png)

![image-20241227111316943](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111306943.png)

## **中国:固定资产投资完成额:基础设施建设投资(不含电力):累计同比**

```python
factor = pd.read_excel('./db/fai_infr_ytd.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241227111610438](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111610438.png)

![image-20241227111618910](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111618910.png)

## **中国:产量:发电量:当月同比**

```python
factor = pd.read_excel('./db/ep_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241227111915093](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111915093.png)

![image-20241227111924655](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227111924655.png)

## **存款类机构质押式回购加权利率：7天**

```python
factor = pd.read_excel('./db/dr007.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_ = factor_.ewm(span=8,adjust=True).mean().ffill()  # ffill用来清理一周都时非工作日的情况
factor_
factor_ = factor_
factor_sig = mom_rule(factor_,isMTM=False)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

> :rotating_light: 这里涉及到对日频数据转为周频数据的特征提取方法优化。
>
> 首先需要说明的是：![image-20241227174201314](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241227174201314.png)
>
> 所以，我这里就直接摒弃了上述方法，剩下的尝试有：
>
> 1.原方法（过去20天滚动平均/ewm，再取每周最后一天当作这周的代理） 
>
> 2.周内ewm直接提取这周的特征
>
> 3.首先周内取均值，再进行月的ewm（和2相比，这个方权重不够平滑）
>
> 方法1和方法2其实一样，只是span不同罢了，所以他们的结果其实差不多，最终发现span=10回测结果最好。但是，单纯用日度ewm然后取单值省去了平滑性，没有提取到数据长期趋势特征。而再这个基础上取周频数据的ewm则可以。所以方法三最好。
>
> 另外，ewm的效果要好于sma效果，猜测是因为ewm更能捕捉近期波动。经过多组数据检验，证实了方法3稳定由于方法1和方法2，下列对日度数据的降频均采用方法3

![image-20250102131620918](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102131620918.png)

![image-20250102131629464](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102131629464.png)

```python
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
```

![image-20250423102304108](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423102304108.png)

![image-20250423102328030](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423102328030.png)

## **银行间质押式回购加权利率:7 天**

```python
由于r007的波动太大了，因此我们在特征提取时，用过去8周（更长的周期）做ewm，此时可以达到和r007类似的效果。这一结果在数学上也是可以证明的。
factor = pd.read_excel('./db/r007.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_ = factor_.ewm(span=8,adjust=True).mean().ffill()  # ffill用来清理一周都时非工作日的情况
factor_
factor_sig = mom_rule(factor_,isMTM=False)
bt_data = get_bt_data(df_881001,factor_sig,'20120101','20241130',freq='W')
```

![image-20250102130711936](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102130711936.png)

![image-20250102130727177](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102130727177.png)

![image-20250423102717988](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423102717988.png)

![image-20250423102737491](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423102737491.png)

## **中长期贷款余额同比**

```python
factor = pd.read_excel('./db/m&ltl_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102111648795](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102111648795.png)

![image-20250102111659687](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102111659687.png)

## **中国:金融机构:中长期贷款余额:人民币**

- 每个月都比上个月高，没得算

## **中债国债到期收益率:3个月**

- 国债的收益率相对稳定，不需要长周期。利息相关的效果都一般

```python
factor = pd.read_excel('./db/tby_3m.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-MON').last() #last取数，当周一非工作日时用本周期最近数据
factor_ = factor_.ewm(span=4,adjust=True).mean().ffill()  # ffill用来清理一周都时非工作日的情况
factor_sig = mom_rule(factor_,isMTM=False)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

![image-20250102131839000](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102131839000.png)

![image-20250102131850131](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102131850131.png)

![image-20250423102947792](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423102947792.png)

![image-20250423103000501](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423103000501.png)

##  **中国:M1:同比**

```python
factor = pd.read_excel('./db/m1_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102113103207](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113103207.png)

![image-20250102113112401](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113112401.png)









## **中国:M1:同比-中国:M2:同比**

```python
factor_1 = pd.read_excel('./db/m1_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_2 = pd.read_excel('./db/m2_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor = factor_1 - factor_2
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102113158376](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113158376.png)

![image-20250102113210213](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113210213.png)

## **中国:M2:同比**

```python
factor = pd.read_excel('./db/m2_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20250102113252818](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113252818.png)

![image-20250102113302987](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102113302987.png)

## **中国:社会融资规模存量:同比**

```python
factor = pd.read_excel('./db/tsf_yoy.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_sig = mom_rule(factor_,isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='M')
```

![image-20241231151742895](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241231151742895.png)

![image-20241231151756527](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20241231151756527.png)

## **中国:新成立基金份额:偏股型**

```python
factor = pd.read_excel('./db/neqf.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-MON').last() #last取数，当周一非工作日时用本周期最近数据
benchmark_neqf = get_sma_idc(factor_,timeperiod=8)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE')
```

- 对于突破均线策略，我们在提取周度特征的时候，不再进行周度移动。

![image-20250102134214223](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102134214223.png)

![image-20250102134231127](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102134231127.png)

```python
# 位置原则，参考线
benchmark_neqf = get_sma_idc(factor_,timeperiod=20)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
```

![image-20250423103326119](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423103326119.png)

![image-20250423103351979](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423103351979.png)

## **融资买入额**

```python
factor = pd.read_excel('./db/mrb.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
benchmark_neqf = get_sma_idc(factor_,timeperiod=8)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE')
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')

```

![image-20250102140903486](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102140903486.png)

![image-20250102140912693](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102140912693.png)

```python
# 位置原则，参考线
benchmark_neqf = get_sma_idc(factor_,timeperiod=20)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
```

![image-20250423103835000](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423103835000.png)

![image-20250423103847121](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423103847121.png)

## USDCNH:即期汇率

- 由于汇率的波动总是很小的，因此我们使用过去一年作为sam中枢

```python
factor = pd.read_excel('./db/usdcnh_spot.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
benchmark_neqf = get_sma_idc(factor_,timeperiod=52)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE')
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

![image-20250102161101754](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102161101754.png)

![image-20250102161116318](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102161116318.png)

```python

# 位置原则，参考线
benchmark_neqf = get_sma_idc(factor_,timeperiod=20)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
```

![image-20250423104254489](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423104254489.png)

![image-20250423104305854](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423104305854.png)

## 中债国债到期收益率:10年-美国:国债收益率:10年

```python
factor_1 = pd.read_excel('./db/tby_10y.xlsx').set_index('Unnamed: 0').dropna()
factor_2 = pd.read_excel('./db/us_tby_10y.xlsx').set_index('Unnamed: 0').dropna()
factor = factor_1 - factor_2
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
benchmark_neqf = get_sma_idc(factor_,timeperiod=26)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

![image-20250102163908911](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102163908911.png)

![image-20250102163917377](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102163917377.png)

```python
# 位置原则，参考线
benchmark_neqf = get_sma_idc(factor_,timeperiod=80)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
```

![image-20250423104942694](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423104942694.png)

![image-20250423104950579](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423104950579.png)

## 美国:国债收益率:10年

```python
factor = pd.read_excel('./db/us_tby_10y.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_ = factor_.ewm(span=8,adjust=True).mean().ffill()  # ffill用来清理一周都时非工作日的情况
factor_sig = mom_rule(factor_,isMTM=False)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

![image-20250102164433789](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102164433789.png)

![image-20250102164441309](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102164441309.png)

![image-20250423105328406](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423105328406.png)

![image-20250423105345420](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423105345420.png)

## 美元指数

```python
factor = pd.read_excel('./db/dxy.xlsx').set_index('Unnamed: 0').dropna()
# ini date,预处理，补全工作日保持数据连续
factor = factor.resample('B').last().ffill()
factor_ = factor.rolling(5).apply(lambda row:row.ewm(span=5,adjust=True).mean().iloc[-1])  # 周内窗口指数平均
factor_ = factor_.resample('W-FRI').last().resample('W-MON').last().shift(-1)   # 用周一作为index
factor_sig = mom_rule(factor_,isMTM=False)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250211154422791](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250211154422791.png)

![image-20250211154830797](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250211154830797.png)

![image-20250423105743729](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423105743729.png)

![image-20250423105756059](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423105756059.png)



## VIX恐慌指数

```python
- 周频的恐慌指数在标准2倍boolinband下太难穿透了，这里简单更改了参数
factor = pd.read_excel('./db/s&p500_vix.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_['lowerband'] = factor_['CLOSE'].rolling(26).mean() - 1 * (factor_['CLOSE'].rolling(26).std())
factor_['upperband'] = factor_['CLOSE'].rolling(26).mean() + 1 * (factor_['CLOSE'].rolling(26).std())
factor_sig = positional_rule_bands(factor_,upperband='upperband',lowerband='lowerband',reference_line='CLOSE',isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130',freq='W')
```

```python
factor = pd.read_excel('./db/s&p500_vix.xlsx').set_index('Unnamed: 0').dropna()
factor_ = factor
factor_['lowerband'] = factor_['CLOSE'].rolling(120).mean() - 1.5 * (factor_['CLOSE'].rolling(120).std())
factor_['upperband'] = factor_['CLOSE'].rolling(120).mean() + 1.5 * (factor_['CLOSE'].rolling(120).std())
factor_sig = positional_rule_bands(factor_,upperband='upperband',lowerband='lowerband',reference_line='CLOSE',isMTM=True)
factor_sig = factor_sig.rolling(5).apply(lambda col:
                           1 if (col ==1).sum() > (col == -1).sum() else
                            -1 if (col == 1).sum() < (col == -1).sum() else
                            0,raw=False)
factor_sig = factor_sig.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

- 考虑到对周频数据采用bollinband可能改变数据结构影响特征提取，我们对上述方法二进行了尝试。结果并没有优于方法一，且由于vix的高波动性，造成2倍stdband无法被穿透的问题也没法得到解决。因此还是采用更加便于理解的方法一。此外，由于时间关系，下列相关boilinband直接使用方法一。

![image-20250102175011623](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102175011623.png)

![image-20250102175017607](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250102175017607.png)

```python
# 位置原则，与通道的距离
factor_['lowerband'] = factor_['CLOSE'].rolling(40).mean() - 1 * (factor_['CLOSE'].rolling(40).std())
factor_['upperband'] = factor_['CLOSE'].rolling(40).mean() + 1 * (factor_['CLOSE'].rolling(40).std())
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE',isMTM=True)
```

![image-20250423110453523](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423110453523.png)

![image-20250423110504578](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423110504578.png)



## 中债国债到期收益率:10年-1/滚动市盈率(TTM):万得全A

```python
factor_1 = pd.read_excel('./db/tby_10y.xlsx').set_index('Unnamed: 0').dropna().resample('D').ffill()  # pe非工作日也有存储（总股本变动），因此需要在这里对数据进行补齐
factor_2 = pd.read_excel('./db/pe_ttm.xlsx').set_index('Unnamed: 0').dropna()
factor = factor_1 - 1/factor_2
factor = factor.resample('D').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=7,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_ = factor
factor_['lowerband'] = factor_['CLOSE'].rolling(52).mean() - 2 * (factor_['CLOSE'].rolling(52).std())
factor_['upperband'] = factor_['CLOSE'].rolling(52).mean() + 2 * (factor_['CLOSE'].rolling(52).std())
factor_sig = positional_rule_bands(factor_,upperband='upperband',lowerband='lowerband',reference_line='CLOSE',isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250103114444747](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103114444747.png)

![image-20250103114456504](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103114456504.png)

```python
# 位置原则，与通道的距离
factor_['lowerband'] = factor_['CLOSE'].rolling(252).mean() - 1 * (factor_['CLOSE'].rolling(252).std())
factor_['upperband'] = factor_['CLOSE'].rolling(252).mean() + 1 * (factor_['CLOSE'].rolling(252).std())
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE',isMTM=True)



# 位置原则，与通道的距离
factor_['lowerband'] = factor_['CLOSE'].rolling(60).mean() - 2 * (factor_['CLOSE'].rolling(60).std())
factor_['upperband'] = factor_['CLOSE'].rolling(60).mean() + 2 * (factor_['CLOSE'].rolling(60).std())
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE',isMTM=True)
```

![image-20250423111537143](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423111537143.png)

![image-20250423111548933](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423111548933.png)

## 中债国债到期收益率:10年-万得全A股息率

```python
factor_1 = pd.read_excel('./db/tby_10y.xlsx').set_index('Unnamed: 0').dropna()
factor_2 = pd.read_excel('./db/dy_ttm.xlsx').set_index('Unnamed: 0').dropna()
factor = factor_1 - factor_2
factor = factor.resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=5,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_ = factor
factor_['lowerband'] = factor_['CLOSE'].rolling(52).mean() - 2 * (factor_['CLOSE'].rolling(52).std())
factor_['upperband'] = factor_['CLOSE'].rolling(52).mean() + 2 * (factor_['CLOSE'].rolling(52).std())
factor_sig = positional_rule_bands(factor_,upperband='upperband',lowerband='lowerband',reference_line='CLOSE',isMTM=True)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250117183444546](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250117183444546.png)

![image-20250103124339635](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103124339635.png)

```python
# 位置原则，与通道的距离
factor_['lowerband'] = factor_['CLOSE'].rolling(60).mean() - 1 * (factor_['CLOSE'].rolling(60).std())
factor_['upperband'] = factor_['CLOSE'].rolling(60).mean() + 1 * (factor_['CLOSE'].rolling(60).std())
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE',isMTM=True)
```

![image-20250423112346954](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423112346954.png)

![image-20250423112359894](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423112359894.png)

## 市净率过去三年分位数

```python
factor = pd.read_excel('./db/pb.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('D').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=7,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_sig = quantile_rule(factor_,26,0.3)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250103131754468](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103131754468.png)![image-20250103131801942](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103131801942.png)

```python
# 分位数原则
factor_sig = quantile_rule(factor_,120,0.3)
```

![image-20250423113124232](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423113124232.png)

![image-20250423113136014](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423113136014.png)



## PE过去三年分位数

- 这种分位数的反转和趋势动量是相反的，而追动量效果又很好，所以反转的效果肯定不及动量

```python
factor = pd.read_excel('./db/pe_ttm.xlsx').set_index('Unnamed: 0').dropna()
factor = factor.resample('D').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.ewm(span=7,adjust=True).mean()
factor_ = factor_.resample('W-SUN').last().resample('W-MON').last().shift(-1) 
factor_sig = quantile_rule(factor_,26,0.3)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250103131154730](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103131154730.png)

![image-20250103131210394](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250103131210394.png)

```python
# 分位数原则
factor_sig = quantile_rule(factor_,120,0.3)
```

![image-20250423113434954](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423113434954.png)

![image-20250423123124326](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250423123124326.png)



## sma

对于量价因子，我们需要讨论以下几点（以sma为例）。下面几个没后缀的都是用close直接对趋势进行捕捉的：

 - 日频一定是有效的。先看一下日频调仓的backtrader业绩和净值.

```python
# 获取close数据
factor = pd.DataFrame()
factor['CLOSE'] = df_881001['close'].resample('B').last()
factor_ = factor
# 位置原则，参考线
benchmark = get_sma_idc(factor_,timeperiod=20)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
```

![image-20250115095612282](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115095612282.png)

 ![image-20250115095620017](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115095620017.png)

 - 把日频降为周频并保持有效，简单的数据降频是无效的。业绩和净值如下：


```python
factor = pd.DataFrame()
factor['CLOSE'] = df_881001['close'].resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
factor_ = factor.rolling(5).apply(lambda row:row.ewm(span=5,adjust=True).mean().iloc[-1])  # 周内指数平均
factor_ = factor_.resample('W-FRI').last().resample('W-MON').last().shift(-1)   # 用周一作为index
# factor_ = factor_.ewm(span=8,adjust=True).mean().ffill()  # ffill用来清理一周都是非工作日的情况
# 位置原则，参考线
benchmark = get_sma_idc(factor_,timeperiod=28)  
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','W')
```

![image-20250115101139938](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115101139938.png)

![image-20250115101148642](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115101148642.png)

 - 这周频数据难以捕捉动量趋势，下面尝试用机器学习改进。出于客观限制，我们仅尝试具有较高普遍性的xgboost和randomforest模型。经测试上述两个模型均有效，randomforest效果更佳（更不容易过拟合），考虑到我们的特征较为简单，这一结论是合理的。下面展示randomforest的业绩和净值。
   - 需要注意的是，这里对信号采取unrolling作为特征进行降频并不是常用方法，实际上是很少见的方式。采用这种方法只是为了在最大程度保留原穿透策略的基础上来改进降频方法罢了。

```python
# 获取close数据
factor = pd.DataFrame()
factor['CLOSE'] = df_881001['close'].resample('B').last()
# 创建signal_·ay和pct_c特征值
factor_w = factor.resample('W-Fri').last()
factor_w['pct_c'] =  factor_w['CLOSE'].pct_change(1,fill_method=None)
factor_w['weekly_signal'] = factor_w['pct_c'].shift(-1).apply(lambda pct_c:1 if pct_c>=0 else -1 )
factor_ = factor
# 位置原则，参考线
benchmark = get_sma_idc(factor_,timeperiod=20)
factor_sig = positional_rule_reference_line(factor_,indicator_name='sma',reference_line='CLOSE',threshold=0.)
# 调整数据结构
factor_.dropna(inplace=True)
factor_ = pd.merge(left=factor_,right=factor_w,
                   left_index=True,right_index=True,how='left',suffixes=('', '_')
                  ).bfill()
factor_
# unrolling 并提取特征
weekly_data = []

def calculate_feature(signal_sum, pct_change):
    if signal_sum > 0 and pct_change >= 0:
        return 1
    elif signal_sum < 0 and pct_change < 0:
        return -1
    else:
        return 0
    

    
factor_['week'] = factor_.index.to_period('W')
for week,group in factor_.groupby('week'):
    group['signal_ewm'] = group['signal'].ewm(span=5,adjust=True).mean()
    week_features = {
        'week':week,
        'signal_1':group['signal'].iloc[0] if len(group)>0 else 0,
        'signal_2':group['signal'].iloc[1] if len(group)>1 else 0,
        'signal_3':group['signal'].iloc[2] if len(group)>2 else 0,
        'signal_4':group['signal'].iloc[3] if len(group)>3 else 0,
        'signal_5':group['signal'].iloc[4] if len(group)>4 else 0,
        
        'signal_6':group['signal_ewm'].iloc[0] if len(group)>0 else 0,
        'signal_7':group['signal_ewm'].iloc[1] if len(group)>1 else 0,
        'signal_8':group['signal_ewm'].iloc[2] if len(group)>2 else 0,
        'signal_9':group['signal_ewm'].iloc[3] if len(group)>3 else 0,
        'signal_10':group['signal_ewm'].iloc[4] if len(group)>4 else 0,
        
        'signal_sum':group['signal'].sum(),
        'pct_change':group['pct_c'].mean()
    }
    
    # 决策树不需要zscore
    week_features['pct_c*sig_s'] = calculate_feature(week_features['signal_sum'], week_features['pct_change'])
#     week_features['ret_std'] = group['pct_change'].std()

    week_features['weekly_signal'] = group['weekly_signal'].mean()

    weekly_data.append(week_features)
    
weekly_data_ = pd.DataFrame(weekly_data).dropna()

# 尝试PCA特征提取
signal_col = ['signal_{}'.format(i) for i in range(1,11)]
signal_data = weekly_data_[signal_col]

#使用pca降维
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(signal_data)

#添加pca_feature 并 剔除原始signal_n
for i in range(pca_result.shape[1]):
    weekly_data_['pca_signal_{}'.format(i+1)] = pca_result[:,i]
weekly_data_.drop(columns=signal_col,inplace=True)

# 手动分割train and test set
weekly_data_['week_start'] = weekly_data_['week'].apply(lambda row:row.start_time.date())
weekly_data_train = weekly_data_[(pd.to_datetime('2012-01-01').date()<weekly_data_['week_start']) 
                                 &( weekly_data_['week_start']<pd.to_datetime('2017-01-01').date())]
weekly_data_test = weekly_data_[weekly_data_['week_start']>pd.to_datetime('2017-01-01').date()]

# 以下仅展示随机森林的建模
# 特征和目标
X=weekly_data_train.drop(columns=['week','weekly_signal','week_start'])
y = weekly_data_train['weekly_signal'].map({-1:0,1:1})  # 转为非负分类

# train-test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=True)

#定义参数网络
param_grid = {
    'n_estimators':[100,150,200],
    'max_depth':[10,15,20],
    'min_samples_split':[5,10,15],  # sample必须split，不然会raise warning
    'min_samples_leaf':[4,6,8]
}

# 初始化模型
model = RandomForestClassifier(random_state=42)

#gridsearchcs搜索超参数
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=1
)
#运行网格搜索
grid_search.fit(X_train,y_train)

#输出最佳参数和对应分数
print('Best Parameters:',grid_search.best_params_)
print('Best Cross-Validation Accuracy:',grid_search.best_score_)

#使用最佳参数对测试集进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('Test Accuracy:',accuracy)

# 利用模型预测信号
X_test = weekly_data_test.drop(columns=['week','weekly_signal','week_start'])
y_pred = pd.Series(best_model.predict(X_test)).map({0:-1,1:1})
signal_pred = pd.DataFrame(index=weekly_data_test['week_start'],data=y_pred.values)
signal_pred.rename(columns={0:'signal'},inplace=True)
```

![image-20250115103641755](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115103641755.png)

![image-20250115103654009](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115103654009.png)

## bbi

- 日频

  ```python
  # 获取close数据
  factor = pd.DataFrame()
  factor['CLOSE'] = df_881001['close'].resample('B').last()
  factor_ = factor
  # 位置原则，参考线
  bbi_idc = get_bbi_idc(factor_)
  factor_sig = positional_rule_reference_line(factor_,indicator_name='bbi',reference_line='CLOSE',threshold=0.)
  bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
  ```
  
  ![image-20250115105955024](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115105955024.png)
  
  ![image-20250115110001594](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115110001594.png)
  
- 周频

  ```python
  脚本几乎同上
  ```

  ![image-20250115111315816](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115111315816.png)

  ![image-20250115111322037](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115111322037.png)

## cci

- 周频

  ```python
  # 获取close数据
  factor = pd.DataFrame()
  factor['CLOSE'] = df_881001['close'].resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
  factor_ = factor
  # 交叉线原则
  cci_idc = get_cci_idc(factor_)
  factor_sig = cross_rule(factor_,indicator_name='cci',value1=100,value2=-100)
  bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
  ```

  ![image-20250115113303477](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115113303477.png)

  ![image-20250115113310215](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115113310215.png)

- 周频

  ```python
  # 这里涉及到如果日频策略会产生0信号，那y应该怎么取的问题。经过尝试，我们发现，如果把0信号也纳入监督信号，结果将很差，理由可能如下：1.经过检验发现，几乎没有一周市场横盘的；2.对于当前特征维度较低的数据结构，softmax没有这个能力进行预测。
  #使用pca降维
  pca = PCA(n_components=3)  # 限定个数，过于复杂会影响随机森林模型表现（发现保留3个效果比较好）
  
  #定义参数网络
  param_grid = {
      'n_estimators':[30,50,100],
      'max_depth':[5,10,15],
      'min_samples_split':[3,5,7],  # sample必须split，不然会raise warning
      'min_samples_leaf':[8,10,12]
  }
  ```

![image-20250115142722083](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115142722083.png)

![image-20250115142729155](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115142729155.png)

## kdj

- 日频
  ```python
  # 获取close数据
  factor = pd.DataFrame()
  factor[['HIGH','CLOSE','LOW']] = df_881001[['high','close','low']].resample('B').last().ffill()  # 预处理，补全工作日保持数据连续
  factor_ = factor
  kdj_idc = get_kdj_idc(factor_)
  factor_sig = cross_rule(kdj_idc,isValue=False,ReferenceLine1='slowk',ReferenceLine2='slowd')
  bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
  ```

  ![image-20250115144805701](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115144805701.png)

  ![image-20250115144813388](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115144813388.png)

## cmo

- 日频

  ```python
  # 位置原则，与给定值的距离
  cmo_idc = get_cmo_idc(factor_)
  positional_rule_value(factor_,'cmo',isInverse=False)  
  bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
  ```

  ![image-20250115160525808](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115160525808.png)

  ![image-20250115160537955](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115160537955.png)

## dma

> 这个dma和dma_ama在震荡市都很垃圾,这是合理的。甚至本身用收盘价算趋势就不如用成交量来的合理

- 日频

```python
dma_idc = get_dma_idc(factor_)
factor_sig =  positional_rule_value(factor_,'dma')
```

![image-20250115172223273](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115172223273.png)

![image-20250115172237296](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115172237296.png)

## dma_ama

- 日频

```python
# dma_ama inclusive
dma_idc = get_ama_idc(factor_)
factor_sig = dma_ama_rule(factor_)
```

![image-20250115172453304](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115172453304.png)

![image-20250115172502903](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115172502903.png)

## trix

> 四倍移动平均，能捕捉出来啥啊？

- 日频

```python
# 位置原则，与给定值的距离
dma_idc = get_trix_idc(factor_)
factor_sig = positional_rule_value(factor_,'trix_diff')
```

![image-20250115162230007](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115162230007.png)![image-20250115162244316](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115162244316.png)

## roc

- 日频

```python
# 位置原则，与给定值的距离
dma_idc = get_roc_idc(factor_)
factor_sig = positional_rule_value(factor_,'roc')
```

![image-20250115162537221](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115162537221.png)

![image-20250115162605932](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115162605932.png)

## bbands

- 日频,效果一般，因为波动还不够

```python
# # 位置原则，与通道的距离
bbands_idc = get_bbands_idc(factor_)
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE')
```

![image-20250115163139723](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115163139723.png)

![image-20250115163148665](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115163148665.png)

## rsi

- 日频

```python
# 位置原则，与给定值的距离
rsi_idc = get_rsi_idc(factor_)
factor_sig =  positional_rule_value(factor_,'rsi',isInverse=True,value1=20,value2=80)
```

![image-20250115175147621](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115175147621.png)

![image-20250115175158952](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115175158952.png)

## macd

- 日频

```python
macd_idc = get_macd_idc(factor_)
factor_sig =  positional_rule_value(factor_,'macd')
```

![image-20250115180855988](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115180855988.png)

![image-20250115180904657](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115180904657.png)

## bbands_iccfe

- 日频

```python
errorcode, factor = w.wsd("IC.CFE", "anal_basisannualyield","2012-01-01", "2024-11-30", "Fill=Previous",usedf=True) 
factor.to_excel('./db/ic_cfe.xlsx')
factor.rename(columns={'ANAL_BASISANNUALYIELD':'CLOSE'},inplace=True)
# # 位置原则，与通道的距离
bbands_idc_iccfe = get_bbands_idc(factor_,nbdevup=1, nbdevdn=1)  # 获取indicator
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE')
```

![](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115183758556.png)

![image-20250115183813054](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115183813054.png)

## bbands_ifcfe

- 日频。300对日频趋势的捕捉不及500是合理的。

```python
errorcode, factor = w.wsd("IF.CFE", "anal_basisannualyield","2012-01-01", "2024-11-30", "Fill=Previous",usedf=True) 
factor.to_excel('./db/if_cfe.xlsx')
factor.rename(columns={'ANAL_BASISANNUALYIELD':'CLOSE'},inplace=True)
factor_ = factor
# # 位置原则，与通道的距离
bbands_idc_ifcfe = get_bbands_idc(factor_,nbdevup=1, nbdevdn=1)  # 获取indicator
factor_sig = positional_rule_bands(factor_,'upperband','lowerband','CLOSE')

```

![image-20250115185942948](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115185942948.png)

![image-20250115185949472](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250115185949472.png)

## oiratio

- 日频

```python
errorcode, factor = w.wsd("510050.SH", " oiratio", '2012-01-01','2024-11-30',"SettlementMonth=0",usedf=True)  
factor = pd.read_excel('./db/50etf_oiratio.xlsx',index_col=0)
factor = factor.resample('B').last().ffill()
factor_ = factor
#同比变动原则
factor_['oiratio_'] = factor_['OIRATIO'].rolling(20).mean()
factor_sig = yoy_mtm_rule(factor_,'oiratio_',shift_p=20)
bt_data = get_bt_data(df_881001,factor_sig,'20170101','20241130','D')
```

![image-20250116110940346](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250116110940346.png)

![image-20250116110950261](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250116110950261.png)

## Volume

### volume-price momentum

- 日频。还得是共振啊。

```python
factor = pd.read_excel('./db/881001_aug.xlsx',index_col=0)[['close','volume']]
factor = factor.resample('B').last().ffill()
factor_ = factor
factor_sig = volume_price_momentum_rule(factor_)
```

![image-20250121093838085](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250121093838085.png)

![image-20250116124739874](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250116124739874.png)

### macd_vol

- 日频。结果和close类似，因此目前就不对volume重复close的测试结果了，此外我们还测试了：

  - **column_spike**。由于波动过于剧烈，暂未找到处理方法

  - **量价背离。**效果一般，没找到处理方法

  - **on balance value。**是一个有效跟踪的数据，但没找到有信号产生的策略

    ![image-20250116124542528](E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250116124542528.png)

  但是需要说明的是，volume和turnover是一个很有挖掘意义的方向，后续将继续完成目前尚未完成的工作。

# 因子合成

- Testing对截至20250211模型需要的因子进行了相对全面的区间回测。并根据筛选，选出了6大类共22个因子。本节目标，是通过尝试，找到合适的因子合成方法。

- 市场上常见的合成方法有：全因子等权、大类因子等权后再内部等权、IC\ICIR加权（仅特征类，信号类不适用）、非线性

- 首先，我们的因子具有多个频率（月、周、日），且上一版本的模型发现，单纯从信号角度对全部因子进行线性加权容易因特征丢失导致最终结果趋于平庸，因此我们首先尝试非线性加权的方法，具体参考《华泰证券-华泰证券金工深度研究：基于全频段量价特征的选股模型（2023-12）》中“使用GRU基于多任务学习的低频量价模型”章节，构造低频模型。

  > [20231208-华泰证券-华泰证券金工深度研究：基于全频段量价特征的选股模型.pdf](20231208-华泰证券-华泰证券金工深度研究：基于全频段量价特征的选股模型.pdf) 

  - 关于GRU的使用：经过金融市场检验，GRU比lstm在金融序列的实证结果表现更好**（后面需要收集资料证明这一点）**

- 模型介绍：

  <img src="E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250611185946343.png" alt="image-20250611185946343" style="zoom:25%;" />

  

  <img src="E:\ZYXT\files\script\timingFactors_update\files\assets\image-20250611191411864.png" alt="image-20250611191411864" style="zoom: 33%;" />

-  参数调试：使用交叉熵作为损失函数。结合实际需求，我们对未来5日和未来20日的收益率进行预测，并进行参数调优，采取grid search+Timeseries Split方法进行超参数搜索，最终选择参数选择为：


## 对经济数据相关性的讨论

  对于基本面、政策面等宏观经济方面的数据，我们不考虑其数据的相关性。核心原因在于，当我们把这些数据纳入模型时，我们考虑的是其从宏观层面对经济的表征，而宏观数据有其本身的经济学意义。相对于其经济学意义，这些数据在数学上可能的相关性则不必过分担心，甚至从某种程度上，宏观数据的相关性本身就是经济意义的体现。



## 数据清洗和特征提取

- 首先实现从wind端口提取数据，存入database，用于后续更新。
- 数据特征提取时，需要注意对重复值的清洗。

# VScode整理

- 20250603首先把上述内容整理到vscode中形成结构规范的factor-test代码，然后再进行后续操作

# 关于因子有效性筛选的讨论

> [(99+ 封私信 / 6 条消息) 如何评价一个信号型因子的有效性？ - 搜索结果 - 知乎](https://www.zhihu.com/search?type=content&q=如何评价一个信号型因子的有效性？)

- 对象的定义：
  - 特征（因子、指标）：通过原始数据提取出来的数据特征
  - 信号（-1、0、1）：根据特征加工出来的交易指导
  - 策略：按照规则把信号组合起来指导实际投资
- 常用评价指标：
  - 特征【预测能力、与标的的相关性】：IC/IR/相关性等
  - 信号【预测有效性、稳定性】：胜率赔率、偏度、峰度等
  - 策略【最终的赚钱能力和风险收益特征】：年化收益、夏普比率、最大回撤等指标

无论如何，我们认为 “只要长期期望收益为正，那就是好信号”，因此选用信号指导下标的业绩的夏普比率作为筛选的标准对信号进行筛选。

>  



















































