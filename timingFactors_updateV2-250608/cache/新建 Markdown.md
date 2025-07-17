```PYTHON
# PPI:全部工业品:当月同比_mf09
error_code, ppi = w.edb("M0001227",p_start_month,end_date,"Fill=Previous", usedf=True)
ppi.to_sql(name="ppi",con=engine,if_exists='append',index=True,index_label='Date')
# CPI:当月同比_mf09
error_code, cpi = w.edb("M0000612",p_start_month,end_date,"Fill=Previous", usedf=True)
cpi.to_sql(name="cpi",con=engine,if_exists='append',index=True,index_label='Date')
# PMI：新出口订单_mf27
error_code, pmi_neo = w.edb("M0017129",p_start_month,end_date,"Fill=Previous", usedf=True)
pmi_neo.to_sql(name="pmi_neo",con=engine,if_exists='append',index=True,index_label='Date')
#中国:制造业PMI_mf27
error_code, pmi = w.edb("M0017126",p_start_month,end_date,"Fill=Previous", usedf=True)
pmi.to_sql(name="pmi",con=engine,if_exists='append',index=True,index_label='Date')
# 服务业PMI_mf27
error_code, pmi_sv = w.edb("M5207838",p_start_month,end_date,"Fill=Previous", usedf=True)
pmi_sv.to_sql(name="pmi_sv",con=engine,if_exists='append',index=True,index_label='Date')
# PMI:生产_mf27
error_code, pmi_p = w.edb("M0017127",p_start_month,end_date,"Fill=Previous", usedf=True)
pmi_p.to_sql(name="pmi_p",con=engine,if_exists='append',index=True,index_label='Date')
# 中国:社会消费品零售总额:当月同比_mf17
error_code, rs_yoy = w.edb("M0001428",p_start_month,end_date,"Fill=Previous", usedf=True)
rs_yoy.to_sql(name="rs_yoy",con=engine,if_exists='append',index=True,index_label='Date')
# 中国:固定资产投资完成额:基础设施建设投资(不含电力):累计同比_mf17
error_code, fai_infr_ytd = w.edb("M5531328",p_start_month,end_date,"Fill=Previous", usedf=True)
fai_infr_ytd.to_sql(name="fai_infr_ytd",con=engine,if_exists='append',index=True,index_label='Date')
# 银行间质押式回购加权利率:7 天_daf
error_code, r007 = w.edb("M0041653",p_start_date,end_date,"Fill=Previous", usedf=True)
r007.to_sql(name="r007",con=engine,if_exists='append',index=True,index_label='Date')
# 中长期贷款余额同比_mf17, 涉及到计算同比，需要先写入后计算
# 写入raw date db
error_code, m_ltl = w.edb("M0043418",p_start_month,end_date,"Fill=Previous", usedf=True)
m_ltl.to_sql(name="m&ltl",con=engine,if_exists='append',index=True,index_label='Date')
#计算yoy，写入yoy db
m_ltl_yoy = pd.read_sql('SELECT * FROM `m&ltl`', engine,index_col='Date').pct_change(12)
m_ltl_yoy.to_sql(name="m&ltl_yoy",con=engine,if_exists='replace',index=True,index_label='Date')
# 中国:社会融资规模存量:同比_mf14
error_code, tsf_yoy = w.edb("M5525763",p_start_month,end_date,"Fill=Previous", usedf=True)
tsf_yoy.to_sql(name="tsf_yoy",con=engine,if_exists='append',index=True,index_label='Date')
# 中国:新成立基金份额:偏股型_daf
error_code, neqf = w.edb("M0060433",p_start_date,end_date,"Fill=Previous", usedf=True)
neqf.to_sql(name="neqf",con=engine,if_exists='append',index=True,index_label='Date')
# USDCNH:即期汇率_daf
error_code, usdcnh_spot = w.edb("M0290205",p_start_date,end_date,"Fill=Previous", usedf=True)
usdcnh_spot.to_sql(name="usdcnh_spot",con=engine,if_exists='append',index=True,index_label='Date')
# 中债国债到期收益率:10年-美国:国债收益率:10年_daf
error_code, tby_10y = w.edb("S0059749,G0000891",p_start_date,end_date,"Fill=Previous", usedf=True)
tby_10y_diff = tby_10y['S0059749'] - tby_10y['G0000891']
tby_10y_diff.to_sql(name="tby_10y_diff",con=engine,if_exists='append',index=True,index_label='Date')
# 美元指数_daf
error_code, dxy = w.edb("M0000271",p_start_date,end_date,"Fill=Previous", usedf=True)
dxy.to_sql(name="dxy",con=engine,if_exists='append',index=True,index_label='Date')
# 美国:国债收益率:10年_daf
error_code, us_tby_10y = w.edb("G0000891",p_start_date,end_date,"Fill=Previous", usedf=True)
us_tby_10y.to_sql(name="us_tby_10y",con=engine,if_exists='append',index=True,index_label='Date')
# VIX恐慌指数_daf
error_code, s_p500_vix = w.edb("G0003892",p_start_date,end_date,"Fill=Previous", usedf=True)
s_p500_vix.to_sql(name="s&p500_vix",con=engine,if_exists='append',index=True,index_label='Date')
# 中债国债到期收益率:10年-1/滚动市盈率(TTM):万得全A_daf
error_code, tby_pe = w.edb("S0059749,M0330161",p_start_date,end_date,"Fill=Previous", usedf=True)# pe非工作日也有存储（总股本变动）
tby_recipe = tby_pe['S0059749'] - 1/tby_pe['M0330161']
tby_recipe.to_sql(name="tby_recipe",con=engine,if_exists='append',index=True,index_label='Date')
# 市净率过去三年分位数_daf
error_code, pb = w.edb("M0330166",p_start_date,end_date,"Fill=Previous", usedf=True)
pb.to_sql(name="pb",con=engine,if_exists='append',index=True,index_label='Date')
# PE过去三年分位数_daf
error_code, pe_ttm = w.edb("M0330161",p_start_date,end_date,"Fill=Previous", usedf=True)
pe_ttm.to_sql(name="pe_ttm",con=engine,if_exists='append',index=True,index_label='Date')
# 881001.WI_daf
error_code, nv_index_e = w.wsd("881001.WI", "open,high,low,close,volume", p_start_date,end_date, usedf=True)
nv_index_e['ret'] = nv_index_e['CLOSE'].pct_change(1)
nv_index_e.to_sql(name='881001_wi',con=engine,if_exists='append',index=True,index_label='Date')
# CBA00101.CS_daf
erro_code, nv_index_b = w.wsd("CBA00101.CS", "close",p_start_date,end_date, usedf=True)
nv_index_b['ret'] = nv_index_b['CLOSE'].pct_change(1)
nv_index_b.to_sql(name='cba00101_cs',con=engine,if_exists='append',index=True,index_label='Date')
# ic.cfe_daf
errorcode, ic_cfe = w.wsd("IC.CFE", "anal_basisannualyield",p_start_date,end_date, "Fill=Previous",usedf=True)
ic_cfe.to_sql(name="ic_cfe",con=engine,if_exists='append',index=True,index_label='Date')
# if.cfe_daf
errorcode, if_cfe = w.wsd("IF.CFE", "anal_basisannualyield",p_start_date, end_date, "Fill=Previous",usedf=True)
if_cfe.to_sql(name="if_cfe",con=engine,if_exists='append',index=True,index_label='Date')
# oirario
errorcode, oiratio = w.wsd("510050.SH", " oiratio", p_start_date,end_date,"SettlementMonth=0",usedf=True) 
oiratio.to_sql(name="oiratio",con=engine,if_exists='append',index=True,index_label='Date')



```

