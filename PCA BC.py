import pandas as pd
import pandas_datareader as pdr
from reportlab.pdfgen import canvas
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from reportlab.lib.colors import red, blue, white, black, grey, green, greenyellow
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import scipy.stats as sp
from sklearn.decomposition import PCA
from matplotlib.dates import date2num
import yfinance as yf

location = '/Users/User/Desktop/Practice Python'
dateToday = date.today()

def end_of_month(dates):
    delta_m = relativedelta(months=1)
    delta_d = timedelta(days=1)
    next_month = dates + delta_m
    end_of_month = date(next_month.year, next_month.month, 1) - delta_d
    return end_of_month

end_date = end_of_month((dateToday - relativedelta(months=1)))

def fix_dates(series):
    missing_months = relativedelta(end_of_month(dateToday-relativedelta(months=1)), series.index[-1])
    missing_months = (missing_months.years * 12) + missing_months.months
    for d in range(missing_months):
        eolastmonth = dateToday - relativedelta(months=missing_months-d)
        eolastmonth = end_of_month(eolastmonth)
        eomonth = list(series.index)
        eomonth.append(eolastmonth)
        eomonth = pd.Series(eomonth)
        series.index = eomonth[len(eomonth) - len(series):]
    return series

def winsorize(s, limits):
    return s.clip(lower=s.squeeze().quantile(limits[0], interpolation='lower'), 
                  upper=s.squeeze().quantile(1-limits[1], interpolation='higher'))

def norm(economic_data):
    avg = economic_data.mean()
    std = economic_data.std()
    zscore = (economic_data - avg) / std
    return zscore

def zscore(economic_data):
    economic_data = winsorize(economic_data, limits=(0.01,0.01))
    avg = economic_data.mean()
    std = economic_data.std()
    zscore = (economic_data - avg) / std
    normalized = zscore.ewm(halflife=6,min_periods=6).mean()
    normalized.index = economic_data.index
    normalized = normalized.dropna()
    return normalized

def multi_period_return(return_stream):
    return np.prod(1 + return_stream) - 1 

#Personal Savings Rate
personal_savings = pdr.get_data_fred('PSAVERT', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
personal_savings = personal_savings.resample('M').last()
adj_personal_savings = fix_dates(personal_savings)
norm_adj_personal_savings = zscore(adj_personal_savings)
norm_adj_personal_savings.columns = ['Personal Savings Rate']

#Bank Credit
bank_credit = pdr.get_data_fred('H8B1001NCBCMG', start='1947-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
bank_credit = bank_credit.resample('M').last()
adj_bank_credit = fix_dates(bank_credit)
norm_adj_bank_credit = zscore(adj_bank_credit)
norm_adj_bank_credit.columns = ['Bank Credit']

#Motor Vehicle Loans
vehicle_loans = pdr.get_data_fred('DTBOVLRXDFBANA', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
vehicle_loans = vehicle_loans.resample('M').last()
adj_vehicle_loans = fix_dates(vehicle_loans)
nomr_adj_vehicle_loans = zscore(adj_vehicle_loans)
nomr_adj_vehicle_loans.columns = ['Motor Vehicle Loans']

#Transfer Receipts
trans_receipt = pdr.get_data_fred('B931RC1', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
trans_receipt = trans_receipt.resample('M').last()
adj_trans_receipt = np.log(trans_receipt).diff()
adj_trans_receipt = fix_dates(adj_trans_receipt)
norm_adj_trans_receipt = zscore(adj_trans_receipt)
norm_adj_trans_receipt.columns = ['Net Transfer Receipts']

#Part-Time Employees
part_time = pdr.get_data_fred('LNS12032198', start='1955-05-01', end=end_of_month(dateToday - relativedelta(months=1)))
part_time = part_time.resample('M').last()
adj_part_time = np.log(part_time).diff()
adj_part_time = fix_dates(adj_part_time)
norm_adj_part_time = zscore(adj_part_time)
norm_adj_part_time.columns = ['Part-Time Employees']

#Commercial & Industrial Loans
bus_loans = pdr.get_data_fred('BUSLOANS', start='1947-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
bus_loans = bus_loans.resample('M').last()
adj_bus_loans = np.log(bus_loans).diff()
adj_bus_loans = fix_dates(adj_bus_loans)
norm_adj_bus_loans = zscore(adj_bus_loans)
norm_adj_bus_loans.columns = ['Commercial & Industrial Loans']

#Imports
imports = pdr.get_data_fred('XTIMVA01USM657S', start='1960-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_imports = imports.resample('M').last()
adj_imports = fix_dates(adj_imports)
norm_adj_imports = zscore(adj_imports)
norm_adj_imports.columns = ['Imports']

#Exports
exports = pdr.get_data_fred('XTEXVA01USM657S', start='1960-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_exports = exports.resample('M').last()
adj_exports = fix_dates(adj_exports)
norm_adj_exports = zscore(adj_exports)
norm_adj_exports.columns = ['Exports']

#Hourly Earnings
earnings = pdr.get_data_fred('USAHOUREAMISMEI', start='1960-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
earnings = earnings.resample('M').last()
adj_earnings = np.log(earnings).diff()
adj_earnings = fix_dates(adj_earnings)
norm_adj_earnings = zscore(adj_earnings)
norm_adj_earnings.columns = ['Hourly Earnings']

#CPI
cpi = pdr.get_data_fred('CPALTT01USM661S', start='1960-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
cpi = cpi.resample('M').last()
adj_cpi = np.log(cpi).diff()
adj_cpi = fix_dates(adj_cpi)
norm_adj_cpi = zscore(adj_cpi)
norm_adj_cpi.columns = ['CPI']

#Manufacturing Confidence
man_conf = pdr.get_data_fred('BSCICP03USM665S', start='1960-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_man_conf = man_conf.resample('M').last()
adj_man_conf = fix_dates(adj_man_conf)
norm_adj_man_conf = zscore(adj_man_conf)
norm_adj_man_conf.columns = ['Manufacturing Confidence']

#Price Pressure Index
pp = pdr.get_data_fred('STLPPM', start='1990-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_pp = pp.resample('M').last()
norm_adj_pp = zscore(adj_pp)
norm_adj_pp.columns = ['Price Pressure Index']

#AAA Credit Spread
aaa_spread = pdr.get_data_fred('AAA10YM', start='1985-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
aaa_spread = aaa_spread.resample('M').last()
adj_aaa_spread = np.log(aaa_spread)
norm_adj_aaa_spread = zscore(adj_aaa_spread)
norm_adj_aaa_spread.columns = ['AAA Spread']

#BBB Credit Spread
bbb_spread = pdr.get_data_fred('BAA10YM', start='1955-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
bbb_spread = bbb_spread.resample('M').last()
adj_bbb_spread = np.log(bbb_spread)
norm_adj_bbb_spread = zscore(adj_bbb_spread)
norm_adj_bbb_spread.columns = ['BBB Spread']

#Wheat
wheat = pdr.get_data_fred('APU0000703112', start='1984-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
wheat = wheat.resample('M').last()
wheat = wheat.interpolate(method ='linear', limit_direction ='forward')
adj_wheat = np.log(wheat).diff()
adj_wheat = fix_dates(adj_wheat)
norm_adj_wheat = zscore(adj_wheat)
norm_adj_wheat.columns = ['Wheat']

#Ground Beef
beef = pdr.get_data_fred('APU0000703112', start='1984-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
beef = beef.resample('M').last()
beef = beef.interpolate(method ='linear', limit_direction ='forward')
adj_beef = np.log(beef).diff()
adj_beef = fix_dates(adj_beef)
norm_adj_beef = zscore(adj_beef)
norm_adj_beef.columns = ['Ground Beef']

#Crude Oil
crude = pdr.get_data_fred('WTISPLC', start='1946-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
crude = crude.resample('M').last()
adj_crude = np.log(crude).diff()
adj_crude = fix_dates(adj_crude)
norm_adj_crude = zscore(adj_crude)
norm_adj_crude.columns = ['WTI Crude']

#Copper
copper = pdr.get_data_fred('PCOPPUSDM', start='1990-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
copper = copper.resample('M').last()
adj_copper = np.log(copper).diff()
adj_copper = fix_dates(adj_copper)
norm_adj_copper = zscore(adj_copper)
norm_adj_copper.columns = ['Copper']

#30Y Fixed
fixed_mortgage = pdr.get_data_fred('MORTGAGE30US', start='1971-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
fixed_mortgage = fixed_mortgage.resample('M').last()
adj_fixed_mortgage = np.log(fixed_mortgage).diff()
norm_adj_fixed_mortgage = zscore(adj_fixed_mortgage)
norm_adj_fixed_mortgage.columns = ['30Y Fixed Rate']

#Industrial Production
all_ind_prod = pdr.get_data_fred('INDPRO', start='1919-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
all_ind_prod = all_ind_prod.resample('M').last()
adj_all_ind_prod = np.log(all_ind_prod).diff()
adj_all_ind_prod = fix_dates(adj_all_ind_prod)
norm_adj_all_ind_prod = zscore(adj_all_ind_prod)
norm_adj_all_ind_prod.columns = ['Industrial Production']

#Industrial Production - Manufacturing
man_ind_prod = pdr.get_data_fred('IPMANSICS', start='1919-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
man_ind_prod = man_ind_prod.resample('M').last()
adj_man_ind_prod = np.log(man_ind_prod).diff()
adj_man_ind_prod = fix_dates(adj_man_ind_prod)
norm_adj_man_ind_prod = zscore(adj_man_ind_prod)
norm_adj_man_ind_prod.columns = ['Industrial Production - Manufacturing']

#Industrial Production - Durable
dur_man = pdr.get_data_fred('IPDMAN', start='1972-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
dur_man = dur_man.resample('M').last()
adj_dur_man = np.log(dur_man).diff()
adj_dur_man = fix_dates(adj_dur_man)
norm_adj_dur_man = zscore(adj_dur_man)
norm_adj_dur_man.columns = ['Industrial Production - Durable']

#Industrial Production - Non-durable
nondur_man = pdr.get_data_fred('IPNMAN', start='1972-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
nondur_man = nondur_man.resample('M').last()
adj_nondur_man = np.log(nondur_man).diff()
adj_nondur_man = fix_dates(adj_nondur_man)
norm_adj_nondur_man = zscore(adj_nondur_man)
norm_adj_nondur_man.columns = ['Industrial Production - Non-durable']

#Capacity Utilization
capacity = pdr.get_data_fred('TCU', start='1967-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_capacity = capacity.resample('M').last()
adj_capacity = fix_dates(adj_capacity)
norm_adj_capacity = zscore(adj_capacity)
norm_adj_capacity.columns = ['Capacity Utilization']

#Real Personal Income
real_income = pdr.get_data_fred('W875RX1', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
real_income = real_income.resample('M').last()
adj_real_income = np.log(real_income).diff()
adj_real_income = fix_dates(adj_real_income)
norm_adj_real_income = zscore(adj_real_income)
norm_adj_real_income.columns = ['Real Personal Income']

#Nonfarm Payrolls
all_payrolls = pdr.get_data_fred('PAYEMS', start='1939-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
all_payrolls = all_payrolls.resample('M').last()
adj_all_payrolls = np.log(all_payrolls).diff()
norm_adj_all_payrolls = zscore(adj_all_payrolls)
norm_adj_all_payrolls.columns = ['Nonfarm Payrolls']

#Service-Providing Payrolls
service_payrolls = pdr.get_data_fred('SRVPRD', start='1939-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
service_payrolls = service_payrolls.resample('M').last()
adj_service_payrolls = np.log(service_payrolls).diff()
norm_adj_service_payrolls = zscore(adj_service_payrolls)
norm_adj_service_payrolls.columns = ['Service-Providing Payrolls']

#Goods-Producing Payrolls
goods_payrolls = pdr.get_data_fred('USGOOD', start='1939-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
goods_payrolls = goods_payrolls.resample('M').last()
adj_goods_payrolls = np.log(goods_payrolls).diff()
norm_adj_goods_payrolls = zscore(adj_goods_payrolls)
norm_adj_goods_payrolls.columns = ['Goods-Producing Payrolls']

#Unemployment Rate
unemployment = pdr.get_data_fred('UNRATE', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_unemployment = unemployment.resample('M').last()
norm_adj_unemployment = zscore(adj_unemployment)
norm_adj_unemployment.columns = ['Unemployment Rate']

#Unemployment Rate 25-54 Years
core_unemployment = pdr.get_data_fred('LNS14000060', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_core_unemployment = core_unemployment.resample('M').last()
norm_adj_core_unemployment = zscore(adj_core_unemployment)
norm_adj_core_unemployment.columns = ['Unemployment Rate 25-54 Years']

#Unemployment Rate - High School
hs_unemployment = pdr.get_data_fred('LNS14027660', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
adj_hs_unemployment = hs_unemployment.resample('M').last()
norm_adj_hs_unemployment = zscore(adj_hs_unemployment)
norm_adj_hs_unemployment.columns = ['Unemployment Rate - High School']

#Initial Unemployment Claims
init_claims = pdr.get_data_fred('IC4WSA', start='1967-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
init_claims = init_claims.resample('M').last()
adj_init_claims = init_claims.diff()
norm_adj_init_claims = zscore(adj_init_claims)
norm_adj_init_claims.columns = ['Initial Unemployment Claims']

#Continuing Unemployment Claims
con_claims = pdr.get_data_fred('CC4WSA', start='1967-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
con_claims = con_claims.resample('M').last()
adj_con_claims = con_claims.diff()
norm_adj_con_claims = zscore(adj_con_claims)
norm_adj_con_claims.columns = ['Continuing Unemployment Claims']

#Duration Unemployment
dur_unemployment = pdr.get_data_fred('UEMPMEAN', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
dur_unemployment = dur_unemployment.resample('M').last()
adj_dur_unemployment = dur_unemployment.diff()
norm_adj_dur_unemployment = zscore(adj_dur_unemployment)
norm_adj_dur_unemployment.columns = ['Duration Unemployment']

#Percent Unemployed Long Term
long_unemploy = pdr.get_data_fred('LNS13025703', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
long_unemploy = long_unemploy.resample('M').last()
adj_long_unemploy = np.log(long_unemploy)
norm_adj_long_unemploy = zscore(adj_long_unemploy)
norm_adj_long_unemploy.columns = ['Percent Unemployed 27 Weeks & over']

#Average Manufacturing Hours Worked
avg_hours = pdr.get_data_fred('AWHMAN', start='1939-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
avg_hours = avg_hours.resample('M').last()
adj_avg_hours = avg_hours.diff()
norm_adj_avg_hours = zscore(adj_avg_hours)
norm_adj_avg_hours.columns = ['Average Manufacturing Hours Worked']

#Average Overtime Manufacturing Hours Worked
avg_ot_hours = pdr.get_data_fred('AWOTMAN', start='1956-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
avg_ot_hours = avg_ot_hours.resample('M').last()
adj_avg_ot_hours = avg_ot_hours.diff()
norm_adj_avg_ot_hours = zscore(adj_avg_ot_hours)
norm_adj_avg_ot_hours.columns = ['Average Overtime Manufacturing Hours Worked']

#Nonagricultural Employment 
nonag_employment = pdr.get_data_fred('LNS12035019', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
nonag_employment = nonag_employment.resample('M').last()
adj_nonag_employment = np.log(nonag_employment).diff()
norm_adj_nonag_employment = zscore(adj_nonag_employment)
norm_adj_nonag_employment.columns = ['Nonagricultural Employment']

#Real PCE
pce = pdr.get_data_fred('DPCERA3M086SBEA', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
pce = pce.resample('M').last()
adj_pce = np.log(pce).diff()
adj_pce = fix_dates(adj_pce)
norm_adj_pce = zscore(adj_pce)
norm_adj_pce.columns = ['Real PCE']

#Real PCE - Durable Goods
pce_durable = pdr.get_data_fred('DDURRA3M086SBEA', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
pce_durable = pce_durable.resample('M').last()
adj_pce_durable = np.log(pce_durable).diff()
adj_pce_durable = fix_dates(adj_pce_durable)
norm_adj_pce_durable = zscore(adj_pce_durable)
norm_adj_pce_durable.columns = ['Real PCE - Durable Goods']

#Real PCE - Nondurable Goods
pce_nondurable = pdr.get_data_fred('DNDGRA3M086SBEA', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
pce_nondurable = pce_nondurable.resample('M').last()
adj_pce_nondurable = np.log(pce_nondurable).diff()
adj_pce_nondurable = fix_dates(adj_pce_nondurable)
norm_adj_pce_nondurable = zscore(adj_pce_nondurable)
norm_adj_pce_nondurable.columns = ['Real PCE - Nondurable Goods']

#Real PCE - Services
pce_services = pdr.get_data_fred('DSERRA3M086SBEA', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
pce_services = pce_services.resample('M').last()
adj_pce_services = np.log(pce_services).diff()
adj_pce_services = fix_dates(adj_pce_services)
norm_adj_pce_services = zscore(adj_pce_services)
norm_adj_pce_services.columns = ['Real PCE - Services']

#Building Permits
permits = pdr.get_data_fred('PERMIT', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
permits = permits.resample('M').last()
adj_permits = np.log(permits).diff()
adj_permits = fix_dates(adj_permits)
norm_adj_permits = zscore(adj_permits)
norm_adj_permits.columns = ['Building Permits']

#New Housing Starts
house = pdr.get_data_fred('HOUST', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
house = house.resample('M').last()
adj_house = np.log(house).diff()
adj_house = fix_dates(adj_house)
norm_adj_house = zscore(adj_house)
norm_adj_house.columns = ['New Housing Starts']

#Real Retail & Food Sales
retailfood_sales = pdr.get_data_fred('RRSFS', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
retailfood_sales = retailfood_sales.resample('M').last()
adj_retailfood_sales = np.log(retailfood_sales).diff()
adj_retailfood_sales = fix_dates(adj_retailfood_sales)
norm_adj_retailfood_sales = zscore(adj_retailfood_sales)
norm_adj_retailfood_sales.columns = ['Real Retail & Food Sales']

#Real Sales
real_sales = pdr.get_data_fred('CMRMTSPL', start='1967-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
real_sales = real_sales.resample('M').last()
adj_real_sales = np.log(real_sales).diff()
adj_real_sales = fix_dates(adj_real_sales)
norm_adj_real_sales = zscore(adj_real_sales)
norm_adj_real_sales.columns = ['Real Sales']

#Inventory to Sales Ratio
inv_to_sales = pdr.get_data_fred('ISRATIO', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
inv_to_sales = inv_to_sales.resample('M').last()
adj_inv_to_sales = fix_dates(inv_to_sales)
norm_adj_inv_to_sales = zscore(adj_inv_to_sales)
norm_adj_inv_to_sales.columns = ['Inventory to Sales Ratio']

#Manufacturing Inventory to Sales Ratio
man_inv_to_sales = pdr.get_data_fred('MNFCTRIRSA', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
man_inv_to_sales = man_inv_to_sales.resample('M').last()
adj_man_inv_to_sales = fix_dates(man_inv_to_sales)
norm_adj_man_inv_to_sales = zscore(adj_man_inv_to_sales)
norm_adj_man_inv_to_sales.columns = ['Manufacturing Inventory to Sales Ratio']

#Wholesale Inventory to Sales Ratio
wholesale_inv_to_sales = pdr.get_data_fred('WHLSLRIRSA', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
wholesale_inv_to_sales = wholesale_inv_to_sales.resample('M').last()
adj_wholesale_inv_to_sales = fix_dates(wholesale_inv_to_sales)
norm_adj_wholesale_inv_to_sales = zscore(adj_wholesale_inv_to_sales)
norm_adj_wholesale_inv_to_sales.columns = ['Wholesale Inventory to Sales Ratio']

#Manufacturing New Orders
new_orders = pdr.get_data_fred('AMTMNO', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
new_orders = new_orders.resample('M').last()
adj_new_orders = np.log(new_orders).diff()
adj_new_orders = fix_dates(adj_new_orders)
norm_adj_new_orders = zscore(adj_new_orders)
norm_adj_new_orders.columns = ['Manufacturing New Orders']

#Manufacturing New Orders - Capital Goods
capital_orders = pdr.get_data_fred('NEWORDER', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
capital_orders = capital_orders.resample('M').last()
adj_capital_orders = np.log(capital_orders).diff()
adj_capital_orders = fix_dates(adj_capital_orders)
norm_adj_capital_orders = zscore(adj_capital_orders)
norm_adj_capital_orders.columns = ['Manufacturing New Orders - Capital Goods']

#Manufacturing New Orders - Consumer Goods
consumer_orders = pdr.get_data_fred('ACOGNO', start='1992-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
consumer_orders = consumer_orders.resample('M').last()
adj_consumer_orders = np.log(consumer_orders).diff()
adj_consumer_orders = fix_dates(adj_consumer_orders)
norm_adj_consumer_orders = zscore(adj_consumer_orders)
norm_adj_consumer_orders.columns = ['Manufacturing New Orders - Consumer Goods']

#Slope of Yield Curve
slope = pdr.get_data_fred('T10YFF', start='1962-01-02', end=end_of_month(dateToday - relativedelta(months=1)))
slope = slope.resample('M').last()
norm_adj_slope = zscore(slope)
norm_adj_slope.columns = ['Slope of Yield Curve']
slope_chg = slope.diff()
norm_adj_slope_chg = zscore(slope_chg)
norm_adj_slope_chg.columns = ['Slope of Yield Curve Change']

#M2 Money Supply
m2 = pdr.get_data_fred('M2SL', start='1959-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
m2 = m2.resample('M').last()
adj_m2 = np.log(m2).diff()
adj_m2 = fix_dates(adj_m2)
norm_adj_m2 = zscore(adj_m2)
norm_adj_m2.columns = ['M2 Money Supply']

#Consumer Credit
consumer_credit = pdr.get_data_fred('TOTALSL', start='1943-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
consumer_credit = consumer_credit.resample('M').last()
adj_consumer_credit = np.log(consumer_credit).diff()
adj_consumer_credit = fix_dates(adj_consumer_credit)
norm_adj_consumer_credit = zscore(adj_consumer_credit)
norm_adj_consumer_credit.columns = ['Consumer Credit']

#Prime Rate
prime_rate = pdr.get_data_fred('MPRIME', start='1949-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
prime_rate = prime_rate.resample('M').last()
adj_prime_rate = np.log(prime_rate).diff()
norm_adj_prime_rate = zscore(adj_prime_rate)
norm_adj_prime_rate.columns = ['Prime Rate']

#Labor Force Participation Rate
participation_rate = pdr.get_data_fred('CIVPART', start='1948-01-01', end=end_of_month(dateToday - relativedelta(months=1)))
participation_rate = participation_rate.resample('M').last()
adj_participation_rate = participation_rate.diff()
norm_adj_participation_rate = zscore(adj_participation_rate)
norm_adj_participation_rate.columns = ['Participation Rate']

#Home Prices
home = pdr.get_data_fred('CSUSHPINSA', start='1987-03-01', end=end_of_month(dateToday - relativedelta(months=1)))
home = home.resample('M').last()
adj_home = np.log(home).diff()
adj_home = fix_dates(adj_home)
norm_adj_home = zscore(adj_home)
norm_adj_home.columns = ['Home Prices']

#Policy Uncertainty (Survey)
policy = pdr.get_data_fred('USEPUINDXD', start='1985-03-01', end=end_of_month(dateToday - relativedelta(months=1)))
policy = policy.resample('M').last()
adj_policy = policy.diff()
adj_policy = fix_dates(adj_policy)
norm_adj_policy = zscore(adj_policy)
norm_adj_policy.columns = ['Policy Uncertainty']

#VIX
vix = pdr.get_data_fred('EMVOVERALLEMV', start='1985-03-01', end=end_of_month(dateToday - relativedelta(months=1)))
vix = vix.resample('M').last()
adj_vix = vix.diff()
adj_vix = fix_dates(adj_vix)
norm_adj_vix = zscore(adj_vix)
norm_adj_vix.columns = ['VIX']

#Sticky Prices
sticky = pdr.get_data_fred('STICKCPIM157SFRBATL', start='1985-03-01', end=end_of_month(dateToday - relativedelta(months=1)))
sticky = sticky.resample('M').last()
adj_sticky = sticky.diff()
adj_sticky = fix_dates(adj_sticky)
norm_adj_sticky = zscore(adj_sticky)
norm_adj_sticky.columns = ['Sticky CPI']

#Consumer Sentiment
sentiment = pdr.get_data_fred('UMCSENT', start='1979-03-01', end=end_of_month(dateToday - relativedelta(months=1)))
sentiment = sentiment.resample('M').last()
adj_sentiment = sentiment.diff()
adj_sentiment = fix_dates(adj_sentiment)
norm_adj_sentiment = zscore(adj_sentiment)
norm_adj_sentiment.columns = ['Consumer Sentiment']


df = pd.DataFrame()
vars=[norm_adj_sentiment, norm_adj_sticky, norm_adj_vix, norm_adj_policy, norm_adj_home, norm_adj_participation_rate, norm_adj_prime_rate, norm_adj_consumer_credit, norm_adj_m2, norm_adj_slope,
    norm_adj_slope_chg, norm_adj_consumer_orders, norm_adj_capital_orders, norm_adj_new_orders, norm_adj_wholesale_inv_to_sales,
    norm_adj_man_inv_to_sales, norm_adj_inv_to_sales, norm_adj_real_sales, norm_adj_retailfood_sales,
    norm_adj_house, norm_adj_permits, norm_adj_pce_services, norm_adj_pce_nondurable, norm_adj_pce_durable,
    norm_adj_pce, norm_adj_nonag_employment, norm_adj_avg_ot_hours, norm_adj_avg_hours, norm_adj_long_unemploy,
    norm_adj_dur_unemployment, norm_adj_con_claims, norm_adj_init_claims, norm_adj_hs_unemployment,
    norm_adj_core_unemployment, norm_adj_unemployment, norm_adj_goods_payrolls, norm_adj_service_payrolls,
    norm_adj_all_payrolls, norm_adj_real_income, norm_adj_capacity, norm_adj_nondur_man, norm_adj_dur_man,
    norm_adj_man_ind_prod, norm_adj_all_ind_prod, norm_adj_fixed_mortgage, norm_adj_crude, norm_adj_copper,
    norm_adj_beef, norm_adj_wheat, norm_adj_bbb_spread, norm_adj_aaa_spread, norm_adj_pp, norm_adj_man_conf,
    norm_adj_cpi, norm_adj_earnings, norm_adj_exports, norm_adj_imports, norm_adj_bus_loans, norm_adj_part_time,
    norm_adj_trans_receipt, nomr_adj_vehicle_loans, norm_adj_bank_credit, norm_adj_personal_savings]

'''Variables to add:
    Baker Highes US Rig Count
    NFIB Small Business Index
    '''

for n in vars:
    df=df.join(n,how='outer') 
df = df.dropna()
pca_bc = PCA(n_components=3)
pc = pca_bc.fit_transform(df)
pc = pd.DataFrame(pc)
pc.index = df.index
pc.columns = ['PC1', 'PC2', 'PC3']
for col in pc.columns:
    pc[col] = norm(pc[col])
pc = pc.ewm(halflife=1.5,min_periods=3).mean()
x = pca_bc.explained_variance_ratio_
print(f'Explained variation per principal component: {x}. Total explanatory power is {sum(x)}')

#plt.plot(pc, label=pc.columns)
#plt.legend()
#plt.show()
#exit()

pc['BC'] = -pc['PC1']/3 + pc['PC2']/6 + -pc['PC3']/2
pc['BC'] = sp.norm.cdf(pc['BC'], 0)
pc['BC'] = (pc['BC']-0.5) * 100 + 100

bc_3m = (pc['BC'].rolling(window=3).mean() - 
         pc['BC'].rolling(window=12).mean()) / pc['BC'].rolling(window=12).mean()

bc_6m = pc['BC'].pct_change()
bc_6m = bc_6m.rolling(window=6).apply(multi_period_return, raw=False)

bc_momo = 0.75 *  bc_3m + 0.25 * bc_6m
bc_momo = pd.DataFrame(bc_momo)

bus_cycle_signal = []

for i,j in zip(pc['BC'].values, bc_momo.values):
    if i > 100 and j > 0:
        bus_cycle_signal.append(1)
    elif i > 100 and j < 0:
        bus_cycle_signal.append(2)
    elif i < 100 and j < 0:
        bus_cycle_signal.append(3)
    elif i < 100 and j > 0:
        bus_cycle_signal.append(4)
    else:
        bus_cycle_signal.append(np.NaN)

bc_signal = pd.Series(bus_cycle_signal, index=pc['BC'].index[len(pc['BC']) - len(bus_cycle_signal):])
bc_signal = bc_signal.dropna()
bc_signal = bc_signal.rename('Business_Cycle_Indicator')

ee = []
le = []
ec = []
lc = []
for phase in bc_signal:
    if phase == 1:
        ee.append(1)
        le.append(np.NaN)
        ec.append(np.NaN)
        lc.append(np.NaN)
    elif phase == 2:
        ee.append(np.NaN)
        le.append(1)
        ec.append(np.NaN)
        lc.append(np.NaN)
    elif phase == 3:
        ee.append(np.NaN)
        le.append(np.NaN)
        ec.append(1)
        lc.append(np.NaN)
    elif phase == 4:
        ee.append(np.NaN)
        le.append(np.NaN)
        ec.append(np.NaN)
        lc.append(1)
    else:
        pass

phases = pd.DataFrame({'Early_Expansion':ee, 'Late_Expansion':le,
                       'Early_Contraction':ec, 'Late_Contraction':lc},
                      index=bc_signal.index
    )

# =============================================================================
#                            Sector Returns
# =============================================================================

def get_sector_etf_rets(tickers, sector_names, start):
    data = []
    for etf, name in zip(tickers, sector_names):
        price = yf.download(etf, start=start, end=str(dateToday))['Adj Close']
        price = price.resample('M').last().shift(periods=-1).dropna()
        rets = price.pct_change().dropna()
        rets = pd.DataFrame({name:rets})
        data.append(rets)
    sector_returns = data[0].join(data[1:])
    return sector_returns

tickers = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLU']
sector_names = ['CD', 'CS', 'EN', 'FN', 'HC', 'ID', 'IT', 'MA', 'UT']

sector_rets = get_sector_etf_rets(tickers, sector_names, '1998-12-16')
phase_rets = []

for phase in phases.columns:
    rets = get_sector_etf_rets(tickers, sector_names, '1998-12-16')
    for sector in sector_rets.columns:
        rets[sector] = sector_rets[sector] * phases[phase]
    rets = rets.dropna()
    ann_ret = np.prod(1 + rets) ** (1/(len(rets)/12)) - 1
    ann_stdev = np.std(rets) * np.sqrt(12)
    phase_ir = ann_ret / ann_stdev
    phase_ir = pd.DataFrame({phase:phase_ir})
    phase_rets.append(phase_ir)
phase_rets = phase_rets[0].join(phase_rets[1:])


def ir_charts(phases_bc, top_bottom):
    fig = plt.figure(figsize=[11,9])
    for i, name in enumerate(phases_bc):
        i += 1
        ax = fig.add_subplot(2,2,i)
        ax.title.set_text(name)
        ax.tick_params(direction='in', length=8)
        colors = ['r' if i <= phase_rets[name].nsmallest(top_bottom).iloc[-1] 
          else 'g' if i >= phase_rets[name].nlargest(top_bottom).iloc[-1] 
          else 'grey' for i in phase_rets[name].values]
        phase_rets[name].plot(kind='bar', color=colors)
        ax.axhline(y=0, color='black', linewidth=1)
    return fig

charts = ir_charts(phase_rets.columns, 3)
plt.savefig(f'{location}/IR_Charts.png')        
spy = yf.download('SPY', start='1998-12-16', end=str(dateToday))['Adj Close']
spy = spy.resample('M').last().shift(periods=-1).dropna()
spy = spy.pct_change().dropna()
spy_tr = spy / (np.std(spy) * np.sqrt(12) * 100)
plt.clf()

# =============================================================================
#                            Sector Strategy
# =============================================================================
num_pos = 3
longs = [phase_rets[name].nlargest(num_pos).index.values for name in phase_rets.columns]
shorts = [phase_rets[name].nsmallest(num_pos).index.values for name in phase_rets.columns]

long_rets = [sector_rets[name] for name in longs]
long_rets = [(returns.sum(axis=1) * 1/num_pos) for returns in long_rets]
long_rets = pd.concat(long_rets, axis=1)
long_rets = long_rets.mul(phases.shift(1)[len(phases)-len(long_rets):].values)
long_rets = long_rets.sum(axis = 1)
long_rets_tr = long_rets / (np.std(long_rets) * np.sqrt(12) * 100)

short_rets = [sector_rets[name] for name in shorts]
short_rets = [(returns.sum(axis=1) * 1/num_pos) for returns in short_rets]
short_rets = pd.concat(short_rets, axis=1)
short_rets = short_rets.mul(phases.shift(1)[len(phases)-len(short_rets):].values)
short_rets = short_rets.sum(axis = 1)
short_rets_tr = short_rets / (np.std(short_rets) * np.sqrt(12) * 100)

long_short = long_rets - short_rets.values
long_short_tr = long_short / (np.sqrt(12) * np.std(long_short) * 100)

cum_long_ret = np.cumprod(1 + long_rets) - 1
cum_short_ret = np.cumprod(1 + short_rets) - 1
cum_long_short = np.cumprod(1 + long_short) - 1
cum_spy = np.cumprod(1 + spy) - 1

cum_long_ret_tr = np.cumprod(1 + long_rets_tr) - 1
cum_short_ret_tr = np.cumprod(1 + short_rets_tr) - 1
cum_long_short_tr = np.cumprod(1 + long_short_tr) - 1
cum_spy_tr = np.cumprod(1 + spy_tr) - 1

plt.clf()
plt.plot(cum_long_ret_tr, color='red', label='long')
plt.plot(cum_long_ret_tr-cum_spy_tr, color='blue', label='excess return')
plt.plot(cum_spy_tr, color='black', label='SPY')
plt.legend()
plt.show()


# =============================================================================
#                     Output 
# =============================================================================
pdf = canvas.Canvas(f'/Users/User/Desktop/Practice Python/Business Cycle Monthly Report {(dateToday - relativedelta(months=1)).strftime("%m")}-{(dateToday- relativedelta(months=1)).year}.pdf')
pdf.setTitle('Business Cycle Monthly Report')

pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 770, 700, 150, fill=1)

pdf.setFillColor(white)
pdf.setFont('Helvetica', 48)
pdf.drawCentredString(300, 790, 'US Business Cycle Report')

pdf.setFillColor(grey, alpha=0.7)
pdf.setLineWidth(0)
pdf.rect(-10, 735, 620, 35, fill=1)

pdf.setFillColor(black)
pdf.setFont('Helvetica', 24)
lastmonth = (dateToday - relativedelta(months=1)).strftime('%B %Y')
pdf.drawString(50, 740, 'For the Month of ' + lastmonth)

def get_index_rets(tickers, names):
    index_rets = [pd.DataFrame({name:yf.download(ticker, start=str(dateToday - relativedelta(years=1)),
                                      end=str(end_date))['Adj Close']}) for name, ticker in zip(names, tickers)]
    index_rets = index_rets[0].join(index_rets[1:])
    index_rets = index_rets.pct_change().dropna()
    index_rets.iloc[0,:] = 0
    index_rets = np.cumprod(1 + index_rets) - 1
    return index_rets

index_rets = get_index_rets(['^IXIC', '^GSPC', '^RUT'],
                            ['NASDAQ Composite', 'S&P 500', 'Russell 2000'])
index_rets = index_rets * 100
plt.rcParams["font.family"] = "sans-serif"
fig, ax = plt.subplots(figsize=[8.5,4])
ax = index_rets['S&P 500'].plot(color='blue', label='S&P 500')
index_rets['NASDAQ Composite'].plot(color='dimgray', label='NASDAQ Composite', linestyle='dotted')
index_rets['Russell 2000'].plot(color='black', label='Russell 2000', linestyle='--')
ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
ax.axhline(y=0, color='black', linewidth=1)
ax.tick_params(direction='in', length=8)
ax.legend(edgecolor='black', loc='lower left')
x_axis = ax.axes.get_xaxis()
x_label = x_axis.get_label()
x_label.set_visible(False)
plt.savefig(f'{location}/Pic.png')
pdf.drawImage(f'{location}/Pic.png', 0, 450, 600, 285)
pdf.setFont('Helvetica', 14)
pdf.drawString(75, 710, 'US Equity Indices Trailing 12 Month Cumulative Returns')
pdf.setFont('Helvetica', 9)
pdf.drawString(75, 460, 'Source: Yahoo Finance, as of ' + end_date.strftime('%Y-%m-%d'))

pdf.setFont('Helvetica', 24)
pdf.drawString(50, 425, 'Monthly Summary')
pdf.line(47, 417, 550, 417)

pdf.setFont('Helvetica', 18)
pdf.drawString(100,390, 'Business Cycle Phase')
pdf.drawString(350,390, 'Target Asset Allocation')


current_phase = bus_cycle_signal[-1]
if current_phase == 1:
    pdf.setLineWidth(1)
    pdf.setFillColorRGB(0.1, 0.75, 0.35)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.setFillColorRGB(0.1, 0.75, 0.35, alpha=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(92.5, 332, 'Early Expansion')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(340, 335, f'Target Sectors: {", ".join(longs[0])}')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(341, 250, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(338, 165, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(343, 80, f'Target Sectors: {", ".join(longs[3])}')
elif current_phase == 2:
    pdf.setLineWidth(1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.setFillColorRGB(0.6, 0.9, 0.4, alpha=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.setFillColorRGB(0.6, 0.9, 0.4, alpha=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.setFont('Helvetica-Bold', 22)
    pdf.setFillColor(white)
    pdf.drawString(95, 247, 'Late Expansion')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 335, f'Target Sectors: {", ".join(longs[0])}')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(337, 250, f'Target Sectors: {", ".join(longs[1])}')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(338, 165, f'Target Sectors: {", ".join(longs[2])}')
    pdf.drawString(343, 80, f'Target Sectors: {", ".join(longs[3])}')
elif current_phase == 3:
    pdf.setLineWidth(1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.setFillColor(red, alpha=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.setFillColor(red, alpha=0.85)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(85, 162, 'Early Contraction')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(98, 77, 'Late Contraction')
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 335, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(341, 250, f'Target Sectors: {", ".join(longs[1])}')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(334, 165, f'Target Sectors: {", ".join(longs[2])}')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(343, 80, f'Target Sectors: {", ".join(longs[3])}')
elif current_phase == 4:
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(65, 300, 225, 75, fill=1)
    pdf.rect(65, 215, 225, 75, fill=1)
    pdf.rect(65, 130, 225, 75, fill=1)
    pdf.setFillColorRGB(1,0.85,0, alpha=1.0)
    pdf.rect(65, 45, 225, 75, fill=1)
    pdf.setFillColor(grey, alpha=0.5)
    pdf.rect(325, 300, 225, 75, fill=1)
    pdf.rect(325, 215, 225, 75, fill=1)
    pdf.rect(325, 130, 225, 75, fill=1)
    pdf.setFillColorRGB(1,0.85,0, alpha=1.0)
    pdf.rect(325, 45, 225, 75, fill=1)
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 22)
    pdf.drawString(95, 332, 'Early Expansion')
    pdf.drawString(102, 247, 'Late Expansion')
    pdf.drawString(95, 162, 'Early Contraction')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 22)
    pdf.drawString(88, 77, 'Late Contraction')
    pdf.setFillColor(black)
    pdf.setFont('Helvetica', 16)
    pdf.drawString(342, 335, f'Target Sectors: {", ".join(longs[0])}')
    pdf.drawString(341, 250, f'Target Sectors: {", ".join(longs[1])}')
    pdf.drawString(338, 175, f'Target Sectors: {", ".join(longs[2])}')
    pdf.setFillColor(white)
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(336, 80, f'Target Sectors: {", ".join(longs[3])}')
else:
    pass


# =============================================================================
#                      Page 1 - BC
# =============================================================================
pdf.showPage()
comp_bc = pd.DataFrame(pc['BC'])
comp_bc.index = pc.index
bc = pd.DataFrame({
    'BC': comp_bc.iloc[-36:,0],
    'Momo': bc_momo.iloc[-36:,0]
    }, index=comp_bc.index[-36:])

fig, ax = plt.subplots(figsize=[9.5,4])
ax.plot(comp_bc, color='blue')
ax.tick_params(direction='in', length=8)
ax.axvspan(date2num(datetime(2001,3,1)), date2num(datetime(2001,11,30)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(2007,12,1)), date2num(datetime(2009,6,30)), color='grey', alpha=0.5)
ax.axvspan(date2num(datetime(2020,3,1)), date2num(datetime(2020,12,30)), color='grey', alpha=0.5)
ax.axhline(y=100, color='black', linewidth=1)
ax.set_ylim(25, 175)
plt.savefig(f'{location}/Bus_Cycle.png')
pdf.drawImage(f'{location}/Bus_Cycle.png', -40, 480, 700, 270)
            
fig, ax = plt.subplots(figsize=[7.5,5])
ax.plot(bc.Momo * 100, bc.BC, color='black')
ax.set_xlim(15, -15)
ax.set_ylim(60,140)
plt.axhline(y=100, color='black', linewidth=1)
plt.axvline(x=0, color='black', linewidth=1)
ax.get_xaxis().set_major_formatter(ticker.PercentFormatter())
ax.set_xlabel('Index Momentum')
ax.set_ylabel('Index Level')
ax.axvspan(0, 15, 0.5, 1, color='green', alpha=0.7)
ax.axvspan(-15, 0, 0.5, 1, color='green', alpha=0.4)
ax.axvspan(-15, 0, 0, 0.5, color='red', alpha=0.6)
ax.axvspan(0, 15, 0, 0.5, color='gold', alpha=0.6)
plt.annotate('{:%b-%y}'.format(pd.to_datetime(bc.index[0])), xy=(bc.Momo[1] * 100 + 2, bc.BC[1]-5))
plt.annotate('{:%b-%y}'.format(pd.to_datetime(bc.index[-1])), xy=(bc.Momo[-1]*100 + 3, bc.BC[-1]-5))
plt.savefig(f'{location}/BC_Map.png')
pdf.drawImage(f'{location}/BC_Map.png', 40, 80, 550, 350)

pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'Composite Business Cycle Indicator')
pdf.setFont('Helvetica', size=18)
pdf.drawString(50,740, 'Composite Business Cycle Indicator')
pdf.drawString(50,425, 'Business Cycle Compass')
pdf.setLineWidth(1)
pdf.line(45, 417, 550, 417)
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 479, 'Note: Shaded areas represent US recessions as indicated by NBER.')
pdf.drawString(47, 466, 'Reading greater than 100 indicates expansion, less than 100 indicates contraction.')
pdf.setFont('Helvetica', size=32)
pdf.setFillColor(black)
pdf.drawString(120, 360, '1')
pdf.drawString(500, 360, '2')
pdf.drawString(500, 125, '3')
pdf.drawString(120, 125, '4')
pdf.setFont('Helvetica', 10)
pdf.drawString(47, 75, 'Note: Momentum calculated as 25% 6M Change, 75% 3M Average less 12M Average.')
pdf.drawString(47, 62, '1 = Early Expansion, 2 = Late Expansion, 3 = Early Contraction, 4 = Late Contraction')
pdf.line(45, 57, 550, 57)


# =============================================================================
#                          Sector Strategy
# =============================================================================
pdf.showPage()

pdf.setFillColor(black)
pdf.drawImage(f'{location}/IR_Charts.png', -25, 290, width=635, height=495)

pdf.setLineWidth(0)
pdf.setFillColorRGB(0.1,0.05,0.55)
pdf.rect(-10, 780, 700, 5, fill=1)
pdf.setFillColor(black)
pdf.setFont('Helvetica', 36)
pdf.drawCentredString(300, 800, 'US Sector Rotation Strategy')
pdf.setFont('Helvetica', size=18)
pdf.drawString(40,750,'US Sectors Information Ratios by Phase of Business Cycle')
pdf.setLineWidth(1)
pdf.line(35, 745, 560, 745)

tr_1 = True

if tr_1 == True:
    fig, ax = plt.subplots(figsize=[8.5,4])
    ax = (cum_long_ret_tr * 100).plot(color='green', label='Long')
    (cum_short_ret_tr * 100).plot(color='red', label='Short')
    (cum_long_short_tr * 100).plot(color='blue', label='Long-Short')
    (cum_spy_tr * 100).plot(color='black', label='SPY')
    ax.legend(edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.tick_params(which='major', direction='in', length=8)
    ax.tick_params(which='minor', direction='in', length=4)
    ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    plt.savefig(f'{location}/SectorIRs.png')
    pdf.drawImage(f'{location}/SectorIRs.png', -10 , 3, 590, 290)
    pdf.setFontSize(10)
    pdf.drawString(45, 309, 'Note: Green bars denote sectors with long position, red short, and grey neutral.')
    pdf.drawString(45, 296, 'Sector returns based on SPDR Sector ETF returns from 1999-01-31 to ' + end_of_month(dateToday - relativedelta(months=1)).strftime('%Y-%m-%d'))
    pdf.line(35, 291, 560, 291)
    pdf.setFontSize(18)
    pdf.drawString(45, 264, 'Sector Strategy Cumulative Returns, Risk Level 1%')
elif tr_1 == False:
    fig, ax = plt.subplots(figsize=[8.5,4])
    ax = (cum_long_ret * 100).plot(color='green', label='Long', linestyle='--')
    (cum_short_ret * 100).plot(color='red', label='Short', linestyle='--')
    (cum_long_short * 100).plot(color='blue', label='Long-Short')
    (cum_spy * 100).plot(color='black', label='SPY')
    ax.legend(edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.tick_params(which='major', direction='in', length=8)
    ax.tick_params(which='minor', direction='in', length=4)
    ax.get_yaxis().set_major_formatter(ticker.PercentFormatter())
    x_axis = ax.axes.get_xaxis()
    x_label = x_axis.get_label()
    x_label.set_visible(False)
    plt.savefig(f'{location}/SectorIRs.png')
    pdf.drawImage(f'{location}/SectorIRs.png', -10 , 3, 590, 290)
    pdf.setFontSize(10)
    pdf.drawString(45, 309, 'Note: Green bars denote sectors with long position, red short, and grey neutral.')
    pdf.drawString(45, 296, 'Sector returns based on SPDR Sector ETF returns from 1999-01-31 to ' + end_date.strftime('%Y-%m-%d'))
    pdf.line(35, 291, 560, 291)
    pdf.setFontSize(18)
    pdf.drawString(45, 264, 'Sector Strategy Cumulative Returns')


pdf.save()
print('Done')
