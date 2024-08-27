import math
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import pandas as pd
from decimal import Decimal, getcontext, ROUND_HALF_UP

getcontext().prec = 150

seconds_in_year = 365.25 * 24 * 60 * 60

PctValsMaxReputation = 0
# val_rotation_speed_coef = 2592000
minRC = 1
maxRC = 3
maxRT = 157766400 #157766400 (5 лет)
ARFC = 1000


def dec(number):
    #getcontext().prec = 200
    #return(number)
    return(Decimal(str(number)))


# Вероятно некорректный подсчет

#def avg_rep_coef(minRC, maxRC, maxRT, ARFC, TTMT, PctValsMaxReputation):
#    u = dec(dec(math.log(ARFC))) / dec(maxRT)
#    avg_rep_coef = (dec(minRC) * (dec(PctValsMaxReputation) * ((dec(1) - dec(math.exp(-u * dec(maxRT)))) / (u * dec(maxRT)) - dec(1)) + dec(1)) + \
#                    dec(maxRC) * dec(PctValsMaxReputation) * ((dec(math.exp(-u * dec(maxRT))) - dec(1)) / (u * dec(maxRT)) + dec(1))) * dec(maxRT) / dec(TTMT) + \
#                    (dec(minRC) - dec(minRC) * dec(PctValsMaxReputation) + dec(maxRC) * dec(PctValsMaxReputation)) * (dec(TTMT) - dec(maxRT)) / dec(TTMT)
#    return avg_rep_coef

#ЭТОТ ПАРАМЕТР БЫЛ ОТРЕДАКТИРОВАН (УМНОЖЕН НА 10) В ТОТ МОМЕНТ, КОГДА МЫ УВЕЛИЧИЛИ КОЛИЧЕСТВО ТОКЕНОВ ДО 104 МЛРД
ABRFC = 100000 # Регулировать этот параметр (БЫЛ 100.000 ПРИ 10.4 МЛРД, РАВЕН 1.000.000 ПРИ 100 МЛРД TOKEN CAP)
TTMT = 2000000000
ABRPC = 1 # dec(PctValsMaxReputation) * dec(maxRC) + (dec(1) - dec(PctValsMaxReputation)) * dec(minRC) #avg_rep_coef(minRC, maxRC, maxRT, ARFC, TTMT, PctValsMaxReputation)
TotalCap = 10400000000 #10400000000

KF = 10 ** (-5) # 10 ** (-5)
#KF = ABRFC / TotalCap

#st.write(KF)

maxFF = 1 / 3
FFF = 10 ** (-2)

FF_coef = 1 / 3


def years_to_second(years):
    return dec(dec(365.25) * dec(24) * dec(60) * dec(60) * dec(years))


#def total_basic_reward(x, ABRFC, ABRPC, TotalCap, TTMT):
#    u = -1 * dec(1) / dec(TTMT) * dec(math.log(dec(ABRFC) / dec(TotalCap + ABRFC)))
#    return dec(TotalCap + ABRFC) / dec(ABRPC) * (dec(1) - dec(math.exp(-u * dec(x))))

def total_basic_reward(x, KF, TotalCap, TTMT):
    u = -1 * dec(1) / dec(TTMT) * dec(math.log(dec(KF) / (dec(1) + dec(KF))))
    return dec(dec(TotalCap) * (dec(1) + dec(KF)) * (dec(1) - dec(math.exp(-u * dec(x)))))

def free_float(x, FFF, maxFF, TTMT):
    u = -1 * dec(1) / dec(TTMT) * dec(math.log(dec(FFF) / (dec(1) + dec(FFF))))
    return dec(dec(maxFF) * (dec(1) + dec(FFF)) * (dec(1) - dec(math.exp(-u * dec(x)))))


def reputation_coef(x, minRC, maxRC, maxRT, ARFC):
    u = dec(dec(math.log(ARFC))) / dec(maxRT)
    if dec(0) <= dec(x) <= dec(maxRT):
        return dec(minRC) + dec(maxRC - minRC) / (dec(1) - dec(1)/dec(ARFC)) * (dec(1) - dec(math.exp(-u * dec(x))))
    else:
        return dec(maxRC)


# Общее количество токенов, вышедших на рынок, с учетом растущей репутации и процентного
# соотношение валидаторов с максимальной репутацией и с нулевой репутацией
def total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT):
    cur_basic_reward = total_basic_reward(x, KF, TotalCap, TTMT)
    return(cur_basic_reward)
    #cur_max_rep = reputation_coef(x, minRC, maxRC, maxRT, ARFC)
    #return(cur_basic_reward * dec(PctValsMaxReputation) * cur_max_rep + cur_basic_reward * (dec(1) - dec(PctValsMaxReputation)) * dec(minRC))

def input_minValNum_min_stake ():
    return st.number_input(
        label = r'Insert the starting number of validators', 
        help = r"Starting number of validators is in $ \lbrack 1, 1000000 \rbrack $", 
        value = 100, 
        format = "%i",
        min_value = 0,
        max_value = 1000000,
        key = "minValNum_min_stake"
        )

def input_maxValNum_min_stake ():
    return st.number_input(
        label = r'Insert the end number of validators', 
        help = r"End number of validators is in $ \lbrack \mathsf{minValNum}, 1000000 \rbrack $", 
        value = 10000, 
        format = "%i",
        min_value = minValNum,
        max_value = 1000000, 
        key = "maxValNum_min_stake"
        )

def input_cur_year ():
    return st.number_input(
        label = r'Insert the number of years since the network launch', 
        help = r"Current year since the network launch is in $ \lbrack 0.1, 63 \rbrack $", 
        value = 1.0, 
        format = "%f",
        min_value = 0.1,
        max_value = 63.000, 
        key = "cur_year"
        )


def input_cur_month ():
    return st.number_input(
        label = r'Insert the current year', 
        help = r"Current year since the network launch is in $ \lbrack 0.0001, 63 \rbrack $", 
        value = 0.64, 
        format = "%f",
        min_value = 0.0001,
        max_value = 63.000, 
        key = "cur_year"
        )

def input_minValNum_min_stake_from_vals ():
    return st.number_input(
        label = r'Insert the starting number of validators', 
        help = r"Starting number of validators is in $ \lbrack 1, 1000000 \rbrack $", 
        value = 100, 
        format = "%i",
        min_value = 0,
        max_value = 1000000,
        key = "minValNum_min_stake_from_vals"
        )

def input_maxValNum_min_stake_from_vals ():
    return st.number_input(
        label = r'Insert the end number of validators', 
        help = r"End number of validators is in $ \lbrack \mathsf{minValNum}, 1000000 \rbrack $", 
        value = 20000, 
        format = "%i",
        min_value = minValNum,
        max_value = 1000000, 
        key = "maxValNum_min_stake_from_vals"
        )

def cur_needed_vals_calc(x, minValNum, maxValNum, TTMT):
    return (dec(maxValNum) - dec(minValNum)) / dec(TTMT) * dec(x) + dec(minValNum)

def min_stake_calc(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, FF_coef):
    cur_supply = total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT)
    cur_needed_validators = 10000#cur_needed_vals_calc(x, minValNum, maxValNum, TTMT)
    return (cur_supply * (dec(1) - dec(free_float(x, FFF, maxFF, TTMT))) / dec(2)) / cur_needed_validators


def apy_calc(x, FF_coef, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, seconds_in_year):
    #x *= seconds_in_year
    #st.write(x)
    delta_reward = total_supply_acki_nacki(x * seconds_in_year, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) - \
                    total_supply_acki_nacki((x - 1) * seconds_in_year, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT)
    #st.write(total_supply_acki_nacki(x * seconds_in_year, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT))
    #st.write(total_supply_acki_nacki((x - 1) * seconds_in_year, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT))
    #st.write(delta_reward)
    total_period_stake = 0
    help_value = 0
    for cur_time in range(int((x - 1) * seconds_in_year), int(x * seconds_in_year + 1), 100000):
        total_period_stake += total_supply_acki_nacki(cur_time, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) * \
                                (dec(1) - dec(free_float(cur_time, FFF, maxFF, TTMT))) / dec(2) #((dec(1) - dec(FF_coef)) / dec(2))
        help_value += 1
    #st.write(help_value)
    #st.write(len(range(int((x - 1) * seconds_in_year), int(x * seconds_in_year + 1), 10000)))
    #st.write(((dec(1) - dec(FF_coef)) / dec(2)))
    #st.write(len(range(int((int(x) - 1) * seconds_in_year), int(int(x) * seconds_in_year + 1), int(10000))))
    avg_stake = total_period_stake / len(range(int((x - 1) * seconds_in_year), int(x * seconds_in_year + 1), 100000))
    #st.write(delta_reward)
    #st.write(avg_stake)
    apy = delta_reward / avg_stake * 100
    #st.write(apy)
    #st.write(apy)
    return(apy)

#def input_TTMT ():
#    return st.number_input(
#        label = r'Insert the number of minting years', 
#        help = r"Number of minting years is in $ \lbrack 1, 100 \rbrack $", 
#        value = 63.0, 
#        format = "%f",
#        min_value = 1.0,
#        max_value = 100.0
#        )


def round_decimal(number, nums):
    # Находим порядок числа (показатель степени 10)
    exponent = int(math.floor(math.log10(abs(number))))
    # Округляем число до двух знаков после запятой
    rounded_number = round(number * 10**(-exponent), nums)
    result = str(rounded_number) + " * 10^(" + str(exponent) + ")"
    return result


def min_stake_from_vals_num(x, minValNum, maxValNum, cur_NV, cur_base_min_stake, cur_FF):
    b = cur_base_min_stake
    #st.write(b)
    m = maxValNum
    #st.write(m)
    g = minValNum
    #st.write(g)
    N = cur_NV
    #st.write(N)
    f = cur_FF * dec(0.05)
    #st.write(f)
    a = dec(math.log(dec(10000))) / (N - g)
    #st.write(a)
    #if 2 * N - g < m:
    #    M_numerator = dec(2) * (dec(math.exp(a)) - dec(1)) * (b * (g - N - 1) + f)
    #    M_denominator = b * (dec(math.exp(a * (N - g + 1))) + dec(math.exp(a)) * (g - N - 1) + N - g)
    #    M = M_numerator / M_denominator
    #else:
    #    M_numerator = dec(2) * (dec(math.exp(a)) - dec(1)) * (b * (N - m - 1) + f)
    #    M_denominator = b * (dec(math.exp(a * (m - N + 1))) + dec(math.exp(a)) * (N - m - 1) + m - N)
    #    M = M_numerator / M_denominator
    #st.write(f"a: {a}")
    #st.write(f"b (cur_base_min_stake): {b}")
    #st.write(f"m (maxValNum): {m}")
    #st.write(f"g (minValNum): {g}")
    #st.write(f"N (cur_NV): {N}")
    #st.write(f"f (cur_FF * 0.05): {f}")
    #st.write(f"M numerator: {M_numerator}")
    #st.write(f"M denominator: {M_denominator}")
    #st.write(f"M: {M}")
    #st.write(M)
    if g <= dec(x) <= N:
        minStake = b * (dec(1) - dec(math.exp(-a * (dec(x) - g))))
        #st.write(minStake)
        return(minStake)
        #if minStake <= dec(0):
        #     return dec(0)
        #else:
        #    return minStake
    if dec(x) > N:
        minStake = b * (dec(1) + dec(math.exp(-dec(a) * (2 * N - g - dec(x)))))
        #if minStake <= dec(0):
        #     return dec(0)
        #else:
        #    return minStake
        #st.write(minStake)
        return minStake










def input_number_of_block_keepers():
    return st.number_input(
        label = r'Insert a number of Block Keepers', 
        help = r"Number of Block Keepers is in $ \lbrack 100, 10000 \rbrack $", 
        value = 10000, 
        format = "%i",
        min_value = 100,
        max_value = 10000
        )

def input_number_of_block_managers():
    return st.number_input(
        label = r'Insert a number of Block Managers', 
        help = r"Number of Block Managers is in $ \lbrack 100, 10000 \rbrack $", 
        value = 10000, 
        format = "%i",
        min_value = 100,
        max_value = 10000
        )

def expected_apy_calc(TotalSupply, KFS, u, SecondsInYear, FRC, ParticipantsNum):
    return FRC * TotalSupply * (dec(1) + KFS) * (dec(1) - dec(math.exp(-u * SecondsInYear))) / ParticipantsNum

def input_number_of_licenses():
    return st.number_input(
        label = r'Insert a number of licenses', 
        help = r"Number of licenses is in $ \lbrack 1, 200 \rbrack $", 
        value = 1, 
        format = "%i",
        min_value = 1,
        max_value = 200
        )

def input_plot_scale():
    return st.number_input(
        label = r'Insert a plot scale (in years)', 
        help = r"Plot scale (in years) is in $ \lbrack 1, 60\rbrack $", 
        value = 1, 
        format = "%i",
        min_value = 1,
        max_value = 60
        )

TotalSupply = dec(10400000000)
KFS = dec(10 ** (-5))
TTMT = dec(2000000000)
u = -dec(1) / TTMT * dec(math.log(KFS / (dec(1) + KFS)))
SecondsInYear = dec(31557600)




st.title("Acki Nacki Tokenomics")

node_type_option = st.selectbox(
   r"Select the type of node:",
   (r"Block Keeper", 
    r"Block Manager"),
   index=None,
)

if node_type_option == r"Block Keeper":
    node_price_option = st.selectbox(
    r"Select the node license price:",
    (r"1500 $", 
    r"1750 $", 
    r"2041 $", 
    r"2381 $", 
    r"2778 $", 
    r"3240 $", 
    r"3780 $", 
    r"4409 $", 
    r"5143 $", 
    r"6000 $"),
   index=0,
    )
    if node_price_option == r"1500 $":
        node_license_price = dec(1500)
        #number_of_licenses = dec(2500)
    if node_price_option == r"1750 $":
        node_license_price = dec(1750)
        #number_of_licenses = dec(2000)
    if node_price_option == r"2041 $":
        node_license_price = dec(2041)
        #number_of_licenses = dec(1300)
    if node_price_option == r"2381 $":
        node_license_price = dec(2381)
        #number_of_licenses = dec(1100)
    if node_price_option == r"2778 $":
        node_license_price = dec(2778)
        #number_of_licenses = dec(800)
    if node_price_option == r"3240 $":
        node_license_price = dec(3240)
        #number_of_licenses = dec(650)
    if node_price_option == r"3780 $":
        node_license_price = dec(3780)
        #number_of_licenses = dec(530)
    if node_price_option == r"4409 $":
        node_license_price = dec(4409)
        #number_of_licenses = dec(454)
    if node_price_option == r"5143 $":
        node_license_price = dec(5143)
        #number_of_licenses = dec(390)
    if node_price_option == r"6000 $":
        node_license_price = dec(6000)
        #number_of_licenses = dec(276)
    BKNum = dec(input_number_of_block_keepers())
    FRC = dec(0.675) # Function Reward Coefficient
    expected_bk_apy = expected_apy_calc(TotalSupply, KFS, u, SecondsInYear, FRC, BKNum)
    #raised_amount = node_license_price * number_of_licenses
    implied_1_y_token_price = node_license_price / expected_bk_apy

if node_type_option == r"Block Manager":
    node_price_option = st.selectbox(
    r"Select the node license price:",
    (r"400 $", 
    r"478 $", 
    r"572 $", 
    r"684 $", 
    r"818 $", 
    r"978 $", 
    r"1170 $", 
    r"1399 $", 
    r"1673 $", 
    r"2000 $"),
   index=0,
    )
    if node_price_option == r"400 $":
        node_license_price = dec(400)
        #number_of_licenses = dec(2500)
    if node_price_option == r"478 $":
        node_license_price = dec(478)
        #number_of_licenses = dec(2000)
    if node_price_option == r"572 $":
        node_license_price = dec(572)
        #number_of_licenses = dec(1300)
    if node_price_option == r"684 $":
        node_license_price = dec(684)
        #number_of_licenses = dec(1100)
    if node_price_option == r"818 $":
        node_license_price = dec(818)
        #number_of_licenses = dec(800)
    if node_price_option == r"978 $":
        node_license_price = dec(978)
        #number_of_licenses = dec(650)
    if node_price_option == r"1170 $":
        node_license_price = dec(1170)
        #number_of_licenses = dec(530)
    if node_price_option == r"1399 $":
        node_license_price = dec(1399)
        #number_of_licenses = dec(454)
    if node_price_option == r"1673 $":
        node_license_price = dec(1673)
        #number_of_licenses = dec(390)
    if node_price_option == r"2000 $":
        node_license_price = dec(2000)
        #number_of_licenses = dec(276)
    BMNum = dec(input_number_of_block_managers())
    FRC = dec(0.1) # Function Reward Coefficient
    expected_bm_apy = expected_apy_calc(TotalSupply, KFS, u, SecondsInYear, FRC, BMNum)
    #raised_amount = node_license_price * number_of_licenses
    implied_1_y_token_price = node_license_price / expected_bm_apy

number_of_licenses = input_number_of_licenses()
st.write("Implied 1Y Token Price = " + str(round(implied_1_y_token_price, 7)))
plot_scale = input_plot_scale()


# 1 plot


fig, ax = plt.subplots()
values_x = np.arange(0, int(TTMT), int(TTMT) / 1000)
values_supply = np.array([total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) for x in values_x])

values_free_float = np.array([total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) * free_float(x, FFF, maxFF, TTMT) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_supply))
max_y_value = max(list(values_supply))

ax.plot(values_x, values_supply, label = r"Supply", color = "Red")
ax.plot(values_x, values_free_float, label = r"Free Float", color = "Blue")
ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
#ax.hlines(y=TotalCap / 3, xmin=min_x_value, xmax=max_x_value, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Token Amount (in billions)')
yticks = list([0, 2 * 10 ** 9, 4 * 10 ** 9, 6 * 10 ** 9, 8 * 10 ** 9, 10.4 * 10 ** 9])
ylabels = list([0, 2, 4, 6, 8, 10.4])
ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 70, 5)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
ax.set_xlim([min_x_value, max_x_value * 1.05])

ax.legend()
ax.grid(True)
st.pyplot(fig)


#st.write([int(total_supply_acki_nacki(x * seconds_in_year, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT)) for x in [1, 2, 3, 4, 5]])

# 2 plot

fig, ax = plt.subplots()
values_x = np.arange(0, int(6 * seconds_in_year * 1.05), int(6 * seconds_in_year * 1.05) / 1000)
values_rep_coef = np.array([reputation_coef(x, minRC, maxRC, maxRT, ARFC) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_rep_coef))
max_y_value = max(list(values_rep_coef))

ax.plot(values_x, values_rep_coef, label = r"Reputation Coefficient", color = "Red")
ax.hlines(y=maxRC, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
ax.vlines(x=maxRT, ymin=0, ymax=maxRC, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Reputation Coefficient')
#yticks = list([0, 2 * 10 ** 9, 4 * 10 ** 9, 6 * 10 ** 9, 8 * 10 ** 9, 10.4 * 10 ** 9])
#ylabels = list([0, 2, 4, 6, 8, 10.4])
#ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 7, 1)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
ax.set_xlim([min_x_value, max_x_value])

ax.legend()
ax.grid(True)
st.pyplot(fig)


# 3 plot

BitcoinTotalCap = 21000000

# Загрузка данных из CSV файла с кодировкой UTF-16
file_path = 'Bitcoin_Total_Supply_filtered.csv'  # Укажите путь к вашему CSV файлу
data = pd.read_csv(file_path, encoding='utf-16', delimiter=',')
# Переименование столбцов
data.columns = ['Time', 'Supply']
# Преобразование столбца 'Time' в datetime
data['Time'] = pd.to_datetime(data['Time'])
# Вычисление разницы в секундах относительно первой даты
start_time = data['Time'].iloc[0]
data['Seconds'] = (data['Time'] - start_time).dt.total_seconds()
# Переименование столбца
data.rename(columns={data.columns[1]: 'Supply / Total Cap'}, inplace=True)
# Деление значений во втором столбце на Total Cap Bitcoin
data['Supply / Total Cap'] = data['Supply / Total Cap'] / BitcoinTotalCap


fig, ax = plt.subplots()
values_x = np.arange(0, 17 * seconds_in_year, 17 * seconds_in_year / 1000)
values_acki_nacki_supply_per_total_cap = np.array([total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) / TotalCap for x in values_x])


min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_acki_nacki_supply_per_total_cap))
max_y_value = max(list(values_acki_nacki_supply_per_total_cap))

ax.plot(values_x, values_acki_nacki_supply_per_total_cap, label = r"Acki Nacki", color = "Red", linewidth=0.8)
ax.plot(data['Seconds'], data['Supply / Total Cap'], label = r"Bitcoin", color = "Blue", linewidth=0.8)
ax.hlines(y=1, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)


ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Supply / Total Supply')
yticks = list([i / 100 for i in range(0, 110, 10)])
ylabels = list([str(int(i * 100)) + "%" for i in yticks])
ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 18, 1)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * dec(1.1)])
ax.set_xlim([min_x_value, max_x_value])

ax.legend()
ax.grid(True)
st.pyplot(fig)





# 4 plot


#minValNum = input_minValNum_min_stake()
#maxValNum = input_maxValNum_min_stake()
nv = 10000
fig, ax = plt.subplots()
values_x = np.arange(0, int(TTMT), int(TTMT) / 1000)
values_min_stake = np.array([min_stake_calc(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, FF_coef) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_min_stake))
max_y_value = max(list(values_min_stake))

ax.plot(values_x, values_min_stake, label = r"Basic Min Val Stake ($\mathsf{NV} = $" + str(10000) + r")", color = "Red")
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Basic Min Val Stake (in thousands)')
yticks = list([i for i in range(0, (math.ceil(max_y_value * dec(1.2) / dec(1000)) + 1) * 1000, 25000)])
ylabels = list([int(i / 1000) for i in yticks])
ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 70, 5)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
ax.set_xlim([min_x_value, max_x_value * 1.05])

ax.legend()
ax.grid(True)
st.pyplot(fig)


# 5 plot


minValNum = input_minValNum_min_stake_from_vals()
maxValNum = input_maxValNum_min_stake_from_vals()

cur_year = input_cur_year()
cur_sec = seconds_in_year * cur_year
nv = 10000

#cur_NV = cur_needed_vals_calc(cur_sec, minValNum, maxValNum, TTMT)
cur_NV = 10000
cur_base_min_stake = min_stake_calc(cur_sec, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, FF_coef)
cur_FF = total_supply_acki_nacki(cur_sec, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) * dec(FF_coef)
#st.write(total_supply_acki_nacki(cur_sec, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT))
#st.write(total_supply_acki_nacki(cur_sec, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) * dec(FF_coef))
fig, ax = plt.subplots()
values_x = np.concatenate([np.linspace(minValNum, cur_NV, 100), np.linspace(int(cur_NV + 1), int(2 * cur_NV - minValNum), 100)])
#st.write(values_x)
values_min_stake = np.array([float(min_stake_from_vals_num(x, dec(minValNum), dec(maxValNum), cur_NV, cur_base_min_stake, cur_FF)) for x in values_x])
#st.write(values_min_stake)

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_min_stake))
max_y_value = max(list(values_min_stake))
label = r"Min Val Stake ($\mathsf{minValNum} = $" + str(minValNum) + ",\n$\mathsf{maxValNum} = $" + str(maxValNum) + r", " + str(int(cur_year * 12)) + " months, \n" + "$\mathsf{basicMinValStake} = $" + str(round(cur_base_min_stake, 0)) + ",\n" + "$\mathsf{NV} = $" + str(round(cur_NV, 0)) + ")"
ax.plot(values_x, values_min_stake, label=label, color="Red")
ax.vlines(x=cur_NV, ymin=min_x_value, ymax=cur_base_min_stake, color='black', linestyle='--', linewidth=0.8)
ax.hlines(y=cur_base_min_stake, xmin=0, xmax=3 * cur_NV - minValNum, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Number of Validators')
ax.set_ylabel(r'Min Stake (in thousands)')
#yticks = list([0, 2 * 10 ** 9, 4 * 10 ** 9, 6 * 10 ** 9, 8 * 10 ** 9, 10.4 * 10 ** 9])
#ylabels = list([0, 2, 4, 6, 8, 10.4])
#ax.set_yticks(yticks, ylabels)
m = maxValNum
g = minValNum
N = cur_NV
if 2 * N - g < m:
#    xlabels = [g] + [i for i in range(1000, int(2 * N - g), 1000)]
#    xticks = list([i for i in xlabels])
#    ax.set_xticks(xticks, xlabels)
    ax.set_xlim([min_x_value, (2 * N - g) * dec(1.05)])
else:
#    xlabels = [g] + [i for i in range(1000, int(m), 1000)]
#    xticks = list([i for i in xlabels])
#    ax.set_xticks(xticks, xlabels)
    ax.set_xlim([min_x_value, (m) * dec(1.05)])
#if max_y_value > min_y_value * 1000 + 1:
#   ax.set_yscale('log')
#ax.set_yscale('log')
yticks = list([i for i in range(0, (math.ceil(max_y_value * 1.2 / 1000) + 1) * 1000, 25000)])
ylabels = list([int(i / 1000) for i in yticks])
ax.set_yticks(yticks, ylabels)
ax.set_ylim([min_y_value, dec(max_y_value) * dec(1.2)])


ax.legend()
ax.grid(True)
st.pyplot(fig)




# 6 plot

# Загрузка данных из CSV файла с кодировкой UTF-16
file_path_2 = 'Bitcoin_FF_div_Total_Supply_Per_filtered.csv'  # Укажите путь к вашему CSV файлу
data_2 = pd.read_csv(file_path_2, encoding='utf-16', delimiter=',')
# Преобразование столбца 'Time' в datetime
data_2['Time'] = pd.to_datetime(data_2['Time'])
# Вычисление разницы в секундах относительно первой даты
start_time = data_2['Time'].iloc[0]
data_2['Seconds'] = (data_2['Time'] - start_time).dt.total_seconds()

fig, ax = plt.subplots()
values_x = np.arange(0, 17 * seconds_in_year, 17 * seconds_in_year / 1000)
values_free_float_pct = np.array([free_float(_, FFF, maxFF, TTMT) * dec(100) for _ in values_x])


# Вычисление медианы Free Float Bitcoin по всей дате
mean_free_float_bitcoin_all_time = data_2['FF_pct'].mean()
# Исключение первых полтора года данных
one_and_half_year_seconds = 1.5 * 365.25 * 24 * 60 * 60
start_time = data_2['Time'].iloc[0]
filtered_data = data_2[data_2['Time'] > (start_time + pd.to_timedelta(one_and_half_year_seconds, unit='s'))]
# Вычисление медианы Free Float Bitcoin исключая первые полтора года
mean_free_float_bitcoin_filtered = filtered_data['FF_pct'].mean()
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
#ax.hlines(y=TotalCap / 3, xmin=min_x_value, xmax=max_x_value, color='black', linestyle='--', linewidth=0.8)
values_mean_free_float_bitcoin_all_time = np.array([mean_free_float_bitcoin_all_time for _ in values_x])
values_mean_free_float_bitcoin_filtered = np.array([mean_free_float_bitcoin_filtered for _ in values_x])


min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(data_2['FF_pct']))
max_y_value = max(list(data_2['FF_pct']))

ax.plot(values_x, values_free_float_pct, label = r"Acki Nacki", color = "Red")
ax.plot(data_2['Seconds'], data_2['FF_pct'], label = r"Bitcoin", color = "Grey", alpha = 0.5)
ax.plot(values_x, values_mean_free_float_bitcoin_all_time, label = r"Mean, Bitcoin, All time", color = "cyan")
ax.plot(values_x, values_mean_free_float_bitcoin_filtered, label = r"Mean, Bitcoin, Excluding first one and a half years", color = "Blue")





ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Free Float / Supply')
yticks = list([i for i in range(0, 110, 10)])
ylabels = list([str(int(i)) + "%" for i in yticks])
ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 18, 1)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * 1.2])
ax.set_xlim([min_x_value, max_x_value])

ax.legend()
ax.grid(True)
st.pyplot(fig)


# 6 * plot

def convert_to_float(value):
    # Проверка, если значение содержит запятую
    if ',' in value:
        integer_part, fractional_part = value.split(',')
        return float(f"{integer_part}.{fractional_part}")
    else:
        return float(value)

bitcoin_total_supply = 21000000

# Загрузка данных из CSV файла с кодировкой UTF-16
file_path_3 = 'Bitcoin_FF_filtered.csv'  # Укажите путь к вашему CSV файлу
data_3 = pd.read_csv(file_path_3, encoding='utf-16', delimiter=',')
# Преобразование столбца 'Time' в datetime
data_3['Time'] = pd.to_datetime(data_3['Time'])
# Вычисление разницы в секундах относительно первой даты
start_time = data_3['Time'].iloc[0]
data_3['Seconds'] = (data_3['Time'] - start_time).dt.total_seconds()

fig, ax = plt.subplots()
values_x = np.arange(0, 17 * seconds_in_year, 17 * seconds_in_year / 1000)
values_acki_nacki_supply_perc_total = ([total_supply_acki_nacki(x, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT) * free_float(x, FFF, maxFF, TTMT) / dec(TotalCap) * dec(100) for x in values_x])

data_3['FF'] = data_3['FF'].apply(convert_to_float) / bitcoin_total_supply * 100
# Вычисление медианы Free Float Bitcoin по всей дате
mean_free_float_bitcoin_all_time = data_3['FF'].mean()
# Исключение первых полтора года данных
one_and_half_year_seconds = 1.5 * 365.25 * 24 * 60 * 60
start_time = data_3['Time'].iloc[0]
filtered_data = data_3[data_3['Time'] > (start_time + pd.to_timedelta(one_and_half_year_seconds, unit='s'))]
# Вычисление медианы Free Float Bitcoin исключая первые полтора года
mean_free_float_bitcoin_filtered = filtered_data['FF'].mean()
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
#ax.hlines(y=TotalCap / 3, xmin=min_x_value, xmax=max_x_value, color='black', linestyle='--', linewidth=0.8)
values_mean_free_float_bitcoin_all_time = np.array([mean_free_float_bitcoin_all_time for _ in values_x])
values_mean_free_float_bitcoin_filtered = np.array([mean_free_float_bitcoin_filtered for _ in values_x])


min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(data_3['FF']))
max_y_value = max(list(data_3['FF']))
#st.write(min_y_value)
#st.write(max_y_value)

ax.plot(data_3['Seconds'], data_3['FF'], label = r"Bitcoin", color = "Grey", alpha = 0.5)
ax.plot(values_x, values_mean_free_float_bitcoin_all_time, label = r"Mean, Bitcoin, All time", color = "cyan")
ax.plot(values_x, values_mean_free_float_bitcoin_filtered, label = r"Mean, Bitcoin, Excluding first one and a half years", color = "Blue")
ax.plot(values_x, values_acki_nacki_supply_perc_total, label = r"Acki Nacki", color = "Red")

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'Free Float / Total Supply')
yticks = list([i for i in range(0, 110, 10)])
ylabels = list([str(int(i)) + "%" for i in yticks])
ax.set_yticks(yticks, ylabels)
xlabels = list([i for i in range(0, 18, 1)])
xticks = list([i * seconds_in_year for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_ylim([min_y_value, max_y_value * 1.2])
ax.set_xlim([min_x_value, max_x_value])

ax.legend()
ax.grid(True)
st.pyplot(fig)

#st.write(values_acki_nacki_supply_perc_total)
#st.write(values_mean_free_float_bitcoin_filtered)
#st.write(values_mean_free_float_bitcoin_all_time)

# 7 plot

fig, ax = plt.subplots()
values_x = np.arange(1, math.floor(TTMT / seconds_in_year), 1)
values_apy = np.array([apy_calc(x, FF_coef, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, seconds_in_year) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_apy))
max_y_value = max(list(values_apy))

ax.plot(values_x, values_apy, label = r"APY", color = "Red")
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'APY (in percent)')
#yticks = list([0, 2 * 10 ** 9, 4 * 10 ** 9, 6 * 10 ** 9, 8 * 10 ** 9, 10.4 * 10 ** 9])
#ylabels = list([0, 2, 4, 6, 8, 10.4])
#ax.set_yticks(yticks, ylabels)
xlabels = list([1] + [i for i in range(5, 70, 5)])
xticks = list([i for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_xlim([min_x_value - 1, max_x_value * 1.05])  
#ax.set_yscale('log')
ax.set_ylim([min_y_value, max_y_value * dec(1.2)])

ax.legend()
ax.grid(True)
st.pyplot(fig)

# 8 plot

# 7 plot

fig, ax = plt.subplots()
values_x = np.arange(1, 6, 1)
values_apy = np.array([apy_calc(x, FF_coef, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, seconds_in_year) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_apy))
max_y_value = max(list(values_apy))

ax.plot(values_x, values_apy, label = r"APY", color = "Red")
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'APY (in percent)')

xlabels = list(i for i in range(1, 6))
xticks = list([i for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_xlim([0, 6])  

#yticks = list([int(i) for i in values_apy])
#ylabels = list([i for i in yticks])
#ax.set_yticks(yticks, ylabels)

yticks = list(ax.get_yticks())
ylabels = list(ax.get_yticklabels())
#yticks.append(50)
#ylabels.append("50")
ax.set_yticks(yticks, ylabels)
ax.set_ylim([0, max_y_value * dec(1.2)])

ax.legend()
ax.grid(True)

# Отключение одной горизонтальной риски
#gridlines = ax.get_ygridlines()

# Отключение линии сетки для определенного значения метки
#for line in gridlines:
#    if line.get_ydata()[0] == 50:  # Укажите значение метки, для которой нужно убрать линию сетки
#        line.set_visible(False)

st.pyplot(fig)

# 7 plot

fig, ax = plt.subplots()
values_x = np.arange(1, math.floor(TTMT / seconds_in_year), 1)
values_apy = np.array([apy_calc(x, FF_coef, PctValsMaxReputation, minRC, maxRC, maxRT, ARFC, ABRFC, ABRPC, TotalCap, TTMT, seconds_in_year) for x in values_x])

min_x_value = min(list(values_x))
max_x_value = max(list(values_x))
min_y_value = min(list(values_apy))
max_y_value = max(list(values_apy))

ax.plot(values_x, values_apy, label = r"APY", color = "Red")
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)

ax.set_xlabel(r'Time (in years)')
ax.set_ylabel(r'APY (in percent)')
#yticks = list([0, 2 * 10 ** 9, 4 * 10 ** 9, 6 * 10 ** 9, 8 * 10 ** 9, 10.4 * 10 ** 9])
#ylabels = list([0, 2, 4, 6, 8, 10.4])
#ax.set_yticks(yticks, ylabels)
xlabels = list([1] + [i for i in range(5, 70, 5)])
xticks = list([i for i in xlabels])
ax.set_xticks(xticks, xlabels)
ax.set_xlim([min_x_value - 1, max_x_value * 1.05])  
ax.set_yscale('log')
yticks = list(ax.get_yticks())
ylabels = [item.get_text() for item in ax.get_yticklabels()]
yticks = yticks[:-3]
ylabels = ylabels[:-3]
yticks.extend([1, 10, 100])
ylabels.extend(["1", "10", "100"])
ax.set_yticks(yticks, ylabels)
ax.set_ylim([min_y_value / dec(10), max_y_value * dec(1.2)])

ax.legend()
ax.grid(True)
st.pyplot(fig)



# 9 help plot



#fig, ax = plt.subplots()
#values_x = np.arange(0, int(TTMT / 5), int(TTMT) / 10000)
#need_val = np.array([cur_needed_vals_calc(x, minValNum, maxValNum, TTMT) for x in values_x])

#min_x_value = min(list(values_x))
#max_x_value = max(list(values_x))
#min_y_value = min(list(need_val))
#max_y_value = max(list(need_val))

#ax.plot(values_x, need_val, label = r"NV", color = "Red")
#st.write(cur_needed_vals_calc(6 * seconds_in_year, minValNum, maxValNum, TTMT))
#ax.hlines(y=TotalCap, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
#ax.hlines(y=TotalCap / 3, xmin=min_x_value, xmax=max_x_value, color='black', linestyle='--', linewidth=0.8)

#ax.set_xlabel(r'Time (in years)')
#ax.set_ylabel(r'Needed Vals Num')
#xlabels = list([i for i in range(0, 70, 5)])
#xticks = list([i * seconds_in_year for i in xlabels])
#ax.set_xticks(xticks, xlabels)
#ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
#ax.set_xlim([min_x_value, max_x_value * 1.05])

#ax.legend()
#ax.grid(True)
#st.pyplot(fig)
# 10 plot 


# Загрузим данные из CSV файла
#elec_price = pd.read_csv('Electricity_price.csv', encoding='utf-8')
# Преобразуем столбец DATE в формат datetime
#elec_price['DATE'] = pd.to_datetime(elec_price['DATE'], errors='coerce')
# Очистим столбец с ценами, удалив нечисловые символы и преобразовав его в числовой формат
#elec_price['APU000072610'] = pd.to_numeric(elec_price['APU000072610'].str.replace(',', '').str.replace(' ', ''), errors='coerce')
# Извлекаем год из столбца DATE
#elec_price['YEAR'] = elec_price['DATE'].dt.year
# Группируем данные по годам и рассчитываем среднюю цену
#average_prices = elec_price.groupby('YEAR')['APU000072610'].mean().reset_index()
# Фильтруем данные с 2010 по 2023 год
#filtered_prices = average_prices[(average_prices['YEAR'] >= 2010) & (average_prices['YEAR'] <= 2023)]
# Преобразуем результат в список средних значений
#elec_price_list = filtered_prices['APU000072610'].tolist()


#elec_price_list = [0.04 for i in range(14)]

#st.write("elec_price_list")
#st.write(elec_price_list)




#energy_consumption = [188, 15, 30, 360, 340, 590, 1293, 1323, 1350, 2094, 3250, 3050, 3010, 5304]
#energy_consumption = [i / 1000 for i in energy_consumption]

#st.write("energy_consumption in kW * h")
#st.write(energy_consumption)

#mining_rig_hash_power = [
#    0.4e9,      # 2010: 0.4 GH/s
#    25e6,       # 2011: 25 MH/s
#    5e9,        # 2012: 5 GH/s
#    180e9,      # 2013: 180 GH/s
#    441e9,      # 2014: 441 GH/s
#    1.155e12,   # 2015: 1.155 TH/s
#    4.73e12,    # 2016: 4.73 TH/s
#    13.5e12,    # 2017: 13.5 TH/s
#    14e12,      # 2018: 14 TH/s
#    53e12,      # 2019: 53 TH/s
#    110e12,     # 2020: 110 TH/s
#    100e12,     # 2021: 100 TH/s
#    140e12,     # 2022: 140 TH/s
#    255e12      # 2023: 255 TH/s
#]

#st.write("mining_rig_hash_power")
#st.write(mining_rig_hash_power)

# Чтение CSV файла с указанием кодировки и разделителя, если необходимо
#bitcoin_hash_rate = pd.read_csv('Bitcoin_Hashrate_filtered.csv', encoding='utf-16', delimiter=',')
# Преобразование столбца Time в формат datetime
#bitcoin_hash_rate['Time'] = pd.to_datetime(bitcoin_hash_rate['Time'], format='%Y-%m-%d')
# Замена запятых на точки в столбце Hashrate
#bitcoin_hash_rate['Hashrate'] = bitcoin_hash_rate['Hashrate'].str.replace(',', '.')
# Преобразование столбца Hashrate в числовой формат
#bitcoin_hash_rate['Hashrate'] = pd.to_numeric(bitcoin_hash_rate['Hashrate'], errors='coerce')
# Добавление столбца с годом
#bitcoin_hash_rate['Year'] = bitcoin_hash_rate['Time'].dt.year
# Преобразование хешрейта из TH/s в Hash/s
#bitcoin_hash_rate['Hashrate'] = bitcoin_hash_rate['Hashrate'] * 1e12
# Создание списка для хранения среднегодового хешрейта
#average_bitcoin_hashrate_per_year_list = []
# Вычисление среднего хешрейта за каждый год с 2010 по 2023
#for year in range(2010, 2024):
    # Фильтрация данных за конкретный год
#    yearly_data = bitcoin_hash_rate[bitcoin_hash_rate['Year'] == year]
    # Вычисление среднего хешрейта за год и добавление в список
#    average_bitcoin_hashrate = yearly_data['Hashrate'].mean()
#    average_bitcoin_hashrate_per_year_list.append(average_bitcoin_hashrate)

#st.write("average_bitcoin_hashrate_per_year_list")
#st.write(average_bitcoin_hashrate_per_year_list)


#average_consumption_usd_per_hour = []
# Вычисление среднего потребления в USD в час для каждого элемента
#for i in range(len(elec_price_list)):
#    consumption_usd_per_hour = (elec_price_list[i] * energy_consumption[i] * average_bitcoin_hashrate_per_year_list[i]) / mining_rig_hash_power[i]
#    average_consumption_usd_per_hour.append(consumption_usd_per_hour)
#average_consumption_usd_per_year = [x * 24 * 365.25 for x in average_consumption_usd_per_hour]

#st.write("average_consumption_usd_per_year")
#st.write(average_consumption_usd_per_year)

# Даты халвингов
#halving_dates = [
#    {'date': '2012-11-28', 'reward': 25},  # Первый халвинг
#    {'date': '2016-07-09', 'reward': 12.5},  # Второй халвинг
#    {'date': '2020-05-11', 'reward': 6.25}  # Третий халвинг
#]
# Начальная награда
#initial_reward = 50
# Функция для расчета средней награды за год
#def calculate_yearly_average_reward(year):
#    if year < 2012:
#        return initial_reward
#    elif year == 2012:
#        days_before_halving = (pd.Timestamp('2012-11-28') - pd.Timestamp(f'{year}-01-01')).days
#        days_after_halving = 365 - days_before_halving
#        return (initial_reward * days_before_halving + 25 * days_after_halving) / 365
#    elif 2013 <= year < 2016:
#        return 25
#    elif year == 2016:
#        days_before_halving = (pd.Timestamp('2016-07-09') - pd.Timestamp(f'{year}-01-01')).days
#        days_after_halving = 365 - days_before_halving
#        return (25 * days_before_halving + 12.5 * days_after_halving) / 365
#    elif 2017 <= year < 2020:
#        return 12.5
#    elif year == 2020:
#        days_before_halving = (pd.Timestamp('2020-05-11') - pd.Timestamp(f'{year}-01-01')).days
#        days_after_halving = 366 - days_before_halving  # 2020 is a leap year
#        return (12.5 * days_before_halving + 6.25 * days_after_halving) / 366
#    else:
#        return 6.25
# Список для хранения средней награды за блок для каждого года с 2009 по 2023
#average_block_rewards = []
# Расчет средней награды за каждый год
#for year in range(2010, 2024):
#    average_block_rewards.append(calculate_yearly_average_reward(year))

#st.write("average block rewards")
#st.write(average_block_rewards)

# Чтение CSV файла
#bitcoin_price = pd.read_csv('Bitcoin_price_filtered.csv', encoding='utf-16')
# Преобразование столбца Time в формат datetime
#bitcoin_price['Time'] = pd.to_datetime(bitcoin_price['Time'], format='%Y-%m-%d')
# Замена запятых на точки в столбце Price
#bitcoin_price['Price'] = bitcoin_price['Price'].str.replace(',', '.').astype(float)
# Фильтрация данных до 2023 года включительно
#bitcoin_price = bitcoin_price[bitcoin_price['Time'] <= '2023-12-31']
# Добавление столбца с годом
#bitcoin_price['Year'] = bitcoin_price['Time'].dt.year
# Создание списка для хранения средней годовой цены
#average_price_per_year = []
# Расчет средней цены за 2010 год, начиная с 8 июля
#year_2010_data = bitcoin_price[(bitcoin_price['Year'] == 2010) & (bitcoin_price['Time'] >= '2010-07-08')]
#average_price_2010 = year_2010_data['Price'].mean()
####################average_price_per_year.append(average_price_2010)
# Расчет средней цены за каждый год с 2011 по 2023
#for year in range(2010, 2024):
#    yearly_data = bitcoin_price[bitcoin_price['Year'] == year]
#    average_price = yearly_data['Price'].mean()
#    average_price_per_year.append(average_price)

# Чтение CSV файла
#bitcoin_price = pd.read_csv('Bitcoin_price_filtered.csv', encoding='utf-16')
# Преобразование столбца Time в формат datetime
#bitcoin_price['Time'] = pd.to_datetime(bitcoin_price['Time'], format='%Y-%m-%d')
# Замена запятых на точки в столбце Price
#bitcoin_price['Price'] = bitcoin_price['Price'].str.replace(',', '.').astype(float)
# Фильтрация данных до 2023 года включительно
#bitcoin_price = bitcoin_price[bitcoin_price['Time'] <= '2023-12-31']
# Добавление столбца с годом
#bitcoin_price['Year'] = bitcoin_price['Time'].dt.year
# Создание списка для хранения 75% от максимальной цены за год
#seventy_five_percent_high_per_year = []
# Расчет 75% от максимальной цены за 2010 год, начиная с 8 июля
#year_2010_data = bitcoin_price[(bitcoin_price['Year'] == 2010) & (bitcoin_price['Time'] >= '2010-07-08')]
#max_price_2010 = year_2010_data['Price'].max()
#seventy_five_percent_high_2010 = 0.75 * max_price_2010
##########seventy_five_percent_high_per_year.append(seventy_five_percent_high_2010)
# Расчет 75% от максимальной цены за каждый год с 2011 по 2023
#for year in range(2010, 2024):
#    yearly_data = bitcoin_price[bitcoin_price['Year'] == year]
#    max_price = yearly_data['Price'].max()
#    seventy_five_percent_high = 0.75 * max_price
#    seventy_five_percent_high_per_year.append(seventy_five_percent_high)

#st.write("average bitcoin price (75% from max price)")
#st.write(seventy_five_percent_high_per_year)


#average_income_usd_per_year = []
#for i in range(len(average_block_rewards)):
#    value = average_price_per_year[i] * average_block_rewards[i] * 6 * 24 * 365.25
    ##################value = seventy_five_percent_high_per_year[i] * average_block_rewards[i] * 6 * 24 * 365.25
#    average_income_usd_per_year.append(value)

#st.write("average_income_usd_per_year")
#st.write(average_income_usd_per_year)

#ten_minutes_per_year = 52596


# Расчет APY для каждого года
#apy_list = []
#for i in range(len(average_income_usd_per_year)):
#    income = average_income_usd_per_year[i]
#    consumption = average_consumption_usd_per_year[i]
#    apy_list.append((income / consumption - 1) * 100)

# Печать списка APY
#st.write("apy")
#st.write(apy_list)

#st.write("average_income_usd_per_year")
#st.write(average_income_usd_per_year)
#st.write("average_consumption_usd_per_year")
#st.write(average_consumption_usd_per_year)










# 8 plot





# Это вставить, для проверки на необходимость log шкалы
# if max_y_value > min_y_value * 100:


# add test plot


#fig, ax = plt.subplots()
#bitcoin_br = 50
#bitcoin_total_supply = 21000000 #21000000
#supply = 0
#block_prod_time = 600 #600 секунд = 10 минут
#bitcoin_supply_list = [0]
#bitcoin_tokens_left = bitcoin_total_supply
#bitcoin_halving_border = bitcoin_total_supply / 2
#bitcoin_supply = 0
#values_x = [0]

#for cur_time in range(block_prod_time, int(TTMT), block_prod_time):
#    if bitcoin_supply >= bitcoin_halving_border:
#        bitcoin_br /= 2
#        bitcoin_halving_border = (bitcoin_total_supply + bitcoin_halving_border) / 2
#    bitcoin_supply += bitcoin_br
#    bitcoin_tokens_left -= bitcoin_br
#    bitcoin_supply_list.append(bitcoin_supply)
#    values_x.append(cur_time)

#values_x = np.array(values_x)
#values_supply = np.array(bitcoin_supply_list)


#min_x_value = min(list(values_x))
#max_x_value = max(list(values_x))
#min_y_value = min(list(values_supply))
#max_y_value = max(list(values_supply))

#ax.plot(values_x, values_supply, label = r"Bitcoin Model Supply", color = "Red")
#ax.hlines(y=bitcoin_total_supply, xmin=min_x_value, xmax=max_x_value * 1.05, color='black', linestyle='--', linewidth=0.8)
###ax.hlines(y=TotalCap / 3, xmin=min_x_value, xmax=max_x_value, color='black', linestyle='--', linewidth=0.8)

#ax.set_xlabel(r'Time (in years)')
#ax.set_ylabel(r'Bitcoin Supply (in millions)')
#yticks = ylabels = list([0 * 10 ** 6, 2 * 10 ** 6, 4 * 10 ** 6, 6 * 10 ** 6, 8 * 10 ** 6, 10 * 10 ** 6, 12 * 10 ** 6, 14 * 10 ** 6, 16 * 10 ** 6, 18 * 10 ** 6, 20 * 10 ** 6, 21 * 10 ** 6])
#ylabels = list([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21])
#ax.set_yticks(yticks, ylabels)
#xlabels = list([i for i in range(0, 70, 5)])
#xticks = list([i * seconds_in_year for i in xlabels])
#ax.set_xticks(xticks, xlabels)
#ax.set_ylim([min_y_value, max_y_value * 1.2])
#ax.set_xlim([min_x_value, max_x_value * 1.05])

#ax.legend()
#ax.grid(True)
#st.pyplot(fig)