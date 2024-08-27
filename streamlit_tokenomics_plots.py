import math
import numpy as np
import matplotlib.pyplot as plt 
import streamlit as st
import pandas as pd
from decimal import Decimal, getcontext, ROUND_HALF_UP

getcontext().prec = 150

def dec(number):
    #getcontext().prec = 200
    #return(number)
    return(Decimal(str(number)))

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
    return FRC * TotalSupply * (dec(1) + KFS) * (dec(1) - dec(math.exp(-u * dec(SecondsInYear)))) / ParticipantsNum

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

def minted_tokens_number_calc(t, TotalSupply, KFS, u, FRC, ParticipantsNum):
    return FRC * TotalSupply * (dec(1) + KFS) * (dec(1) - dec(math.exp(-u * dec(t)))) / ParticipantsNum

TotalSupply = dec(10400000000)
KFS = dec(10 ** (-5))
TTMT = dec(2000000000)
u = -dec(1) / TTMT * dec(math.log(KFS / (dec(1) + KFS)))
SecondsInYear = 31557600
SecondsInMonth = 2629800



st.title("Acki Nacki Tokenomics")

node_type_option = st.selectbox(
   r"Select the type of node:",
   (r"Block Keeper", 
    r"Block Manager"),
   index=0,
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
    ParticipantsNum = dec(input_number_of_block_keepers())
    FRC = dec(0.675) # Function Reward Coefficient
    expected_bk_apy = expected_apy_calc(TotalSupply, KFS, u, SecondsInYear, FRC, ParticipantsNum)
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
    ParticipantsNum = dec(input_number_of_block_managers())
    FRC = dec(0.1) # Function Reward Coefficient
    expected_bm_apy = expected_apy_calc(TotalSupply, KFS, u, SecondsInYear, FRC, ParticipantsNum)
    #raised_amount = node_license_price * number_of_licenses
    implied_1_y_token_price = node_license_price / expected_bm_apy

number_of_licenses = dec(input_number_of_licenses())
st.write("Implied 1Y Token Price = " + str(round(implied_1_y_token_price, 7)))
plot_scale = input_plot_scale()

# 1 plot

fig, ax = plt.subplots()
if plot_scale <= 15:
    values_x = np.arange(0, plot_scale * SecondsInYear * 1.05,  plot_scale * SecondsInYear * 1.05 / 1000)
else:
    values_x = np.arange(0, plot_scale * (SecondsInYear + 4) // 5 * 5 * 1.05,  plot_scale * SecondsInYear * 1.05 / 1000)
values_tokens = np.array([minted_tokens_number_calc(t, TotalSupply, KFS, u, FRC, ParticipantsNum) for t in values_x])
min_y_value = min(list(values_tokens))
max_y_value = max(list(values_tokens))
ax.plot(values_x, values_tokens, color = "Red")
ax.set_ylabel(r'Minted Token Amount (in thousands)')
ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
y_ticks = ax.get_yticks()
new_labels = [f'{int(y_tick / 1e3)}' for y_tick in y_ticks]
new_labels[0] = '0'
ax.set_yticklabels(new_labels)
min_x_value = min(list(values_x))
ax.set_xlim([min_x_value, plot_scale * SecondsInYear * 1.05])

if plot_scale <= 2:
    xlabels = list([i for i in range(0, plot_scale * 12 + 1, 1)])
    xticks = list([i * SecondsInMonth for i in xlabels])
    ax.set_xlabel(r'Time (in months)')
if 3 <= plot_scale <= 5:
    xlabels = list([i for i in range(0, plot_scale * 12 + 1, 5)])
    xticks = list([i * SecondsInMonth for i in xlabels])
    ax.set_xlabel(r'Time (in months)')
if 6 <= plot_scale <= 15:
    xlabels = list([i for i in range(0, plot_scale + 1, 1)])
    xticks = list([i * SecondsInYear for i in xlabels])
    ax.set_xlabel(r'Time (in years)')
if plot_scale >= 16:
    xlabels = list([i for i in range(0, (plot_scale + 4) // 5 * 5 + 1, 5)])
    xticks = list([i * SecondsInYear for i in xlabels])
    ax.set_xlabel(r'Time (in years)')

ax.set_xticks(xticks, xlabels)
ax.grid(True)
st.pyplot(fig)