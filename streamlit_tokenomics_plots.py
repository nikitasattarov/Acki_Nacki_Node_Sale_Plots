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
        label = r'Insert a total number of Block Keepers', 
        help = r"Number of Block Keepers is in $ \lbrack 100, 10000 \rbrack $", 
        value = 10000, 
        format = "%i",
        min_value = 100,
        max_value = 10000
        )

def input_number_of_block_managers():
    return st.number_input(
        label = r'Insert a total number of Block Managers', 
        help = r"Number of Block Managers is in $ \lbrack 100, 10000 \rbrack $", 
        value = 10000, 
        format = "%i",
        min_value = 100,
        max_value = 10000
        )

def expected_apy_calc(TotalSupply, KFS, u_tokens, SecondsInYear, FRC, ParticipantsNum):
    return FRC * TotalSupply * (dec(1) + KFS) * (dec(1) - dec(math.exp(-u_tokens* dec(SecondsInYear)))) / ParticipantsNum

def input_number_of_licenses_per_tier_bk(ParticipantsNum):
    return st.number_input(
        label = r'Insert a number of licenses', 
        #help = r"Number of licenses is in $ \left\lbrack 1, \min\left(200, \ \text{Number of Block Keepers}\right) \right\rbrack $", 
        help = r"Number of licenses is in $ \left\lbrack 1, \text{Number of Block Keepers} \right\rbrack $", 
        value = 1, 
        format = "%i",
        min_value = 1,
        #max_value = min(200, int(ParticipantsNum))
        max_value = int(ParticipantsNum)
        )

def input_number_of_licenses_per_tier_bm(ParticipantsNum):
    return st.number_input(
        label = r'Insert a number of licenses', 
        #help = r"Number of licenses is in $ \left\lbrack 1, \min\left(200, \ \text{Number of Block Managers}\right) \right\rbrack $", 
        help = r"Number of licenses is in $ \left\lbrack 1, \text{Number of Block Managers} \right\rbrack $", 
        value = 1, 
        format = "%i",
        min_value = 1,
        #max_value = min(200, int(ParticipantsNum))
        max_value = int(ParticipantsNum)
        )

def input_server_running_monhtly_cost():
    return st.number_input(
        label = r'Insert server running monthly cost ($)',  
        value = 0, 
        format = "%i",
        )

def input_plot_scale():
    return st.number_input(
        label = r'Insert a plot scale (in years)', 
        help = r"Plot scale (in years) is in $ \lbrack 1, 60\rbrack $", 
        value = 25, 
        format = "%i",
        min_value = 1,
        max_value = 60
        )

def minted_tokens_number_calc(t, TotalSupply, KFS, u_tokens, FRC, ParticipantsNum, number_of_purchased_licenses):
    return FRC * TotalSupply * (dec(1) + KFS) * (dec(1) - dec(math.exp(-u_tokens* dec(t)))) / ParticipantsNum * number_of_purchased_licenses

def free_float(t, FFF, maxFF, u_ff):
    return dec(dec(maxFF) * (dec(1) + dec(FFF)) * (dec(1) - dec(math.exp(-u_ff * dec(t)))))

TotalSupply = dec(10400000000)
KFS = dec(10 ** (-5))
TTMT = dec(2000000000)
u_tokens= -dec(1) / TTMT * dec(math.log(KFS / (dec(1) + KFS)))
SecondsInYear = 31557600
SecondsInMonth = 2629800
maxFF = dec(1 / 3)
FFF = dec(10 ** (-2))
u_ff = -dec(1) / dec(TTMT) * dec(math.log(FFF / (dec(1) + FFF)))


st.title("Acki Nacki Node Sale Tokenomics Plots")

node_type_option = st.selectbox(
   r"Select the type of node:",
   (r"Block Keeper", 
    r"Block Manager"),
   index=0,
)

if node_type_option == r"Block Keeper":
    node_price_option = st.selectbox(
    r"Select the node license price ($):",
    (r"1500", 
    r"1750", 
    r"2041", 
    r"2381", 
    r"2778", 
    r"3240", 
    r"3780", 
    r"4409", 
    r"5143", 
    r"6000"),
   index=0,
    )
    if node_price_option == r"1500":
        node_license_price = dec(1500)
        #number_of_licenses_per_tier = dec(2500)
    if node_price_option == r"1750":
        node_license_price = dec(1750)
        #number_of_licenses_per_tier = dec(2000)
    if node_price_option == r"2041":
        node_license_price = dec(2041)
        #number_of_licenses_per_tier = dec(1300)
    if node_price_option == r"2381":
        node_license_price = dec(2381)
        #number_of_licenses_per_tier = dec(1100)
    if node_price_option == r"2778":
        node_license_price = dec(2778)
        #number_of_licenses_per_tier = dec(800)
    if node_price_option == r"3240":
        node_license_price = dec(3240)
        #number_of_licenses_per_tier = dec(650)
    if node_price_option == r"3780":
        node_license_price = dec(3780)
        #number_of_licenses_per_tier = dec(530)
    if node_price_option == r"4409":
        node_license_price = dec(4409)
        #number_of_licenses_per_tier = dec(454)
    if node_price_option == r"5143":
        node_license_price = dec(5143)
        #number_of_licenses_per_tier = dec(390)
    if node_price_option == r"6000":
        node_license_price = dec(6000)
        #number_of_licenses_per_tier = dec(276)
    ParticipantsNum = dec(input_number_of_block_keepers())
    number_of_purchased_licenses = dec(input_number_of_licenses_per_tier_bk(ParticipantsNum))
    server_monthly_cost = dec(input_server_running_monhtly_cost())
    FRC = dec(0.675) # Function Reward Coefficient
    expected_bk_apy = expected_apy_calc(TotalSupply, KFS, u_tokens, SecondsInYear, FRC, ParticipantsNum)
    #raised_amount = node_license_price * number_of_licenses_per_tier
    implied_1_y_token_price = (node_license_price + server_monthly_cost * 12) / expected_bk_apy
    st.markdown(f"<h2 style='font-weight:bold;'>Implied 1Y Token Price ($) = {round(implied_1_y_token_price, 7)} </h2>", unsafe_allow_html=True)
    st.info(r"""
    Implied 1Y Token Price is the total yearly expenses divided by the ATR (Annual Token Reward).     
    If the token price exceeds this value, you will make a profit calculated as:    
    $P = \left(\text{Token Price} - \text{Implied 1Y Token Price} \right) \cdot \text{ATR}$
    """, icon="ℹ️")
    plot_scale = input_plot_scale()

    fig, ax = plt.subplots()
    if plot_scale <= 15:
        values_x = np.arange(0, plot_scale * SecondsInYear * 1.05,  plot_scale * SecondsInYear * 1.05 / 1000)
        ax.set_xlim([0, plot_scale * SecondsInYear * 1.05])
    else:
        values_x = np.arange(0, SecondsInYear * (plot_scale + 4) // 5 * 5 * 1.05,  SecondsInYear * (plot_scale + 4) // 5 * 5 * 1.05 / 1000)
        ax.set_xlim([0, (plot_scale + 4) // 5 * 5 * SecondsInYear * 1.05])
    values_tokens = np.array([minted_tokens_number_calc(t, TotalSupply, KFS, u_tokens, FRC, ParticipantsNum, number_of_purchased_licenses) for t in values_x])
    values_ff = np.array([minted_tokens_number_calc(t, TotalSupply, KFS, u_tokens, FRC, ParticipantsNum, number_of_purchased_licenses) * free_float(t, FFF, maxFF, u_ff) for t in values_x])
    values_stake = (values_tokens - values_ff)
    min_y_value = min(list(values_tokens))
    max_y_value = max(list(values_tokens))
    ax.plot(values_x, values_tokens, color = "Red", label = "Block Keeper Minted Tokens")
    ax.plot(values_x, values_ff, color = "Blue", label = "Block Keeper Free Float Tokens")
    ax.plot(values_x, values_stake, color = "Black", label = "Block Keeper Minimum Staked Tokens")
    ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
    y_ticks = ax.get_yticks()
    if max_y_value > 10 ** 9:
        ax.set_ylabel(r'Token Amount (in billions)')
        if any(y_tick % 1e9 != 0 for y_tick in y_ticks):
            new_labels = [f'{y_tick / 1e9:.1f}' for y_tick in y_ticks]
        else:
            new_labels = [f'{int(y_tick / 1e9)}' for y_tick in y_ticks]
    elif max_y_value > 10 ** 6:
        ax.set_ylabel(r'Token Amount (in millions)')
        if any(y_tick % 1e6 != 0 for y_tick in y_ticks):
            new_labels = [f'{y_tick / 1e6:.1f}' for y_tick in y_ticks]
        else:
            new_labels = [f'{int(y_tick / 1e6)}' for y_tick in y_ticks]
    else:
        ax.set_ylabel(r'Token Amount (in thousands)')
        new_labels = [f'{int(y_tick / 1e3)}' for y_tick in y_ticks]
    new_labels[0] = '0'
    ax.set_yticklabels(new_labels)
    if plot_scale == 1:
        xlabels = list([i for i in range(0, plot_scale * 12 + 1, 1)])
        xticks = list([i * SecondsInMonth for i in xlabels])
        ax.set_xlabel(r'Time (in months)')
    if plot_scale == 2:
        xlabels = list([i for i in range(0, plot_scale * 12 + 1, 2)])
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
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)



if node_type_option == r"Block Manager":
    node_price_option = st.selectbox(
    r"Select the node license price ($):",
    (r"400", 
    r"478", 
    r"572", 
    r"684", 
    r"818", 
    r"978", 
    r"1170", 
    r"1399", 
    r"1673", 
    r"2000"),
   index=0,
    )
    if node_price_option == r"400":
        node_license_price = dec(400)
        #number_of_licenses_per_tier = dec(2500)
    if node_price_option == r"478":
        node_license_price = dec(478)
        #number_of_licenses_per_tier = dec(2000)
    if node_price_option == r"572":
        node_license_price = dec(572)
        #number_of_licenses_per_tier = dec(1300)
    if node_price_option == r"684":
        node_license_price = dec(684)
        #number_of_licenses_per_tier = dec(1100)
    if node_price_option == r"818":
        node_license_price = dec(818)
        #number_of_licenses_per_tier = dec(800)
    if node_price_option == r"978":
        node_license_price = dec(978)
        #number_of_licenses_per_tier = dec(650)
    if node_price_option == r"1170":
        node_license_price = dec(1170)
        #number_of_licenses_per_tier = dec(530)
    if node_price_option == r"1399 $":
        node_license_price = dec(1399)
        #number_of_licenses_per_tier = dec(454)
    if node_price_option == r"1673":
        node_license_price = dec(1673)
        #number_of_licenses_per_tier = dec(390)
    if node_price_option == r"2000":
        node_license_price = dec(2000)
        #number_of_licenses_per_tier = dec(276)
    ParticipantsNum = dec(input_number_of_block_managers())
    number_of_purchased_licenses = dec(input_number_of_licenses_per_tier_bm(ParticipantsNum))
    server_monthly_cost = dec(input_server_running_monhtly_cost())
    FRC = dec(0.1) # Function Reward Coefficient
    expected_bm_apy = expected_apy_calc(TotalSupply, KFS, u_tokens, SecondsInYear, FRC, ParticipantsNum)
    #raised_amount = node_license_price * number_of_licenses_per_tier
    implied_1_y_token_price = (node_license_price + server_monthly_cost * 12) / expected_bm_apy
    st.markdown(f"<h2 style='font-weight:bold;'>Implied 1Y Token Price ($) = {round(implied_1_y_token_price, 7)} </h2>", unsafe_allow_html=True)
    st.info(r"""
    Implied 1Y Token Price is the total yearly expenses divided by the ATY (Annual Token Yield).     
    If the token price exceeds this value, you will make a profit calculated as:    
    $P = \left(\text{token price} - \text{implied token price} \right) \cdot \text{ATY}$
    """, icon="ℹ️")
    plot_scale = input_plot_scale()

    fig, ax = plt.subplots()
    if plot_scale <= 15:
        values_x = np.arange(0, plot_scale * SecondsInYear * 1.05,  plot_scale * SecondsInYear * 1.05 / 1000)
        ax.set_xlim([0, plot_scale * SecondsInYear * 1.05])
    else:
        values_x = np.arange(0, SecondsInYear * (plot_scale + 4) // 5 * 5 * 1.05,  SecondsInYear * (plot_scale + 4) // 5 * 5 * 1.05 / 1000)
        ax.set_xlim([0, (plot_scale + 4) // 5 * 5 * SecondsInYear * 1.05])
    values_tokens = np.array([minted_tokens_number_calc(t, TotalSupply, KFS, u_tokens, FRC, ParticipantsNum, number_of_purchased_licenses) for t in values_x])
    min_y_value = min(list(values_tokens))
    max_y_value = max(list(values_tokens))
    ax.plot(values_x, values_tokens, color = "Red", label = "Block Manager Minted Tokens")
    ax.set_ylim([min_y_value, max_y_value * dec(1.2)])
    y_ticks = ax.get_yticks()
    if max_y_value > 10 ** 9:
        ax.set_ylabel(r'Minted Token Amount (in billions)')
        if any(y_tick % 1e9 != 0 for y_tick in y_ticks):
            new_labels = [f'{y_tick / 1e9:.1f}' for y_tick in y_ticks]
        else:
            new_labels = [f'{int(y_tick / 1e9)}' for y_tick in y_ticks]
    elif max_y_value > 10 ** 6:
        ax.set_ylabel(r'Minted Token Amount (in millions)')
        if any(y_tick % 1e6 != 0 for y_tick in y_ticks):
            new_labels = [f'{y_tick / 1e6:.1f}' for y_tick in y_ticks]
        else:
            new_labels = [f'{int(y_tick / 1e6)}' for y_tick in y_ticks]
    else:
        ax.set_ylabel(r'Minted Token Amount (in thousands)')
        new_labels = [f'{int(y_tick / 1e3)}' for y_tick in y_ticks]
    new_labels[0] = '0'
    ax.set_yticklabels(new_labels)

    if plot_scale == 1:
        xlabels = list([i for i in range(0, plot_scale * 12 + 1, 1)])
        xticks = list([i * SecondsInMonth for i in xlabels])
        ax.set_xlabel(r'Time (in months)')
    if plot_scale == 2:
        xlabels = list([i for i in range(0, plot_scale * 12 + 1, 2)])
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

    ax.legend()
    ax.set_xticks(xticks, xlabels)
    ax.grid(True)
    st.pyplot(fig)












