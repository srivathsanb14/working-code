"""The reward as a cost a controller can evaluate on predictions.

Comfort is exact given room temperatures, the active band and occupants per room (the kit's shortfall: linear up
to 5 K outside the band, then quadratic, unbounded; regret = - occupancy-weighted mean shortfall).  Energy and
carbon use the kit's own price and carbon tables (copied below from smart_control/reward/*_energy_cost.py and
utils/constants.py, Apache-2.0) applied to POWER ESTIMATED FROM COMMANDS:
  fans      supply + exhaust: flow x pressure / 0.9 (exhaust carries the non-recirculated share)
  cooling   rho c x flow x (mixed - supply)+ / COP 3.5
  pump      water flow x 9810 x head / 0.9 (head = the pump differential-pressure command)
  gas       reheat heat delivered to the rooms (+ boiler tank heating when the water setpoint rises)
The estimate is checked against the kit's logged costs by control/validate_costs.py.
Normalisation: costs divided by the cost of 80,000 W (electricity) and 205,000 W (gas) per AHU at the same time.
"""
from __future__ import annotations

import numpy as np

W_COMFORT, W_ENERGY, W_CARBON = 0.6, 0.2, 0.2
DEV_REF = 5.0
MAX_ELEC_PER_AHU, MAX_GAS_PER_AHU = 80000.0, 205000.0
FAN_EFF, COP, PUMP_EFF, RHO_C_AIR = 0.9, 3.5, 0.9, 1.2 * 1006.0
TANK_J_PER_K = 0.0           # boiler tank heat capacity is not in the metadata; left out until validate_costs says otherwise

ELEC_WEEKDAY_C_PER_KWH = (16, 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 18, 20, 20, 20, 20, 20, 20, 20, 16, 16, 16, 16, 16)
ELEC_WEEKEND_C_PER_KWH = (16,) * 24
ELEC_KG_PER_MWH = (88.19666493, 87.79190866, 87.87607686, 87.83054163, 88.00279618, 88.19648183, 89.70663283, 93.97947901,
                   98.85868291, 100.7853521, 101.3866866, 101.7795612, 102.5919168, 103.4403736, 104.1380294, 104.7359292,
                   102.0714466, 97.04226176, 93.57895651, 92.46355045, 91.72914657, 90.69209747, 89.76552213, 88.99950995)
GAS_USD_PER_KFT3_BY_MONTH = (9.02, 8.35, 7.77, 7.26, 6.69, 6.86, 6.77, 6.76, 6.99, 7.19, 7.96, 8.98)
KWH_PER_KFT3_GAS, J_PER_KWH, GAS_CO2_PER_KFT3 = 293.07107, 3.6e6, 53.12


def shortfall(dev):
    dev = np.maximum(dev, 0.0)
    return np.where(dev <= DEV_REF, dev / DEV_REF, 1.0 + ((dev - DEV_REF) / DEV_REF) ** 2)


def comfort_regret(T, lo, hi, occ_rooms):
    """Normalized productivity regret (<= 0) for room temperatures T (..., V), band lo/hi, occupants per room (..., V)."""
    dev = np.maximum(T - hi, 0.0) + np.maximum(lo - T, 0.0)
    occ = np.asarray(occ_rooms, float)
    tot = occ.sum(-1)
    return np.where(tot > 0, -(occ * shortfall(dev)).sum(-1) / np.maximum(tot, 1e-12), 0.0)


def prices(hour, workday, month):
    """USD/J electricity, kgC/J electricity, USD/J gas, kgC/J gas at this time."""
    e = (ELEC_WEEKDAY_C_PER_KWH if workday else ELEC_WEEKEND_C_PER_KWH)[int(hour) % 24] / 100.0 / 1000.0 / 3600.0
    ec = ELEC_KG_PER_MWH[int(hour) % 24] / 1.0e6 / 3600.0
    gp = np.mean(GAS_USD_PER_KFT3_BY_MONTH) if month is None else GAS_USD_PER_KFT3_BY_MONTH[(int(month) - 1) % 12]
    g = gp / KWH_PER_KFT3_GAS / J_PER_KWH            # month None: the policy cannot read the month, use the annual mean
    gc = GAS_CO2_PER_KFT3 / KWH_PER_KFT3_GAS / J_PER_KWH
    return e, ec, g, gc


def power(aux):
    """Electric and gas power (W) estimated from the equipment quantities of one step (runtime.inputs aux=True)."""
    f, P, r = np.asarray(aux["ahu_flow"], float), np.asarray(aux["ahu_pressure"], float), np.asarray(aux["recirculation"], float)
    fans = float((f * P / FAN_EFF * (1.0 + (1.0 - r))).sum())
    cooling = float((RHO_C_AIR * f * np.maximum(np.asarray(aux["mixed"]) - np.asarray(aux["T_supply_ahu"]), 0.0)).sum() / COP)
    pump = float(aux["water_flow"] * 9810.0 * aux["pump_dp"] / PUMP_EFF) if aux["hws_on"] else 0.0
    tank = TANK_J_PER_K * max(aux["T_water_setpoint"] - aux["T_water"], 0.0) / 300.0 if aux["hws_on"] else 0.0
    gas = max(float(aux["Q_reheat_total"]), 0.0) + tank
    return dict(electric=fans + cooling + pump, gas=gas, fans=fans, cooling=cooling, pump=pump)


def normalized_energy_carbon(elec_W, gas_W, K, hour, workday, month):
    e, ec, g, gc = prices(hour, workday, month)
    E = (e * elec_W + g * gas_W) / (e * MAX_ELEC_PER_AHU * K + g * MAX_GAS_PER_AHU * K)
    C = (ec * elec_W + gc * gas_W) / (ec * MAX_ELEC_PER_AHU * K + gc * MAX_GAS_PER_AHU * K)
    return E, C


def reward(regret, E, C):
    return W_COMFORT * regret - W_ENERGY * E - W_CARBON * C
