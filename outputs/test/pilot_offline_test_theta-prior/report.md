# Test of `outputs/discovery/pilot_offline/best.py` on `test`

PARAMS (outputs/discovery/pilot_offline/summary.json): `{"h_slab": 5.421370105566423, "a_spread": 0.68449950439843}`  
complexity 963 AST nodes

THETA mode `prior` (no fitting on the tested buildings: offline deployment). Open loop from the reset state; forecasts from a Kalman-filtered state with the true future inputs (a model test, not a closed-loop controller result). RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.373 | 0.899 | 0.156 | 0.261 | 0.397 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.373 | 0.899 | 0.156 | 0.261 | 0.397 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.410 | 1.288 | 0.157 | 0.274 | 0.428 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.393 | 0.801 | 0.209 | 0.328 | 0.477 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.258 | 0.793 | 0.140 | 0.238 | 0.360 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.402 | 0.815 | 0.137 | 0.235 | 0.361 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.402 | 0.800 | 0.136 | 0.232 | 0.358 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.373 | 0.899 | 0.156 | 0.261 | 0.397 | 1.196 | 1.300 | 1.743 |
