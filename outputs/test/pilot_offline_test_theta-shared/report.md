# Test of `outputs/discovery/pilot_offline/best.py` on `test`

PARAMS (outputs/discovery/pilot_offline/summary.json (PARAMS and shared THETA)): `{"h_slab": 5.421370105566423, "a_spread": 0.68449950439843}`  
complexity 963 AST nodes

THETA mode `shared` (no fitting on the tested buildings: offline deployment). Open loop from the reset state; forecasts from a Kalman-filtered state with the true future inputs (a model test, not a closed-loop controller result). RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.414 | 0.675 | 0.176 | 0.261 | 0.352 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.414 | 0.675 | 0.176 | 0.261 | 0.352 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.404 | 0.978 | 0.153 | 0.264 | 0.388 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.399 | 0.569 | 0.255 | 0.335 | 0.427 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.460 | 0.613 | 0.137 | 0.217 | 0.295 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.405 | 0.619 | 0.170 | 0.248 | 0.326 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.403 | 0.598 | 0.163 | 0.243 | 0.324 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.414 | 0.675 | 0.176 | 0.261 | 0.352 | 1.196 | 1.300 | 1.743 |
