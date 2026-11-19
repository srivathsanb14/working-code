# Test of `outputs/discovery/pilot_offline/best.py` on `test`

PARAMS (fitted on train (with THETA averaged over train buildings)): `{"h_slab": 5.189957865558505, "a_spread": 1.3432165317770008}`  
complexity 963 AST nodes

THETA mode `trainmean` (no fitting on the tested buildings: offline deployment). Open loop from the reset state; forecasts from a Kalman-filtered state with the true future inputs (a model test, not a closed-loop controller result). RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.323 | 0.645 | 0.142 | 0.225 | 0.325 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.323 | 0.645 | 0.142 | 0.225 | 0.325 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.326 | 0.966 | 0.122 | 0.217 | 0.339 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.313 | 0.608 | 0.211 | 0.302 | 0.412 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.330 | 0.556 | 0.120 | 0.197 | 0.286 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.323 | 0.550 | 0.130 | 0.205 | 0.297 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.323 | 0.544 | 0.125 | 0.202 | 0.294 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.323 | 0.645 | 0.142 | 0.225 | 0.325 | 1.196 | 1.300 | 1.743 |
