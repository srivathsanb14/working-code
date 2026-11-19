# Test of `outputs/discovery/pilot_offline/best.py` on `test`

PARAMS (outputs/discovery/pilot_offline/summary.json): `{"h_slab": 5.421370105566423, "a_spread": 0.68449950439843}`  
complexity 963 AST nodes

THETA mode `day1` (fitted on day 1 of each building: online identification). Open loop from the reset state; forecasts from a Kalman-filtered state with the true future inputs (a model test, not a closed-loop controller result). RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.155 | 0.583 | 0.138 | 0.203 | 0.286 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.155 | 0.583 | 0.138 | 0.203 | 0.286 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.135 | 0.487 | 0.096 | 0.160 | 0.239 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.141 | 0.672 | 0.235 | 0.316 | 0.421 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.216 | 0.606 | 0.104 | 0.162 | 0.236 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.143 | 0.581 | 0.128 | 0.188 | 0.267 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.141 | 0.571 | 0.125 | 0.187 | 0.268 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.155 | 0.583 | 0.138 | 0.203 | 0.286 | 1.196 | 1.300 | 1.743 |
