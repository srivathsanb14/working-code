# Test of `outputs/discovery/pilot/best.py` on `test`

PARAMS (outputs/discovery/pilot/summary.json): `{"h_slab": 5.191306111654754, "a_spread": 1.3432985763289358}`  
complexity 963 AST nodes

Open loop from the reset state with THETA fitted on day 1; forecasts from a Kalman-filtered state. RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.361 | 0.625 | 0.138 | 0.222 | 0.324 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.361 | 0.625 | 0.138 | 0.222 | 0.324 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.356 | 0.886 | 0.122 | 0.215 | 0.335 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.351 | 0.618 | 0.199 | 0.294 | 0.406 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.388 | 0.540 | 0.122 | 0.200 | 0.292 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.357 | 0.541 | 0.125 | 0.202 | 0.296 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.356 | 0.537 | 0.122 | 0.198 | 0.293 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.361 | 0.625 | 0.138 | 0.222 | 0.324 | 1.196 | 1.300 | 1.743 |
