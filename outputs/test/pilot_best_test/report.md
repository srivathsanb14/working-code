# Test of `outputs/discovery/pilot/best.py` on `test`

PARAMS (outputs/discovery/pilot/summary.json): `{"h_slab": 5.191306111654754, "a_spread": 1.3432985763289358}`  
complexity 963 AST nodes

Open loop from the reset state with THETA fitted on day 1; forecasts from a Kalman-filtered state. RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 15 | 0.145 | 0.543 | 0.127 | 0.184 | 0.256 | 1.196 | 1.300 | 1.743 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| minneapolis | 15 | 0.145 | 0.543 | 0.127 | 0.184 | 0.256 | 1.196 | 1.300 | 1.743 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 3 | 0.117 | 0.440 | 0.078 | 0.129 | 0.193 | 0.263 | 0.651 | 1.114 |
| bang_bang | 3 | 0.126 | 0.636 | 0.223 | 0.300 | 0.396 | 3.136 | 2.781 | 3.376 |
| fixed_setpoint | 3 | 0.228 | 0.565 | 0.098 | 0.149 | 0.214 | 0.824 | 0.892 | 1.222 |
| pid | 3 | 0.128 | 0.541 | 0.121 | 0.172 | 0.238 | 0.915 | 1.103 | 1.498 |
| pid_water | 3 | 0.126 | 0.532 | 0.117 | 0.170 | 0.239 | 0.843 | 1.075 | 1.506 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_0801 | 15 | 0.145 | 0.543 | 0.127 | 0.184 | 0.256 | 1.196 | 1.300 | 1.743 |
