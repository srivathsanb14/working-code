# Test of `code/discovery/seeds/seed_slab_wall_network.py` on `val`

PARAMS (fitted on train): `{"h_slab": 6.8720349080012495, "a_spread": 1.5590954525591183}`  
complexity 963 AST nodes

Open loop from the reset state with THETA fitted on day 1; forecasts from a Kalman-filtered state. RMSE in K.


## Overall

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all | 5 | 0.278 | 0.287 | 0.128 | 0.171 | 0.209 | 1.705 | 2.061 | 2.464 |

## By station

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| phoenix | 5 | 0.278 | 0.287 | 0.128 | 0.171 | 0.209 | 1.705 | 2.061 | 2.464 |

## By policy

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| all_off | 1 | 0.281 | 0.568 | 0.115 | 0.199 | 0.290 | 0.371 | 0.999 | 1.749 |
| bang_bang | 1 | 0.316 | 0.217 | 0.194 | 0.214 | 0.224 | 2.491 | 3.727 | 4.755 |
| fixed_setpoint | 1 | 0.270 | 0.214 | 0.112 | 0.141 | 0.167 | 2.219 | 2.879 | 2.527 |
| pid | 1 | 0.262 | 0.215 | 0.112 | 0.153 | 0.183 | 1.747 | 1.476 | 1.635 |
| pid_water | 1 | 0.262 | 0.218 | 0.109 | 0.150 | 0.183 | 1.698 | 1.225 | 1.656 |

## By building

| | runs | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_1h | persistence_3h | persistence_6h |
|---|---|---|---|---|---|---|---|---|---|
| building_train_7 | 5 | 0.278 | 0.287 | 0.128 | 0.171 | 0.209 | 1.705 | 2.061 | 2.464 |
