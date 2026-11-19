# Deployment options for `outputs/discovery/pilot_offline/best.py` on `test`

Offline modes fit nothing on the tested buildings; online fits θ on each building's day 1. Forecasts are from a Kalman-filtered state with the true future inputs. RMSE in K.

| θ | open_loop_day1 | open_loop_after_day1 | forecast_1h | forecast_3h | forecast_6h | persistence_3h |
|---|---|---|---|---|---|---|
| shared θ (offline) | 0.414 | 0.675 | 0.176 | 0.261 | 0.352 | 1.300 |
| train-mean θ (offline) | 0.323 | 0.645 | 0.142 | 0.225 | 0.325 | 1.300 |
| prior θ (offline) | 0.373 | 0.899 | 0.156 | 0.261 | 0.397 | 1.300 |
| θ fitted on day 1 (online) | 0.155 | 0.583 | 0.138 | 0.203 | 0.286 | 1.300 |
