# EDA notes

From `scripts/eda.py`. Numbers come off the CSV files.

## Files

- labeled: 48,000 rows, 2025-01-01 to 2025-10-31 (304 days)
- validation: 12,000 rows, 2025-11-01 to 2025-12-31 (61 days), no `posted_rate`
- December chart: 31 rows, Lexington to Fort Wayne, 360 miles, Dry Van, 32000 lb. No lat/lon, `market_index`, or `quote_signal`.

Load ids: labeled `TR-000001`..`TR-048000`, validation `TE-000001`..`TE-012000`. Duplicates in labeled: 0, validation: 0.

## Missing

Labeled:
```
weight          300
market_index    374
```

Validation:
```
weight          165
market_index    249
```

Labeled missing `weight` by equipment:
```
equipment
Dry Van    175
Flatbed     55
Reefer      70
```

Labeled missing `market_index` by equipment:
```
equipment
Dry Van    202
Flatbed     73
Reefer      99
```

Same holes show up in validation, so dropping rows is not an option for the 12k file.

## Target and distance

posted_rate: min 57.22, median 2030.76, mean 2373.98, max 25533.00

distance: min 70.0, median 953.3, mean 1135.9, max 3439.8

USD/mile: min 0.334, median 2.145, mean 2.215, max 14.125

corr(distance, posted_rate) = 0.909
corr(quote_signal, usd/mile) = 0.049
corr(market_index, usd/mile) = 0.083
corr(quote_signal * distance, posted_rate) = 0.899

quote_signal is not a ready-made $/mile. Distance dominates the dollar amount. Multiplying by quote_signal does not beat distance alone.

## Equipment

Labeled: {'Dry Van': 27202, 'Reefer': 12045, 'Flatbed': 8753}
Validation: {'Dry Van': 6780, 'Reefer': 3051, 'Flatbed': 2169}

Median USD/mile by equipment:
```
equipment
Dry Van    2.046203
Flatbed    2.219683
Reefer     2.311540
```

## Time

Loads per month (labeled):
```
month
2025-01    4918
2025-02    4337
2025-03    5036
2025-04    4819
2025-05    4913
2025-06    4783
2025-07    4912
2025-08    4759
2025-09    4670
2025-10    4853
```

Validation:
```
month
2025-11    5836
2025-12    6164
```

Median posted_rate by month:
```
month
2025-01    1915.200
2025-02    1994.250
2025-03    2022.920
2025-04    2044.240
2025-05    2065.510
2025-06    2120.220
2025-07    2059.145
2025-08    2015.730
2025-09    2057.130
2025-10    2035.900
```

Median USD/mile by month:
```
month
2025-01    2.029158
2025-02    2.061143
2025-03    2.139279
2025-04    2.146699
2025-05    2.190194
2025-06    2.247885
2025-07    2.187422
2025-08    2.120525
2025-09    2.159622
2025-10    2.163828
```

Weekday median rate 2030.03 vs weekend 2031.72.
Weekday median USD/mile 2.151 vs weekend 2.130.
Weekend vs weekday is basically flat. The month pattern is the one to keep (USD/mile peaks in June, lower in Jan).

Sep-Oct is the last labeled window before the Nov-Dec file, so that is the holdout. A random row split would mix those months.

## Cities and lanes

Labeled cities: 64. Validation cities: 72. New in validation: ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']
New pickups: ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']
New deliveries: ['Allentown', 'Charlotte', 'Chicago', 'Jackson', 'Knoxville', 'Laredo', 'Norfolk', 'San Diego']

Unique pickup-delivery pairs: 4014 (median 10 loads, max 39)
Unique pickup-delivery-equipment lanes: 10793 (median 3 loads, max 28)

Most lanes are too thin for a per-lane time-series model.

Each city name maps to one lat/lon in the labeled file (max 1 pair per city).

## Lexington to Fort Wayne

Labeled rows on that OD: 32 across 31 days.
Equipment: {'Dry Van': 21, 'Reefer': 8, 'Flatbed': 3}
Dry Van median rate 807.89 (n=21), median USD/mile 2.233.
Chart distance is 360 vs labeled Dry Van distances 350.9-378.8.

Not much history on the exact chart lane. Calendar features have to do some of the work in December, and `market_index` / `quote_signal` cannot be required.

## Plots

- `reports/eda_monthly_rate.png`
- `reports/eda_rpm_by_equipment.png`
