# Processed research data

`processed/` contains only the de-identified information needed to reproduce
the paper case study. It does **not** contain the original terminal database,
workbooks, container identifiers, personnel data, absolute timestamps, source
row numbers, local paths, or records from other vessels.

- `MSAIV610A_tasks_processed.csv`: synthetic task ID, relative release minute,
  decision step, origin region, and destination region.
- `task_arrivals_15min.csv`: 15-minute OD arrivals derived from those tasks.
- `region_network_edges.csv`: the 34 directed region-network edges.
- `travel_time_matrix.csv`: 240 non-self OD mean travel times in minutes.
- `region_metadata.csv`: synthetic region identifiers and names.
- `scenario_MSAIV610A_processed.json`: public scenario used by the released
  training and evaluation code.
- `cict_16_region_layout.json`: abstract model layout; it is not a surveyed map
  and contains no Google Maps imagery.

Regenerate and validate the CSV views with:

```bash
python scripts/prepare_data.py
```

