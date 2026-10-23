# Early-change analysis (days 0–3)

All pixel maps are **exploratory: spectral difference from healthy reference**, never disease area.
Healthy reference = bi leaves (both groups) + control leaves at all time points.
Distances use a leave-one-unit-out reference when unit labels are present.

WARNING: leaf order within each scan is UNCONFIRMED. The leaf_map must be checked against the dataset's own metadata or confirmed with the dataset authors (Song Li, songli@vt.edu). Do not treat assumed unit/group labels as ground truth.

Lines should overlap at bi; any divergence later is the early signal.

## Per time point × group (mean ± 95% bootstrap CI)

time_point      group  n  p_infected_mean  p_infected_lo  p_infected_hi  atypical_pct_mean  atypical_pct_lo  atypical_pct_hi  median_distance_mean  median_distance_lo  median_distance_hi
        bi    control  0              NaN            NaN            NaN                NaN              NaN              NaN                   NaN                 NaN                 NaN
        bi inoculated  0              NaN            NaN            NaN                NaN              NaN              NaN                   NaN                 NaN                 NaN
       2hr    control  0              NaN            NaN            NaN                NaN              NaN              NaN                   NaN                 NaN                 NaN
       2hr inoculated  0              NaN            NaN            NaN                NaN              NaN              NaN                   NaN                 NaN                 NaN
        d1    control  4         0.363318       0.098954       0.773767           1.036226         0.388046         1.719136              2.836269            2.647932            2.971596
        d1 inoculated  4         0.343420       0.139212       0.552520           0.681855         0.147686         1.216023              2.727336            2.395378            3.192606
        d2    control  4         0.590469       0.250385       0.930552           2.788255         0.108873         7.881138              2.980436            2.458040            3.588083
        d2 inoculated  4         0.544946       0.182874       0.907017           1.605811         0.991847         2.564990              2.742575            2.400591            3.321954
        d3    control  4         0.511645       0.031538       0.991751           1.371926         0.309707         2.987557              2.663732            2.485081            2.878884
        d3 inoculated  4         0.311420       0.062794       0.560047           2.698443         0.481037         6.742741              2.900878            2.571142            3.369231

## AUC inoculated vs control

time_point  n  n_inoculated  n_control  auc_p_infected  auc_atypical_pct
        bi  5             0          5             NaN               NaN
       2hr  6             0          6             NaN               NaN
        d1  8             4          4          0.5625            0.4375
        d2  8             4          4          0.4375            0.7500
        d3  8             4          4          0.4375            0.5625

