# Statistical Analysis of Model Performance

## Intersect Filter

### build_time_s

#### Descriptive Statistics

|           |   count |       mean |        std |         min |         25% |         50% |        75% |        max |
|:----------|--------:|-----------:|-----------:|------------:|------------:|------------:|-----------:|-----------:|
| Histogram |      14 |  381.857   | 1321.04    |  0.0001     |   0.750075  |   11.5      |   70       | 4970       |
| RTree     |      14 |  713.702   | 1211.85    |  1.64852    |  15.9835    |  114.331    |  699.47    | 3553.07    |
| KNN       |      14 |    1.46216 |    2.31528 |  0.00476584 |   0.0292055 |    0.235096 |    1.77557 |    6.89325 |
| NN        |      14 | 2826.39    | 2649.86    | 11.2021     | 424.811     | 2128.69     | 5052.36    | 7940.58    |
| RF        |      14 |  674.268   |  992.88    |  1.2241     |  27.0777    |  146.056    |  930.314   | 3002.09    |
| XGB       |      14 |   45.8915  |   49.713   |  4.48356    |  12.0644    |   22.543    |   66.0462  |  182.137   |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.0323955   |   0.0764923   |   0.000176926 |   0.00933315  |   0.189755    |
| RTree     |   0.0323955   | nan           |   4.77512e-05 |   0.0094304   |   0.7652      |   0.10286     |
| KNN       |   0.0764923   |   4.77512e-05 | nan           |   7.46784e-06 |   1.73823e-05 |   1.14396e-05 |
| NN        |   0.000176926 |   0.0094304   |   7.46784e-06 | nan           |   0.0229413   |   0.000103354 |
| RF        |   0.00933315  |   0.7652      |   1.73823e-05 |   0.0229413   | nan           |   0.0258497   |
| XGB       |   0.189755    |   0.10286     |   1.14396e-05 |   0.000103354 |   0.0258497   | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/intersect_build_time_s_boxplot.png)

---

### model_size_mb

#### Descriptive Statistics

|           |   count |       mean |         std |       min |       25% |        50% |        75% |         max |
|:----------|--------:|-----------:|------------:|----------:|----------:|-----------:|-----------:|------------:|
| Histogram |      14 |   1.24235  |   0.862236  | 0.0247879 |  0.226904 |   1.57366  |   2        |    2        |
| RTree     |      14 | 426.181    | 591.538     | 2.19922   | 19.1465   | 131.639    | 599.017    | 1866.9      |
| KNN       |      14 |  68.7028   |  92.5658    | 0.340713  |  3.16799  |  21.9168   |  98.3339   |  275.338    |
| NN        |      14 |   0.189622 |   0.0793069 | 0.0684452 |  0.108746 |   0.232964 |   0.23939  |    0.248583 |
| RF        |      14 | 253.157    | 331.9       | 0.752881  | 26.0887   |  74.7593   | 382.462    |  958.73     |
| XGB       |      14 |   0.495723 |   0.154112  | 0.0386591 |  0.473188 |   0.560676 |   0.584851 |    0.59197  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   6.74725e-06 |   0.00521822  |   0.0222904   |   4.38901e-05 |   0.0399239   |
| RTree     |   6.74725e-06 | nan           |   0.0456382   |   7.46784e-06 |   0.7652      |   7.46784e-06 |
| KNN       |   0.00521822  |   0.0456382   | nan           |   7.46784e-06 |   0.062761    |   8.54679e-05 |
| NN        |   0.0222904   |   7.46784e-06 |   7.46784e-06 | nan           |   7.46784e-06 |   0.000124734 |
| RF        |   4.38901e-05 |   0.7652      |   0.062761    |   7.46784e-06 | nan           |   7.46784e-06 |
| XGB       |   0.0399239   |   7.46784e-06 |   8.54679e-05 |   0.000124734 |   7.46784e-06 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/intersect_model_size_mb_boxplot.png)

---

### avg_time_ms

#### Descriptive Statistics

|           |   count |      mean |        std |        min |        25% |        50% |        75% |       max |
|:----------|--------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|
| Histogram |      14 | 0.0817906 | 0.0395046  | 0.0260502  | 0.0354816  | 0.0966011  | 0.11571    | 0.121625  |
| RTree     |      14 | 0.0331937 | 0.02804    | 0.00818972 | 0.0096734  | 0.0223655  | 0.0493565  | 0.092796  |
| KNN       |      14 | 0.558751  | 0.256122   | 0.0452346  | 0.401427   | 0.597134   | 0.66674    | 0.929789  |
| NN        |      14 | 0.0865627 | 0.0479308  | 0.00929594 | 0.050632   | 0.0855084  | 0.111204   | 0.19835   |
| RF        |      14 | 0.0527389 | 0.0519269  | 0.0084062  | 0.0211306  | 0.0343806  | 0.0523997  | 0.185848  |
| XGB       |      14 | 0.007504  | 0.00273214 | 0.00279781 | 0.00584677 | 0.00668987 | 0.00888126 | 0.0133422 |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.00140626  |   5.8094e-05  |   0.800491    |   0.062761    |   7.46784e-06 |
| RTree     |   0.00140626  | nan           |   2.13617e-05 |   0.0022467   |   0.301219    |   0.000259355 |
| KNN       |   5.8094e-05  |   2.13617e-05 | nan           |   5.8094e-05  |   1.73823e-05 |   7.46784e-06 |
| NN        |   0.800491    |   0.0022467   |   5.8094e-05  | nan           |   0.0290716   |   1.41156e-05 |
| RF        |   0.062761    |   0.301219    |   1.73823e-05 |   0.0290716   | nan           |   1.73823e-05 |
| XGB       |   7.46784e-06 |   0.000259355 |   7.46784e-06 |   1.41156e-05 |   1.73823e-05 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/intersect_avg_time_ms_boxplot.png)

---

### mae

#### Descriptive Statistics

|           |   count |    mean |      std |     min |      25% |      50% |      75% |      max |
|:----------|--------:|--------:|---------:|--------:|---------:|---------:|---------:|---------:|
| Histogram |      14 | 5404.75 | 11147.9  | 101.21  |  280.594 |  835.104 |  4657.2  | 42363.3  |
| RTree     |      14 | 7219.21 |  6192.54 | 445.122 | 2143.37  | 5451.77  |  9054.67 | 19192    |
| KNN       |      14 | 7117.79 |  8074.45 | 216.08  | 1077.11  | 4208.99  |  9659.76 | 24634.3  |
| NN        |      14 | 7694.55 |  9315.61 | 298.38  | 1220.78  | 2654.65  | 12002.4  | 29935.5  |
| RF        |      14 | 1874.48 |  1910.12 |  98.29  |  306.42  | 1050.11  |  3323.68 |  5874.89 |
| XGB       |      14 | 5856.82 |  7751.83 | 159.03  |  350.47  | 1707.29  |  8958.28 | 22621.3  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |   Histogram |        RTree |         KNN |          NN |           RF |        XGB |
|:----------|------------:|-------------:|------------:|------------:|-------------:|-----------:|
| Histogram | nan         |   0.0565428  |   0.190362  |   0.10286   |   0.7652     |   0.597223 |
| RTree     |   0.0565428 | nan          |   0.565734  |   0.662472  |   0.00543866 |   0.280244 |
| KNN       |   0.190362  |   0.565734   | nan         |   0.945052  |   0.0408872  |   0.395307 |
| NN        |   0.10286   |   0.662472   |   0.945052  | nan         |   0.0365622  |   0.301219 |
| RF        |   0.7652    |   0.00543866 |   0.0408872 |   0.0365622 | nan          |   0.421348 |
| XGB       |   0.597223  |   0.280244   |   0.395307  |   0.301219  |   0.421348   | nan        |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/intersect_mae_boxplot.png)

---

### mape

#### Descriptive Statistics

|           |   count |      mean |       std |      min |      25% |      50% |       75% |      max |
|:----------|--------:|----------:|----------:|---------:|---------:|---------:|----------:|---------:|
| Histogram |      14 |   56.7207 |   36.2871 |  24.1138 |  36.1554 |  49.0019 |   61.7415 |  172.071 |
| RTree     |      14 |  903.83   |  797.491  |  73.864  | 400.147  | 664.967  | 1073.12   | 3149.5   |
| KNN       |      14 |  245.426  |  181.264  |  21.98   | 133.542  | 221.535  |  300.35   |  777.38  |
| NN        |      14 | 2085.78   | 2615.77   | 112.74   | 372.645  | 493.36   | 3580.46   | 8244.39  |
| RF        |      14 |   80.5371 |  113.445  |   3.1    |  36.0675 |  49.74   |   74.86   |  462.65  |
| XGB       |      14 | 1206.8    | 1748.69   |  53.65   | 197.472  | 312.69   | 1744.8    | 6142.8   |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   1.14396e-05 |   0.000216631 |   9.25216e-06 |   0.7652      |   3.20674e-05 |
| RTree     |   1.14396e-05 | nan           |   0.000864703 |   0.662472    |   3.20674e-05 |   0.421348    |
| KNN       |   0.000216631 |   0.000864703 | nan           |   0.000619108 |   0.00101889  |   0.135359    |
| NN        |   9.25216e-06 |   0.662472    |   0.000619108 | nan           |   2.61992e-05 |   0.123744    |
| RF        |   0.7652      |   3.20674e-05 |   0.00101889  |   2.61992e-05 | nan           |   0.000180584 |
| XGB       |   3.20674e-05 |   0.421348    |   0.135359    |   0.123744    |   0.000180584 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/intersect_mape_boxplot.png)

---

## Contain Filter

### build_time_s

#### Descriptive Statistics

|           |   count |        mean |        std |          min |          25% |          50% |          75% |          max |
|:----------|--------:|------------:|-----------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| Histogram |      14 |   381.857   |  1321.04   |   0.0001     |    0.750075  |    11.5      |     70       |   4970       |
| RTree     |      14 |   713.702   |  1211.85   |   1.64852    |   15.9835    |   114.331    |    699.47    |   3553.07    |
| KNN       |      14 |     1.42213 |     2.2819 |   0.00331709 |    0.0273792 |     0.223848 |      1.63868 |      6.87434 |
| NN        |      14 | 68679.2     | 72627.5    | 142.236      | 5872.01      | 40075.6      | 118840       | 236858       |
| RF        |      14 |   692.941   |  1014.83   |   2.4038     |   28.6797    |   141.602    |    929.239   |   3054.71    |
| XGB       |      14 |    54.1716  |    62.681  |   5.47562    |   11.1722    |    22.5117   |     72.6088  |    199.265   |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.0323955   |   0.0764923   |   1.69264e-05 |   0.00933315  |   0.205766    |
| RTree     |   0.0323955   | nan           |   4.77512e-05 |   0.000180584 |   0.696126    |   0.161093    |
| KNN       |   0.0764923   |   4.77512e-05 | nan           |   7.46784e-06 |   1.41156e-05 |   1.14396e-05 |
| NN        |   1.69264e-05 |   0.000180584 |   7.46784e-06 | nan           |   0.000259355 |   1.14396e-05 |
| RF        |   0.00933315  |   0.696126    |   1.41156e-05 |   0.000259355 | nan           |   0.0290716   |
| XGB       |   0.205766    |   0.161093    |   1.14396e-05 |   1.14396e-05 |   0.0290716   | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/contain_build_time_s_boxplot.png)

---

### model_size_mb

#### Descriptive Statistics

|           |   count |       mean |         std |       min |       25% |        50% |        75% |         max |
|:----------|--------:|-----------:|------------:|----------:|----------:|-----------:|-----------:|------------:|
| Histogram |      14 |   1.24235  |   0.862236  | 0.0247879 |  0.226904 |   1.57366  |   2        |    2        |
| RTree     |      14 | 426.181    | 591.538     | 2.19922   | 19.1465   | 131.639    | 599.017    | 1866.9      |
| KNN       |      14 |  68.6769   |  92.5712    | 0.340595  |  3.1679   |  21.9155   |  98.3337   |  275.337    |
| NN        |      14 |   0.190916 |   0.0801877 | 0.0683203 |  0.110307 |   0.236698 |   0.239914 |    0.248972 |
| RF        |      14 | 250.399    | 331.398     | 0.908158  | 26.0883   |  74.7627   | 373.053    |  958.589    |
| XGB       |      14 |   0.515915 |   0.117478  | 0.13769   |  0.493313 |   0.550566 |   0.585239 |    0.589332 |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   6.74725e-06 |   0.00521822  |   0.0222904   |   4.38901e-05 |   0.0553616   |
| RTree     |   6.74725e-06 | nan           |   0.0456382   |   7.46784e-06 |   0.730389    |   7.46784e-06 |
| KNN       |   0.00521822  |   0.0456382   | nan           |   7.46784e-06 |   0.0565428   |   0.000103354 |
| NN        |   0.0222904   |   7.46784e-06 |   7.46784e-06 | nan           |   7.46784e-06 |   5.8094e-05  |
| RF        |   4.38901e-05 |   0.730389    |   0.0565428   |   7.46784e-06 | nan           |   7.46784e-06 |
| XGB       |   0.0553616   |   7.46784e-06 |   0.000103354 |   5.8094e-05  |   7.46784e-06 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/contain_model_size_mb_boxplot.png)

---

### avg_time_ms

#### Descriptive Statistics

|           |   count |      mean |        std |        min |        25% |        50% |        75% |       max |
|:----------|--------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|----------:|
| Histogram |      14 | 0.0848974 | 0.041647   | 0.0269899  | 0.035687   | 0.0983098  | 0.120783   | 0.126714  |
| RTree     |      14 | 0.0375163 | 0.0320679  | 0.0090888  | 0.0108201  | 0.024475   | 0.056371   | 0.105375  |
| KNN       |      14 | 0.459641  | 0.260841   | 0.0459008  | 0.295874   | 0.491832   | 0.608708   | 0.939603  |
| NN        |      14 | 0.0500847 | 0.0213825  | 0.019427   | 0.0284576  | 0.058831   | 0.0637099  | 0.0803149 |
| RF        |      14 | 0.0509195 | 0.0460417  | 0.00796516 | 0.0211334  | 0.0348516  | 0.0635187  | 0.16388   |
| XGB       |      14 | 0.0106894 | 0.00519855 | 0.00618345 | 0.00813393 | 0.00881575 | 0.00989797 | 0.025601  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.00304038  |   5.8094e-05  |   0.0229413   |   0.0695342   |   7.46784e-06 |
| RTree     |   0.00304038  | nan           |   2.13617e-05 |   0.112922    |   0.395307    |   0.000864703 |
| KNN       |   5.8094e-05  |   2.13617e-05 | nan           |   4.77512e-05 |   2.13617e-05 |   7.46784e-06 |
| NN        |   0.0229413   |   0.112922    |   4.77512e-05 | nan           |   0.395307    |   1.73823e-05 |
| RF        |   0.0695342   |   0.395307    |   2.13617e-05 |   0.395307    | nan           |   0.000150233 |
| XGB       |   7.46784e-06 |   0.000864703 |   7.46784e-06 |   1.73823e-05 |   0.000150233 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/contain_avg_time_ms_boxplot.png)

---

### mae

#### Descriptive Statistics

|           |   count |    mean |      std |     min |      25% |      50% |      75% |      max |
|:----------|--------:|--------:|---------:|--------:|---------:|---------:|---------:|---------:|
| Histogram |      14 | 5127.55 | 10145.2  | 113.548 |  281.744 |  841.127 |  4679.38 | 38414.6  |
| RTree     |      14 | 6221.87 |  5153.39 | 436.546 | 2142.85  | 5205.71  |  8630.26 | 16560.8  |
| KNN       |      14 | 7087.91 |  8082.51 | 215.69  | 1077.1   | 4208.76  |  9659.73 | 24634.2  |
| NN        |      14 | 7535.07 |  8829.84 | 279.57  | 1268.48  | 2887.71  | 12265.5  | 27667.7  |
| RF        |      14 | 1866.87 |  1906.79 | 104.65  |  306.647 | 1049.7   |  3296.2  |  5872.38 |
| XGB       |      14 | 5564.27 |  7281.09 | 108.98  |  350.165 | 1683     |  8825.75 | 22892.3  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |   Histogram |        RTree |         KNN |          NN |           RF |        XGB |
|:----------|------------:|-------------:|------------:|------------:|-------------:|-----------:|
| Histogram | nan         |   0.0695342  |   0.206388  |   0.10286   |   0.7652     |   0.662472 |
| RTree     |   0.0695342 | nan          |   0.629486  |   0.730389  |   0.00625915 |   0.280244 |
| KNN       |   0.206388  |   0.629486   | nan         |   0.981671  |   0.0408872  |   0.395307 |
| NN        |   0.10286   |   0.730389   |   0.981671  | nan         |   0.0456382  |   0.323214 |
| RF        |   0.7652    |   0.00625915 |   0.0408872 |   0.0456382 | nan          |   0.421348 |
| XGB       |   0.662472  |   0.280244   |   0.395307  |   0.323214  |   0.421348   | nan        |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/contain_mae_boxplot.png)

---

### mape

#### Descriptive Statistics

|           |   count |     mean |      std |      min |     25% |       50% |       75% |      max |
|:----------|--------:|---------:|---------:|---------:|--------:|----------:|----------:|---------:|
| Histogram |      14 | 2481.49  | 9067.26  |  17.3086 |  38.735 |   53.7051 |   72.9059 | 33984.5  |
| RTree     |      14 |  939.849 |  790.211 | 350.92   | 466.13  |  629.788  | 1074.29   |  3149.5  |
| KNN       |      14 |  259.299 |  169.664 |  86.09   | 163.22  |  216.965  |  317.567  |   777.38 |
| NN        |      14 | 2138.84  | 2528.65  | 118      | 525.033 | 1536.46   | 2096.06   |  8244.39 |
| RF        |      14 |   81.4   |  112.066 |  13.31   |  36.215 |   49.65   |   75.0025 |   462.65 |
| XGB       |      14 | 1394     | 1843.13  |  45.5    | 153.642 |  403.985  | 2060.11   |  6142.8  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.000124734 |   0.000259355 |   0.000180584 |   0.908549    |   0.00192563  |
| RTree     |   0.000124734 | nan           |   4.77512e-05 |   0.206388    |   1.73823e-05 |   0.476348    |
| KNN       |   0.000259355 |   4.77512e-05 | nan           |   0.00140626  |   0.000150233 |   0.190362    |
| NN        |   0.000180584 |   0.206388    |   0.00140626  | nan           |   1.41156e-05 |   0.206388    |
| RF        |   0.908549    |   1.73823e-05 |   0.000150233 |   1.41156e-05 | nan           |   0.000439758 |
| XGB       |   0.00192563  |   0.476348    |   0.190362    |   0.206388    |   0.000439758 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/contain_mape_boxplot.png)

---

## Distance Filter

### build_time_s

#### Descriptive Statistics

|           |   count |        mean |         std |        min |          25% |          50% |          75% |        max |
|:----------|--------:|------------:|------------:|-----------:|-------------:|-------------:|-------------:|-----------:|
| Histogram |      14 |   381.857   |  1321.04    |   0.0001   |    0.750075  |    11.5      |     70       |   4970     |
| RTree     |      14 |   713.702   |  1211.85    |   1.64852  |   15.9835    |   114.331    |    699.47    |   3553.07  |
| KNN       |      14 |     1.95494 |     3.23182 |   0.004201 |    0.0327566 |     0.269027 |      2.36479 |     10.507 |
| NN        |      14 | 83940.5     | 95257.9     | 255.025    | 5239.32      | 27230.6      | 174115       | 247103     |
| RF        |      14 |   864.145   |  1303.04    |   8.10675  |   41.4437    |   186.434    |   1105.5     |   4099.01  |
| XGB       |      14 |    68.3888  |    96.0076  |   8.74224  |   11.9868    |    24.1128   |     79.9241  |    339.715 |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.0323955   |   0.0930761   |   1.69264e-05 |   0.00537516  |   0.147244    |
| RTree     |   0.0323955   | nan           |   7.05349e-05 |   0.000150233 |   0.476348    |   0.206388    |
| KNN       |   0.0930761   |   7.05349e-05 | nan           |   7.46784e-06 |   9.25216e-06 |   1.14396e-05 |
| NN        |   1.69264e-05 |   0.000150233 |   7.46784e-06 | nan           |   0.000259355 |   9.25216e-06 |
| RF        |   0.00537516  |   0.476348    |   9.25216e-06 |   0.000259355 | nan           |   0.0158543   |
| XGB       |   0.147244    |   0.206388    |   1.14396e-05 |   9.25216e-06 |   0.0158543   | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/distance_build_time_s_boxplot.png)

---

### model_size_mb

#### Descriptive Statistics

|           |   count |       mean |         std |        min |       25% |       50% |        75% |         max |
|:----------|--------:|-----------:|------------:|-----------:|----------:|----------:|-----------:|------------:|
| Histogram |      14 |   1.24235  |   0.862236  |  0.0247879 |  0.226904 |   1.57366 |   2        |    2        |
| RTree     |      14 | 426.181    | 591.538     |  2.19922   | 19.1465   | 131.639   | 599.017    | 1866.9      |
| KNN       |      14 |  89.0294   | 125.394     |  0.421808  |  3.92518  |  27.281   | 122.327    |  398.83     |
| NN        |      14 |   0.197162 |   0.0826615 |  0.0712748 |  0.112662 |   0.24264 |   0.253542 |    0.256156 |
| RF        |      14 | 533.831    | 723.902     | 18.1213    | 64.3461   | 158.833   | 715.849    | 2316.54     |
| XGB       |      14 |   0.542819 |   0.0553497 |  0.427653  |  0.499479 |   0.56514 |   0.58855  |    0.599451 |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   6.74725e-06 |   0.0029021   |   0.0222904   |   6.74725e-06 |   0.0553616   |
| RTree     |   6.74725e-06 | nan           |   0.0848817   |   7.46784e-06 |   0.448369    |   7.46784e-06 |
| KNN       |   0.0029021   |   0.0848817   | nan           |   7.46784e-06 |   0.00824201  |   0.000124734 |
| NN        |   0.0222904   |   7.46784e-06 |   7.46784e-06 | nan           |   7.46784e-06 |   7.46784e-06 |
| RF        |   6.74725e-06 |   0.448369    |   0.00824201  |   7.46784e-06 | nan           |   7.46784e-06 |
| XGB       |   0.0553616   |   7.46784e-06 |   0.000124734 |   7.46784e-06 |   7.46784e-06 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/distance_model_size_mb_boxplot.png)

---

### avg_time_ms

#### Descriptive Statistics

|           |   count |      mean |        std |        min |       25% |       50% |       75% |        max |
|:----------|--------:|----------:|-----------:|-----------:|----------:|----------:|----------:|-----------:|
| Histogram |      14 | 0.49788   | 0.314388   | 0.0541019  | 0.126631  | 0.612661  | 0.766056  |  0.79997   |
| RTree     |      14 | 3.22434   | 3.63332    | 0.0691399  | 0.246388  | 1.70414   | 5.24718   | 11.0375    |
| KNN       |      14 | 0.461421  | 0.150321   | 0.196498   | 0.413916  | 0.489004  | 0.54298   |  0.694582  |
| NN        |      14 | 0.0521614 | 0.0235733  | 0.015673   | 0.02959   | 0.0598211 | 0.0669979 |  0.0828039 |
| RF        |      14 | 0.087252  | 0.0697391  | 0.0271517  | 0.0370115 | 0.0541748 | 0.143029  |  0.222223  |
| XGB       |      14 | 0.0112661 | 0.00692544 | 0.00559463 | 0.0073128 | 0.008593  | 0.0123856 |  0.0325532 |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |         RTree |           KNN |            NN |            RF |           XGB |
|:----------|--------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| Histogram | nan           |   0.0326332   |   0.301219    |   0.000150233 |   0.000309886 |   7.46784e-06 |
| RTree     |   0.0326332   | nan           |   0.135359    |   1.41156e-05 |   0.000150233 |   7.46784e-06 |
| KNN       |   0.301219    |   0.135359    | nan           |   7.46784e-06 |   9.25216e-06 |   7.46784e-06 |
| NN        |   0.000150233 |   1.41156e-05 |   7.46784e-06 | nan           |   0.629486    |   2.13617e-05 |
| RF        |   0.000309886 |   0.000150233 |   9.25216e-06 |   0.629486    | nan           |   9.25216e-06 |
| XGB       |   7.46784e-06 |   7.46784e-06 |   7.46784e-06 |   2.13617e-05 |   9.25216e-06 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/distance_avg_time_ms_boxplot.png)

---

### mae

#### Descriptive Statistics

|           |   count |            mean |             std |     min |      25% |     50% |      75% |              max |
|:----------|--------:|----------------:|----------------:|--------:|---------:|--------:|---------:|-----------------:|
| Histogram |      14 |     6.06714e+06 |     2.22239e+07 | 1492.62 | 10416.3  | 78470.5 | 153181   |      8.32794e+07 |
| RTree     |      14 | 94818.6         | 92583.7         | 2627.87 | 23264.9  | 89495.2 | 128490   | 301392           |
| KNN       |      14 | 15637           | 17180.5         |  536.77 |  2454.22 | 11146   |  19250   |  53518           |
| NN        |      14 | 26492.1         | 33401.8         |  785.15 |  5263.15 | 11863.6 |  29903.4 | 100916           |
| RF        |      14 | 18433.9         | 20381.4         |  564.68 |  3001.53 | 12603.8 |  21775.8 |  63358.5         |
| XGB       |      14 | 43367.7         | 63186           |  441.21 |  3463.05 | 16810.3 |  45368.9 | 196456           |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |   Histogram |        RTree |          KNN |          NN |           RF |         XGB |
|:----------|------------:|-------------:|-------------:|------------:|-------------:|------------:|
| Histogram | nan         |   1          |   0.0203215  |   0.0768963 |   0.0408872  |   0.135359  |
| RTree     |   1         | nan          |   0.00408232 |   0.0203215 |   0.00543866 |   0.0695342 |
| KNN       |   0.0203215 |   0.00408232 | nan          |   0.476348  |   0.597223   |   0.448369  |
| NN        |   0.0768963 |   0.0203215  |   0.476348   | nan         |   0.565734   |   0.730389  |
| RF        |   0.0408872 |   0.00543866 |   0.597223   |   0.565734  | nan          |   0.505257  |
| XGB       |   0.135359  |   0.0695342  |   0.448369   |   0.730389  |   0.505257   | nan         |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/distance_mae_boxplot.png)

---

### mape

#### Descriptive Statistics

|           |   count |       mean |         std |      min |      25% |      50% |      75% |        max |
|:----------|--------:|-----------:|------------:|---------:|---------:|---------:|---------:|-----------:|
| Histogram |      14 | 29952.4    | 111896      |  37.8185 |  40.0039 |  41.5769 |  59.9617 | 418723     |
| RTree     |      14 |   286.202  |    245.106  |  35.9936 |  86.1258 | 185.699  | 549.448  |    659.783 |
| KNN       |      14 |    96.8379 |     97.2052 |  29.94   |  46.445  |  74.18   |  93.12   |    406.06  |
| NN        |      14 |   302.044  |    177.365  |  57.8    | 154.787  | 272.34   | 408.127  |    677.9   |
| RF        |      14 |   138.221  |    123.958  |  39.89   |  46.445  | 113.22   | 147.887  |    492.56  |
| XGB       |      14 |   303.914  |    158.916  | 102.67   | 150.8    | 285.63   | 430.45   |    550.75  |

#### Mann-Whitney U Test Results

*p-values for pairwise comparisons (p < 0.05 indicates significant difference)*

|           |     Histogram |        RTree |           KNN |            NN |           RF |           XGB |
|:----------|--------------:|-------------:|--------------:|--------------:|-------------:|--------------:|
| Histogram | nan           |   0.00261615 |   0.147798    |   0.000259355 |   0.0122746  |   0.000124734 |
| RTree     |   0.00261615  | nan          |   0.0139635   |   0.505257    |   0.123744   |   0.505257    |
| KNN       |   0.147798    |   0.0139635  | nan           |   0.000864703 |   0.260285   |   0.000180584 |
| NN        |   0.000259355 |   0.505257   |   0.000864703 | nan           |   0.0122746  |   0.872238    |
| RF        |   0.0122746   |   0.123744   |   0.260285    |   0.0122746   | nan          |   0.00408232  |
| XGB       |   0.000124734 |   0.505257   |   0.000180584 |   0.872238    |   0.00408232 | nan           |

![Boxplot](/home/adminlias/nadir/Spatial-Selectivity-Ext/analyze_results/statistics_tests/results/distance_mape_boxplot.png)

---

