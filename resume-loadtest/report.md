# Resume Load Test Results - 20251112_120844

    - **Total Requests:** 1050
    - **Concurrency Levels Tested:** 50, 100, 150, 200, 250, 300
    - **Overall Analyze Success Rate:** 53.7%

    ## Create API

|   Concurrency |   Avg Create API (ms) |   Std (ms) |   Min (ms) |   Max (ms) |   Create API Success Rate (%) |
|--------------:|----------------------:|-----------:|-----------:|-----------:|------------------------------:|
|            50 |                528.34 |      75.41 |        455 |        918 |                           100 |
|           100 |                653.77 |     104.81 |        381 |        802 |                           100 |
|           150 |                765.26 |     153.08 |        412 |       1010 |                           100 |
|           200 |               1120.16 |     260.67 |        544 |       1572 |                           100 |
|           250 |                974.9  |     170.51 |        455 |       1230 |                           100 |
|           300 |                987.54 |     164.33 |        532 |       1468 |                           100 |

### Create API Visuals
![Create API Box](resume_load_test_visualizations_20251112_120844/create_api_time_box.png)
![Create API Violin](resume_load_test_visualizations_20251112_120844/create_api_time_violin.png)
![Create API Scatter](resume_load_test_visualizations_20251112_120844/create_api_time_scatter.png)
![Create API Avg](resume_load_test_visualizations_20251112_120844/create_api_time_avg.png)
![Create API Percentiles](resume_load_test_visualizations_20251112_120844/create_api_time_percentiles.png)
![Create API Success Rate](resume_load_test_visualizations_20251112_120844/create_api_success_rate.png)

## Upload

|   Concurrency |   Avg Upload (ms) |   Std (ms) |   Min (ms) |   Max (ms) |   Upload Success Rate (%) |
|--------------:|------------------:|-----------:|-----------:|-----------:|--------------------------:|
|            50 |           1610.38 |     531.44 |        789 |       2668 |                       100 |
|           100 |           2037.74 |     628.27 |        720 |       3175 |                       100 |
|           150 |           2366.57 |     673.97 |        742 |       3905 |                       100 |
|           200 |           3124.83 |    1374.17 |        737 |       6465 |                       100 |
|           250 |           3195.71 |    1124.75 |        759 |       5016 |                       100 |
|           300 |           4666.11 |    1846.55 |        783 |       8081 |                       100 |

### Upload Visuals
![Upload Box](resume_load_test_visualizations_20251112_120844/upload_time_box.png)
![Upload Violin](resume_load_test_visualizations_20251112_120844/upload_time_violin.png)
![Upload Scatter](resume_load_test_visualizations_20251112_120844/upload_time_scatter.png)
![Upload Avg](resume_load_test_visualizations_20251112_120844/upload_time_avg.png)
![Upload Percentiles](resume_load_test_visualizations_20251112_120844/upload_time_percentiles.png)
![Upload Success Rate](resume_load_test_visualizations_20251112_120844/upload_success_rate.png)

## Extract Photo

|   Concurrency |   Avg Extract Photo (ms) |   Std (ms) |   Min (ms) |   Max (ms) |   Extract Photo Success Rate (%) |
|--------------:|-------------------------:|-----------:|-----------:|-----------:|---------------------------------:|
|            50 |                   694.92 |     195.09 |        520 |       1494 |                              100 |
|           100 |                   979.97 |     367.5  |        455 |       1574 |                              100 |
|           150 |                  1494.01 |     715.75 |        439 |       2868 |                              100 |
|           200 |                  1668.09 |     664.27 |        695 |       3191 |                              100 |
|           250 |                  1750.05 |     936.19 |        496 |       3562 |                              100 |
|           300 |                  2812.81 |    1324.45 |        507 |       5092 |                              100 |

### Extract Photo Visuals
![Extract Photo Box](resume_load_test_visualizations_20251112_120844/extract_photo_time_box.png)
![Extract Photo Violin](resume_load_test_visualizations_20251112_120844/extract_photo_time_violin.png)
![Extract Photo Scatter](resume_load_test_visualizations_20251112_120844/extract_photo_time_scatter.png)
![Extract Photo Avg](resume_load_test_visualizations_20251112_120844/extract_photo_time_avg.png)
![Extract Photo Percentiles](resume_load_test_visualizations_20251112_120844/extract_photo_time_percentiles.png)
![Extract Photo Success Rate](resume_load_test_visualizations_20251112_120844/extract_photo_success_rate.png)

## Analyze

|   Concurrency |   Avg Analyze (ms) |   Std (ms) |   Min (ms) |   Max (ms) |   Analyze Success Rate (%) |
|--------------:|-------------------:|-----------:|-----------:|-----------:|---------------------------:|
|            50 |            44750.1 |    14841.3 |      25137 |      83907 |                   100      |
|           100 |            43499.2 |    12353.6 |      26304 |      81325 |                   100      |
|           150 |            56618.6 |    16826.9 |      32927 |     106755 |                    96.6667 |
|           200 |            36331.6 |    14213.9 |      19585 |      82710 |                    25      |
|           250 |            30726.7 |    19983.8 |       9826 |      84540 |                    55.6    |
|           300 |            30015.4 |    16815.8 |      13230 |      79989 |                    26.6667 |

### Analyze Visuals
![Analyze Box](resume_load_test_visualizations_20251112_120844/analyze_time_box.png)
![Analyze Violin](resume_load_test_visualizations_20251112_120844/analyze_time_violin.png)
![Analyze Scatter](resume_load_test_visualizations_20251112_120844/analyze_time_scatter.png)
![Analyze Avg](resume_load_test_visualizations_20251112_120844/analyze_time_avg.png)
![Analyze Percentiles](resume_load_test_visualizations_20251112_120844/analyze_time_percentiles.png)
![Analyze Success Rate](resume_load_test_visualizations_20251112_120844/analyze_success_rate.png)

    