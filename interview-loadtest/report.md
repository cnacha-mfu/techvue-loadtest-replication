# Load Test Results - 20250731_080411

## Test Overview

- **Total Requests:** 130
- **Concurrency Levels Tested:** 10, 20, 40, 60
- **Overall API Success Rate:** 100.0%

## API Performance Statistics

|   Concurrency |   Avg Response Time (ms) |   Std Dev (ms) |   Min (ms) |   Max (ms) |   API Success Rate (%) |
|--------------:|-------------------------:|---------------:|-----------:|-----------:|-----------------------:|
|            10 |                  2433.13 |          11.85 |    2419.11 |    2456.76 |                    100 |
|            20 |                  2609.75 |          62.6  |    2554.36 |    2743.57 |                    100 |
|            40 |                  2949.09 |         252.32 |    2181.45 |    3197.26 |                    100 |
|            60 |                  3733.24 |         904.56 |    2495.51 |    6622.4  |                    100 |

## API Performance Visualizations

### Response Time Distribution
![API Response Time Box Plot](load_test_visualizations_20250731_080411/api_response_time_boxplot.png)

### Response Time Distribution (Violin Plot)
![API Response Time Violin Plot](load_test_visualizations_20250731_080411/api_response_time_violin.png)

### Response Times by Status
![API Response Time Scatter Plot](load_test_visualizations_20250731_080411/api_response_time_scatter.png)

### Average Response Times
![Average Response Time](load_test_visualizations_20250731_080411/api_average_response_time.png)

### API Success Rates
![API Success Rate](load_test_visualizations_20250731_080411/api_success_rate.png)

### Response Time Percentiles
![API Response Time Percentiles](load_test_visualizations_20250731_080411/api_response_time_percentiles.png)


## Video Processing Statistics

|   Concurrency |   Expected Videos |   Created Videos |   Avg Success Rate (%) |   Avg Wait Time (s) |   Wait Time Std (s) |   Min Wait Time (s) |   Max Wait Time (s) |   Timeout Rate (%) |
|--------------:|------------------:|-----------------:|-----------------------:|--------------------:|--------------------:|--------------------:|--------------------:|-------------------:|
|            10 |                30 |               30 |                    100 |              144.58 |               15.28 |               128.1 |               165   |                  0 |
|            20 |                60 |               60 |                    100 |              164.2  |               28.8  |               109.1 |               207.2 |                  0 |
|            40 |               120 |              120 |                    100 |              142.83 |               12.02 |               115.5 |               164.1 |                  0 |
|            60 |               180 |              180 |                    100 |              139.89 |                9.16 |               115.1 |               155.8 |                  0 |

## Video Processing Visualizations

### Video Success Rate
![Video Success Rate](load_test_visualizations_20250731_080411/video_success_rate.png)

### Video Wait Time Distribution
![Video Wait Time Distribution](load_test_visualizations_20250731_080411/video_wait_time_boxplot.png)

### Expected vs Created Videos
![Expected vs Created Videos](load_test_visualizations_20250731_080411/video_expected_vs_created.png)

### Video Timeout Rate
![Video Timeout Rate](load_test_visualizations_20250731_080411/video_timeout_rate.png)

### API vs Video Success Rates
![Combined Success Rates](load_test_visualizations_20250731_080411/combined_success_rates.png)

### Video Wait Time vs Success Rate
![Video Wait Time vs Success Rate](load_test_visualizations_20250731_080411/video_wait_vs_success.png)

