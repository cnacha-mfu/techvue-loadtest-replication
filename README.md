# Replication Package: Architecting LLM-Powered Applications

Replication package for the performance evaluation presented in:

> **Architecting LLM-Powered Applications: A Microservices Approach for Automated Video Interviews**
> *Software: Practice and Experience* (submitted)
> Repository: [https://github.com/cnacha-mfu/techvue-loadtest-replication](https://github.com/cnacha-mfu/techvue-loadtest-replication)

## Contents

```
techvue-loadtest-replication/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── interview-loadtest/                 # Interview API load test
│   ├── load_test.py                    # Load test script (sanitized)
│   ├── visualize.py                    # Visualization generator
│   ├── results/                        # Raw CSV results
│   │   ├── load_test_detailed_*.csv    # Per-request metrics
│   │   └── load_test_summary_*.csv     # Aggregate statistics
│   ├── visualizations/                 # Generated PNG charts
│   └── report.md                       # Generated analysis report
├── resume-loadtest/                    # Resume processing load test
│   ├── load_test_resume.py             # Load test script (sanitized)
│   ├── visualize_resume.py             # Visualization generator
│   ├── results/                        # Raw CSV results (photo_url redacted)
│   │   ├── resume_load_test_detailed_*.csv
│   │   └── resume_load_test_summary_*.csv
│   ├── visualizations/                 # Generated PNG charts
│   └── report.md                       # Generated analysis report
└── deployment/
    └── cloud-run-config-example.yaml   # Example deployment configuration
```

## Prerequisites

- Python 3.10+
- Google Cloud Platform account with Cloud Run
- Firebase project with Firestore and Cloud Storage
- Google Cloud service account credentials

## Python Dependencies

```bash
pip install -r requirements.txt
```

## Reproducing Results

### Option A: Regenerate Visualizations from Raw Data (No Cloud Required)

The raw CSV result files are included in `*/results/` directories. To regenerate visualizations:

```bash
# Interview load test visualizations
cd interview-loadtest
python visualize.py

# Resume load test visualizations
cd ../resume-loadtest
python visualize_resume.py
```

### Option B: Run Load Tests Against Your Deployment

1. Deploy the system to Google Cloud Run (see `deployment/` for configuration)
2. Configure environment variables:
   ```bash
   export FIREBASE_CREDENTIAL_PATH=path/to/your-credentials.json
   export AIP_AUTHENTICATION_KEY=your-api-key
   export BASE_URL=https://your-api-service.run.app
   export JOB_ID=your-job-id
   ```
3. Run the load tests:
   ```bash
   cd interview-loadtest
   python load_test.py

   cd ../resume-loadtest
   python load_test_resume.py
   ```

## Data Dictionary

### Interview Load Test (`load_test_detailed_*.csv`)

| Column | Description |
|--------|-------------|
| `request_id` | Sequential request identifier |
| `interview_id` | Firestore document ID for the interview |
| `start_status_code` | HTTP status code for the start API call |
| `start_response_time_ms` | API response time in milliseconds |
| `total_time_ms` | End-to-end time including video processing |
| `expected_videos` | Number of expected final video files |
| `created_videos` | Number of successfully created videos |
| `video_success_rate` | Percentage of videos successfully created |
| `video_wait_time_seconds` | Time waiting for video processing to complete |
| `video_timeout` | Whether the video wait timed out |
| `missing_videos` | IDs of videos that were not created |
| `deleted_blobs` | Number of Cloud Storage objects cleaned up |
| `error` | Error message if the request failed |

### Resume Load Test (`resume_load_test_detailed_*.csv`)

| Column | Description |
|--------|-------------|
| `request_id` | Sequential request identifier |
| `applicant_id` | Firestore document ID for the applicant |
| `file_name` | Resume PDF file used |
| `bytes` | File size in bytes |
| `create_status` | HTTP status for applicant creation |
| `create_ms` | Applicant creation time (ms) |
| `upload_status` | HTTP status for resume upload |
| `upload_ms` | Resume upload time (ms) |
| `extract_status` | HTTP status for photo extraction |
| `extract_ms` | Photo extraction time (ms) |
| `analyze_status` | HTTP status for LLM analysis |
| `analyze_ms` | LLM analysis time (ms) |
| `photo_url` | [REDACTED] - Firebase Storage URL |
| `ok` | Overall success flag |
| `stage` | Last completed pipeline stage |
| `error` | Error message if failed |

### Summary Files (`*_summary_*.csv`)

Aggregate statistics including mean/min/max response times, success rates, and video processing metrics.

## Concurrency Levels Tested

### Interview Load Test
- 10, 20, 40, 60 concurrent users
- Date: 2025-07-31

### Resume Load Test
- 50, 100, 150, 200, 250, 300 concurrent requests
- Date: 2025-11-12

## Privacy and Anonymization

- Firebase credential files are excluded
- API URLs and keys are replaced with placeholders
- `photo_url` columns in resume CSVs are redacted (`[REDACTED]`)
- Firestore document IDs (interview_id, applicant_id) are retained as they are not personally identifiable

## License

MIT License - see LICENSE file.
