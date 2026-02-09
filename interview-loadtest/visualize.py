import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_latest_test_files():
    """Find the most recent detailed and summary CSV files from load tests."""
    detailed_files = glob.glob("load_test_detailed_*.csv")
    if not detailed_files:
        raise FileNotFoundError("No detailed test files found")
    summary_files = glob.glob("load_test_summary_*.csv")
    if not summary_files:
        raise FileNotFoundError("No summary test files found")
    detailed_files.sort(reverse=True)
    summary_files.sort(reverse=True)
    return detailed_files, summary_files

def load_test_data(detailed_files):
    """Load and combine data from detailed test files."""
    all_data = []
    for file in detailed_files:
        concurrency = int(file.split('_')[3])  # load_test_detailed_<concurrency>_<timestamp>.csv
        df = pd.read_csv(file)
        df['Concurrency'] = concurrency
        if 'video_success_rate' in df.columns:
            df['video_success_rate_numeric'] = df['video_success_rate'].str.rstrip('%').astype('float')
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def create_api_visualizations(df, viz_dir):
    """Create visualizations for API performance metrics."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Concurrency', y='start_response_time_ms', data=df)
    plt.title('API Response Time Distribution by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Response Time (ms)')
    plt.savefig(f"{viz_dir}/api_response_time_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Concurrency', y='start_response_time_ms', data=df)
    plt.title('API Response Time Distribution (Violin Plot)')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Response Time (ms)')
    plt.savefig(f"{viz_dir}/api_response_time_violin.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    success_color = df['start_status_code'] == 200
    plt.scatter(df['Concurrency'], df['start_response_time_ms'],
               c=success_color, cmap='RdYlGn', alpha=0.7)
    plt.title('API Response Times by Concurrency Level and Status')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Response Time (ms)')
    plt.colorbar(label='Success (Green=200, Red=Error)')
    plt.savefig(f"{viz_dir}/api_response_time_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    avg_times = df.groupby('Concurrency')['start_response_time_ms'].agg(['mean', 'std']).reset_index()
    plt.figure(figsize=(12, 6))
    plt.errorbar(
        avg_times['Concurrency'],
        avg_times['mean'],
        yerr=avg_times['std'],
        fmt='o-',
        capsize=5,
        linewidth=2,
        markersize=8
    )
    plt.title('Average API Response Time by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Response Time (ms)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{viz_dir}/api_average_response_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    success_rates = df.groupby('Concurrency').apply(
        lambda x: (x['start_status_code'] == 200).mean() * 100
    ).reset_index()
    success_rates.columns = ['Concurrency', 'Success Rate (%)']

    plt.figure(figsize=(12, 6))
    bars = plt.bar(success_rates['Concurrency'], success_rates['Success Rate (%)'],
                   color='skyblue', edgecolor='navy', alpha=0.7)
    plt.title('API Success Rate by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.savefig(f"{viz_dir}/api_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    percentiles = df.groupby('Concurrency')['start_response_time_ms'].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).unstack()
    plt.figure(figsize=(12, 6))
    percentiles.plot(marker='o', linewidth=2, markersize=6)
    plt.title('API Response Time Percentiles by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Response Time (ms)')
    plt.legend(title='Percentile', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{viz_dir}/api_response_time_percentiles.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_video_visualizations(df, viz_dir):
    """Create visualizations for video processing metrics."""
    video_cols = ['expected_videos', 'created_videos', 'video_success_rate_numeric', 'video_wait_time_seconds', 'video_timeout']
    has_video_data = all(col in df.columns for col in video_cols)
    if not has_video_data:
        logger.warning("Video statistics columns not found. Skipping video visualizations.")
        return
    video_df = df.dropna(subset=video_cols)
    if video_df.empty:
        logger.warning("No video data available. Skipping video visualizations.")
        return

    plt.figure(figsize=(12, 6))
    video_success = video_df.groupby('Concurrency')['video_success_rate_numeric'].mean().reset_index()
    bars = plt.bar(video_success['Concurrency'], video_success['video_success_rate_numeric'],
                   color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    plt.title('Video Creation Success Rate by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Video Success Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    plt.savefig(f"{viz_dir}/video_success_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Concurrency', y='video_wait_time_seconds', data=video_df)
    plt.title('Video Processing Wait Time Distribution by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Wait Time (seconds)')
    plt.savefig(f"{viz_dir}/video_wait_time_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    concurrency_levels = video_df['Concurrency'].unique()
    expected_by_concurrency = video_df.groupby('Concurrency')['expected_videos'].sum()
    created_by_concurrency = video_df.groupby('Concurrency')['created_videos'].sum()
    x = np.arange(len(concurrency_levels))
    width = 0.35
    bars1 = ax.bar(x - width/2, expected_by_concurrency, width, label='Expected', alpha=0.7, color='lightblue')
    bars2 = ax.bar(x + width/2, created_by_concurrency, width, label='Created', alpha=0.7, color='orange')
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Number of Videos')
    ax.set_title('Expected vs Created Videos by Concurrency Level')
    ax.set_xticks(x)
    ax.set_xticklabels(concurrency_levels)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    plt.savefig(f"{viz_dir}/video_expected_vs_created.png", dpi=300, bbox_inches='tight')
    plt.close()

    timeout_rates = video_df.groupby('Concurrency')['video_timeout'].apply(
        lambda x: (x == True).mean() * 100
    ).reset_index()
    timeout_rates.columns = ['Concurrency', 'Timeout Rate (%)']
    plt.figure(figsize=(12, 6))
    bars = plt.bar(timeout_rates['Concurrency'], timeout_rates['Timeout Rate (%)'],
                   color='salmon', edgecolor='darkred', alpha=0.7)
    plt.title('Video Processing Timeout Rate by Concurrency Level')
    plt.xlabel('Concurrency Level')
    plt.ylabel('Timeout Rate (%)')
    plt.ylim(0, 100)
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    plt.savefig(f"{viz_dir}/video_timeout_rate.png", dpi=300, bbox_inches='tight')
    plt.close()

    api_success = df.groupby('Concurrency').apply(
        lambda x: (x['start_status_code'] == 200).mean() * 100
    ).reset_index()
    api_success.columns = ['Concurrency', 'API Success Rate (%)']
    video_success = video_df.groupby('Concurrency')['video_success_rate_numeric'].mean().reset_index()
    video_success.columns = ['Concurrency', 'Video Success Rate (%)']
    combined = pd.merge(api_success, video_success, on='Concurrency')
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(combined))
    width = 0.35
    bars1 = ax.bar(x - width/2, combined['API Success Rate (%)'], width,
                   label='API Success Rate', alpha=0.7, color='skyblue')
    bars2 = ax.bar(x + width/2, combined['Video Success Rate (%)'], width,
                   label='Video Success Rate', alpha=0.7, color='lightgreen')
    ax.set_xlabel('Concurrency Level')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('API vs Video Success Rates by Concurrency Level')
    ax.set_xticks(x)
    ax.set_xticklabels(combined['Concurrency'])
    ax.legend()
    ax.set_ylim(0, 100)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
    plt.savefig(f"{viz_dir}/combined_success_rates.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(video_df['video_wait_time_seconds'], video_df['video_success_rate_numeric'],
                         c=video_df['Concurrency'], cmap='viridis', alpha=0.7, s=60)
    plt.colorbar(scatter, label='Concurrency Level')
    plt.xlabel('Video Wait Time (seconds)')
    plt.ylabel('Video Success Rate (%)')
    plt.title('Video Wait Time vs Success Rate')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(f"{viz_dir}/video_wait_vs_success.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(df, timestamp):
    """Create all visualizations from the test data."""
    viz_dir = f"load_test_visualizations_{timestamp}"
    os.makedirs(viz_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    logger.info("Creating API performance visualizations...")
    create_api_visualizations(df, viz_dir)
    logger.info("Creating video processing visualizations...")
    create_video_visualizations(df, viz_dir)
    return viz_dir

def generate_md_report(viz_dir, df, timestamp):
    """Generate a Markdown report with all visualizations and statistics."""
    # API summary statistics
    api_summary_stats = df.groupby('Concurrency').agg({
        'start_response_time_ms': ['mean', 'std', 'min', 'max'],
        'start_status_code': lambda x: (x == 200).mean() * 100
    }).round(2)
    api_summary_stats.columns = ['Avg Response Time (ms)', 'Std Dev (ms)', 'Min (ms)', 'Max (ms)', 'API Success Rate (%)']

    # Convert API stats to markdown table
    api_md_table = api_summary_stats.reset_index().to_markdown(index=False)

    # Video summary statistics if available
    video_cols = ['expected_videos', 'created_videos', 'video_success_rate_numeric', 'video_wait_time_seconds', 'video_timeout']
    has_video_data = all(col in df.columns for col in video_cols)
    video_md_table = ""
    video_section = ""
    if has_video_data:
        video_df = df.dropna(subset=video_cols)
        if not video_df.empty:
            video_summary_stats = video_df.groupby('Concurrency').agg({
                'expected_videos': 'sum',
                'created_videos': 'sum',
                'video_success_rate_numeric': 'mean',
                'video_wait_time_seconds': ['mean', 'std', 'min', 'max'],
                'video_timeout': lambda x: (x == True).mean() * 100
            }).round(2)
            video_summary_stats.columns = ['Expected Videos', 'Created Videos', 'Avg Success Rate (%)',
                                           'Avg Wait Time (s)', 'Wait Time Std (s)', 'Min Wait Time (s)',
                                           'Max Wait Time (s)', 'Timeout Rate (%)']
            video_md_table = video_summary_stats.reset_index().to_markdown(index=False)
            video_section = f"""
## Video Processing Statistics

{video_md_table}

## Video Processing Visualizations

### Video Success Rate
![Video Success Rate]({os.path.join(viz_dir, "video_success_rate.png")})

### Video Wait Time Distribution
![Video Wait Time Distribution]({os.path.join(viz_dir, "video_wait_time_boxplot.png")})

### Expected vs Created Videos
![Expected vs Created Videos]({os.path.join(viz_dir, "video_expected_vs_created.png")})

### Video Timeout Rate
![Video Timeout Rate]({os.path.join(viz_dir, "video_timeout_rate.png")})

### API vs Video Success Rates
![Combined Success Rates]({os.path.join(viz_dir, "combined_success_rates.png")})

### Video Wait Time vs Success Rate
![Video Wait Time vs Success Rate]({os.path.join(viz_dir, "video_wait_vs_success.png")})
"""

    # Markdown report content
    md_report = f"""# Load Test Results - {timestamp}

## Test Overview

- **Total Requests:** {len(df)}
- **Concurrency Levels Tested:** {', '.join(map(str, sorted(df['Concurrency'].unique())))}
- **Overall API Success Rate:** {(df['start_status_code'] == 200).mean() * 100:.1f}%

## API Performance Statistics

{api_md_table}

## API Performance Visualizations

### Response Time Distribution
![API Response Time Box Plot]({os.path.join(viz_dir, "api_response_time_boxplot.png")})

### Response Time Distribution (Violin Plot)
![API Response Time Violin Plot]({os.path.join(viz_dir, "api_response_time_violin.png")})

### Response Times by Status
![API Response Time Scatter Plot]({os.path.join(viz_dir, "api_response_time_scatter.png")})

### Average Response Times
![Average Response Time]({os.path.join(viz_dir, "api_average_response_time.png")})

### API Success Rates
![API Success Rate]({os.path.join(viz_dir, "api_success_rate.png")})

### Response Time Percentiles
![API Response Time Percentiles]({os.path.join(viz_dir, "api_response_time_percentiles.png")})

{video_section}
"""

    with open(f"report.md", 'w') as f:
        f.write(md_report)

def main():
    """Main function to process load test results and generate visualizations."""
    try:
        detailed_files, summary_files = find_latest_test_files()
        logger.info(f"Found {len(detailed_files)} detailed files and {len(summary_files)} summary files")
        df = load_test_data(detailed_files)
        logger.info(f"Loaded data for {len(df)} requests across {df['Concurrency'].nunique()} concurrency levels")
        video_cols = ['expected_videos', 'created_videos', 'video_success_rate', 'video_wait_time_seconds', 'video_timeout']
        has_video_data = any(col in df.columns for col in video_cols)
        if has_video_data:
            logger.info("Video processing statistics detected in data")
        else:
            logger.info("No video processing statistics found - will generate API-only visualizations")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = create_visualizations(df, timestamp)
        generate_md_report(viz_dir, df, timestamp)
        logger.info(f"Visualizations and report generated in {viz_dir}/")
        logger.info(f"Open {viz_dir}/report.md to view the results")
        print("\n" + "="*50)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*50)
        print(f"Total requests: {len(df)}")
        print(f"Concurrency levels: {sorted(df['Concurrency'].unique())}")
        print(f"Overall API success rate: {(df['start_status_code'] == 200).mean() * 100:.1f}%")
        if has_video_data:
            video_df = df.dropna(subset=['video_success_rate_numeric'])
            if not video_df.empty:
                print(f"Overall video success rate: {video_df['video_success_rate_numeric'].mean():.1f}%")
                print(f"Average video wait time: {video_df['video_wait_time_seconds'].mean():.1f} seconds")
        print(f"\nDetailed report: {viz_dir}/report.md")
        print("="*50)
    except Exception as e:
        logger.error(f"Error processing load test results: {e}")
        raise

if __name__ == "__main__":
    main()