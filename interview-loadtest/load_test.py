import asyncio
import json
import time
import aiohttp
from typing import List, Dict, Any
import logging
import os
import csv
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore, storage
import requests
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIAL_PATH", "YOUR_FIREBASE_CREDENTIAL.json"))
    firebase_admin.initialize_app(cred, {
        "storageBucket": "your-project.firebasestorage.app"
    })

db = firestore.client()
stg = storage.bucket()

VALID_AUTH_KEY = os.getenv("AIP_AUTHENTICATION_KEY")

# Configuration
BASE_URL = "https://your-api-service.run.app"
JOB_ID = "YOUR_JOB_ID"
NUM_REQUESTS = [5]

# Headers for all requests
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": VALID_AUTH_KEY
}

# Thread-safe storage for created interview IDs
created_interview_ids = []
interview_ids_lock = threading.Lock()

def delete_interview_from_firebase(interview_id: str) -> bool:
    """Delete an interview document and its subcollections from Firebase."""
    try:
        interview_ref = db.collection("interviews").document(interview_id)
        
        dialogs_ref = interview_ref.collection("dialogs")
        existing_dialogs = dialogs_ref.stream()
        for dialog in existing_dialogs:
            chunks_ref = dialog.reference.collection("chunks")
            existing_chunks = chunks_ref.stream()
            for chunk in existing_chunks:
                chunk.reference.delete()
            dialog.reference.delete()

        interview_ref.delete()
        
        logger.info(f"Successfully deleted interview {interview_id} and its dialogs")
        return True
    except Exception as e:
        logger.error(f"Error deleting interview {interview_id}: {e}")
        return False

def delete_interview_storage_folder(interview_id: str) -> bool:
    """Delete storage folder for interview."""
    try:
        bucket = stg
        folder_path = f"interviews/{interview_id}/"
        blobs = bucket.list_blobs(prefix=folder_path)
        
        deleted_count = 0
        for blob in blobs:
            try:
                blob.delete()
                deleted_count += 1
                logger.debug(f"Deleted file: {blob.name}")
            except Exception as e:
                logger.warning(f"Failed to delete file {blob.name}: {str(e)}")
        
        if deleted_count > 0:
            logger.info(f"Successfully deleted {deleted_count} files from {folder_path}")
        else:
            logger.info(f"No files found in {folder_path} or folder already empty")
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete storage folder for interview {interview_id}: {str(e)}")
        return False

async def delete_interviews() -> None:
    """Delete all interviews created during the test using Firebase Admin SDK."""
    with interview_ids_lock:
        ids_to_delete = created_interview_ids.copy()
    
    if not ids_to_delete:
        logger.info("No interviews to delete")
        return
    
    logger.info(f"Cleaning up {len(ids_to_delete)} created interviews...")
    
    # Use thread pool for concurrent deletion
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=10) as executor:
        delete_tasks = [
            loop.run_in_executor(executor, delete_interview_from_firebase, interview_id)
            for interview_id in ids_to_delete
        ]
        results = await asyncio.gather(*delete_tasks, return_exceptions=True)
    
    successful_deletes = sum(1 for result in results if result is True)
    logger.info(f"Successfully deleted {successful_deletes}/{len(ids_to_delete)} interviews")

def send_invite_sync(request_id: int) -> Dict[str, Any]:
    """Send a single invite request synchronously."""
    payload = {
        "email": f"test{request_id}@example.com",
        "job_id": JOB_ID
    }
    
    start_time = time.time()
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            response = requests.post(
                f"{BASE_URL}/invite",
                headers=HEADERS,
                json=payload,
                timeout=30  # Add timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            response_data = response.json()
            
            if response.status_code == 200 and "interview_id" in response_data:
                interview_id = response_data["interview_id"]
                logger.info(f"Successfully invited interview {interview_id}")
                
                # Thread-safe append
                with interview_ids_lock:
                    created_interview_ids.append(interview_id)
                
                return {
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "response_time_ms": response_time,
                    "interview_id": interview_id
                }
            else:
                logger.warning(f"Invite request failed with status {response.status_code}, attempt {retry_count + 1}/{max_retries}")
                retry_count += 1
                
        except Exception as e:
            logger.error(f"Error in invite request attempt {retry_count + 1}: {e}")
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(1)  # Brief delay before retry
    
    return {
        "request_id": request_id,
        "status_code": None,
        "response_time_ms": (time.time() - start_time) * 1000,
        "error": f"Failed after {max_retries} attempts"
    }

async def upload_data_files_to_gcs_async(interview_id: str) -> None:
    """
    Async wrapper for uploading files to GCS.
    Runs the blocking operation in a thread pool.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        await loop.run_in_executor(executor, upload_data_files_to_gcs_sync, interview_id)

def upload_data_files_to_gcs_sync(interview_id: str):
    """
    Upload files in data/<interview_id>/<dialog_id>/ to GCS bucket.
    This is the blocking version that will be run in a thread pool.
    """
    try:
        bucket = stg

        # Query dialogs for this interview, ordered by sequence_no
        dialogs_ref = db.collection("interviews").document(interview_id).collection("dialogs")
        dialogs_query = dialogs_ref.order_by("sequence_no")
        dialogs = list(dialogs_query.stream())

        if not dialogs:
            logger.warning(f"No dialogs found in Firestore for interview {interview_id}")
            return

        data_root = os.path.join("../data", "interview")
        if not os.path.isdir(data_root):
            logger.warning(f"Interview folder not found: {data_root}")
            return
        
        video_num = 1
        for idx, dialog in enumerate(dialogs):
            dialog_id = dialog.id
            dialog_path = os.path.join(data_root, "video_"+str(video_num))
            
            if not os.path.isdir(dialog_path):
                logger.warning(f"Dialog folder not found: {dialog_path}")
                continue

            files = os.listdir(dialog_path)
            ordered_files = []
            if "question.mp3" in files:
                ordered_files.append("question.mp3")
            
            question_files = sorted([f for f in files if f.startswith("question_") and f != "question.mp3"])
            ordered_files.extend(question_files)
            
            chunk_files = sorted([f for f in files if f.startswith("chunk_")])
            ordered_files.extend(chunk_files)
            
            other_files = [f for f in files if f not in ordered_files]
            ordered_files.extend(other_files)

            # Find max chunk and question numbers
            chunk_numbers = []
            for f in chunk_files:
                try:
                    chunk_num = int(f.split("_")[1].split(".")[0])
                    chunk_numbers.append(chunk_num)
                except (IndexError, ValueError):
                    continue
            
            max_chunk_number = max(chunk_numbers) if chunk_numbers else 0
            
            question_numbers = []
            for f in question_files:
                try:
                    question_num = int(f.split("_")[1].split(".")[0])
                    question_numbers.append(question_num)
                except (IndexError, ValueError):
                    continue
            
            max_question_number = max(question_numbers) if question_numbers else 0

            for filename in ordered_files:
                file_path = os.path.join(dialog_path, filename)
                if not os.path.isfile(file_path):
                    continue
                
                needs_metadata = (filename.startswith("question_") or filename.startswith("chunk_")) and filename.endswith(".mp4")
                logger.info(f"Uploading file {filename} for interview {interview_id}, dialog {dialog_id}")
                try:
                    blob_path = f"interviews/{interview_id}/video_{dialog_id}/"
                    blob = bucket.blob(blob_path + filename)
                    
                    if needs_metadata:
                        chunk_index = 0
                        is_final = False
                        
                        if filename.startswith("chunk_"):
                            try:
                                chunk_index = int(filename.split("_")[1].split(".")[0])
                                is_final = (chunk_index == max_chunk_number)
                            except (IndexError, ValueError):
                                chunk_index = 0
                        elif filename.startswith("question_"):
                            try:
                                chunk_index = int(filename.split("_")[1].split(".")[0])
                                is_final = (chunk_index == max_question_number)
                            except (IndexError, ValueError):
                                chunk_index = 1
                        
                        file_type = "question" if filename.startswith("question_") else "chunk"
                        
                        metadata = {
                            'interviewId': interview_id,
                            'language': 'en',
                            'isFinal': str(is_final).lower(),
                            'dialogId': dialog_id,
                            'type': file_type,
                            'chunkIndex': str(chunk_index),
                            'timestamp': str(int(time.time() * 1000))
                        }
                        
                        blob.metadata = metadata
                        blob.upload_from_filename(file_path)
                        logger.debug(f"Uploaded to gs://{bucket.name}/{blob_path} with metadata")
                    else:
                        blob.upload_from_filename(file_path)
                        logger.debug(f"Uploaded to gs://{bucket.name}/{blob_path}")

                except Exception as e:
                    logger.error(f"Failed to upload {file_path}: {e}")
                    
            video_num = video_num + 1
            if video_num > 3:
                video_num = 1
                
        logger.info(f"Completed file upload for interview {interview_id}")
        
    except Exception as e:
        logger.error(f"Error in upload_data_files_to_gcs_sync for {interview_id}: {e}")

async def wait_for_final_videos_async(interview_id: str) -> Dict[str, Any]:
    """
    Async version of waiting for final video files.
    Uses asyncio.sleep and runs storage checks in thread pool.
    """
    loop = asyncio.get_event_loop()
    
    # Get dialog IDs in thread pool
    with ThreadPoolExecutor(max_workers=1) as executor:
        dialog_ids = await loop.run_in_executor(executor, get_dialog_ids, interview_id)
    
    expected_videos = len(dialog_ids)
    timeout = 300  # 5 minutes timeout
    interval = 5   # Check every 5 seconds
    start_wait_time = time.time()
    
    final_video_results = {
        'expected_videos': expected_videos,
        'created_videos': 0,
        'wait_time_seconds': 0,
        'success_rate': 0.0,
        'timed_out': False,
        'video_creation_times': [],
        'missing_videos': []
    }
    
    videos_found = set()
    elapsed_time = 0
    
    logger.info(f"Waiting for {expected_videos} final videos for interview {interview_id}")
    
    while len(videos_found) < expected_videos and elapsed_time < timeout:
        logger.debug(f"Checking for final videos (elapsed: {elapsed_time}s, found: {len(videos_found)}/{expected_videos})")
        
        # Check for videos in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=5) as executor:
            check_tasks = []
            for dialog_id in dialog_ids:
                if dialog_id not in videos_found:
                    task = loop.run_in_executor(executor, check_video_exists, interview_id, dialog_id)
                    check_tasks.append((dialog_id, task))
            
            # Wait for all checks to complete
            for dialog_id, task in check_tasks:
                try:
                    exists = await task
                    if exists and dialog_id not in videos_found:
                        videos_found.add(dialog_id)
                        video_found_time = time.time() - start_wait_time
                        final_video_results['video_creation_times'].append({
                            'dialog_id': dialog_id,
                            'found_at_seconds': video_found_time
                        })
                        logger.info(f"Final video found for dialog {dialog_id} at {video_found_time:.1f}s")
                except Exception as e:
                    logger.error(f"Error checking video for dialog {dialog_id}: {e}")
        
        if len(videos_found) < expected_videos:
            await asyncio.sleep(interval)
            elapsed_time += interval
    
    # Calculate final statistics
    total_wait_time = time.time() - start_wait_time
    final_video_results['wait_time_seconds'] = total_wait_time
    final_video_results['created_videos'] = len(videos_found)
    final_video_results['success_rate'] = len(videos_found) / expected_videos if expected_videos > 0 else 0.0
    final_video_results['timed_out'] = elapsed_time >= timeout
    
    # Track missing videos
    for dialog_id in dialog_ids:
        if dialog_id not in videos_found:
            final_video_results['missing_videos'].append(dialog_id)
    
    if final_video_results['timed_out']:
        logger.warning(f"Timeout waiting for final videos. Found {len(videos_found)}/{expected_videos} videos in {total_wait_time:.1f}s")
    else:
        logger.info(f"All final videos found in {total_wait_time:.1f}s (success rate: {final_video_results['success_rate']:.2%})")
    
    return final_video_results

def get_dialog_ids(interview_id: str) -> List[str]:
    """Get dialog IDs for an interview (blocking operation)."""
    try:
        dialogs_ref = db.collection("interviews").document(interview_id).collection("dialogs")
        dialog_ids = []
        for dialog in dialogs_ref.stream():
            dialog_ids.append(dialog.id)
        return dialog_ids
    except Exception as e:
        logger.error(f"Error getting dialog IDs for {interview_id}: {e}")
        return []

def check_video_exists(interview_id: str, dialog_id: str) -> bool:
    """Check if a final video exists (blocking operation)."""
    try:
        final_video_path = f"interviews/{interview_id}/final_{dialog_id}.mp4"
        blob = stg.blob(final_video_path)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking video existence for {interview_id}/{dialog_id}: {e}")
        return False

async def cleanup_storage_async(interview_id: str) -> int:
    """Async version of storage cleanup."""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, cleanup_storage_sync, interview_id)

def cleanup_storage_sync(interview_id: str) -> int:
    """Cleanup storage synchronously (blocking operation)."""
    try:
        blobs_to_delete = stg.list_blobs(prefix=f"interviews/{interview_id}/")
        deleted_blobs_count = 0
        
        for blob in blobs_to_delete:
            try:
                blob.delete()
                deleted_blobs_count += 1
                logger.debug(f"Deleted blob: {blob.name}")
            except Exception as e:
                logger.error(f"Error deleting blob {blob.name}: {e}")
        
        logger.info(f"Deleted {deleted_blobs_count} blobs from interviews/{interview_id}/")
        return deleted_blobs_count
        
    except Exception as e:
        logger.error(f"Error in storage cleanup for {interview_id}: {e}")
        return 0

async def send_start_request(session: aiohttp.ClientSession, interview_id: str, request_id: int) -> Dict[str, Any]:
    """Send a single start interview request with true async operations."""
    start_time = time.time()
    
    try:
        # Send the start request
        async with session.get(
            f"{BASE_URL}/start/interviews/{interview_id}",
            headers=HEADERS,
            timeout=aiohttp.ClientTimeout(total=60)  # Add timeout
        ) as response:
            api_end_time = time.time()
            response_time = (api_end_time - start_time) * 1000
            
            response_data = await response.json() if response.headers.get('content-type') == 'application/json' else None

            # Run file upload, video waiting, and cleanup concurrently where possible
            upload_task = upload_data_files_to_gcs_async(interview_id)
            
            # Wait for upload to complete before starting video check
            await upload_task
            
            # Wait for final videos
            final_video_stats = await wait_for_final_videos_async(interview_id)
            
            # Cleanup storage
            deleted_count = await cleanup_storage_async(interview_id)

            total_end_time = time.time()
            total_time = (total_end_time - start_time) * 1000

            return {
                "request_id": request_id,
                "interview_id": interview_id,
                "status_code": response.status,
                "response_time_ms": response_time,
                "total_time_ms": total_time,
                "response": response_data,
                "final_video_stats": final_video_stats,
                "deleted_blobs": deleted_count
            }
    except Exception as e:
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        return {
            "request_id": request_id,
            "interview_id": interview_id,
            "status_code": None,
            "response_time_ms": None,
            "total_time_ms": total_time,
            "error": str(e),
            "final_video_stats": None
        }

async def create_interviews_concurrently(concurrent_requests: int) -> List[Dict[str, Any]]:
    """Create interviews concurrently using thread pool."""
    loop = asyncio.get_event_loop()
    
    # Use thread pool for concurrent invite requests
    with ThreadPoolExecutor(max_workers=min(concurrent_requests, 20)) as executor:
        invite_tasks = [
            loop.run_in_executor(executor, send_invite_sync, i)
            for i in range(concurrent_requests)
        ]
        
        logger.info(f"Creating {concurrent_requests} interviews concurrently...")
        invite_results = await asyncio.gather(*invite_tasks)
    
    return invite_results

async def run_load_test(concurrent_requests: int) -> List[Dict[str, Any]]:
    """Run a load test with the specified number of concurrent requests."""
    # Create interviews concurrently
    invite_results = await create_interviews_concurrently(concurrent_requests)
    
    # Filter successful invites
    successful_interviews = [r for r in invite_results if r.get("status_code") == 200]
    if not successful_interviews:
        logger.error("No interviews were successfully created. Aborting load test.")
        return []
    
    logger.info(f"Successfully created {len(successful_interviews)} interviews")
    
    # Configure session with better settings for concurrency
    timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 minute total, 30s connect
    connector = aiohttp.TCPConnector(
        limit=100,  # Total connection pool size
        limit_per_host=50,  # Connections per host
        ttl_dns_cache=300,  # DNS cache TTL
        use_dns_cache=True,
    )
    
    # Now run the load test for starting interviews
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [
            send_start_request(session, result["interview_id"], i)
            for i, result in enumerate(successful_interviews)
        ]
        
        logger.info(f"Starting load test with {len(tasks)} concurrent start requests...")
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                processed_results.append({
                    "request_id": i,
                    "interview_id": successful_interviews[i]["interview_id"] if i < len(successful_interviews) else "unknown",
                    "status_code": None,
                    "error": str(result),
                    "final_video_stats": None
                })
            else:
                processed_results.append(result)
        
        total_time = end_time - start_time
        
        # Calculate statistics
        api_times = [r.get("response_time_ms", 0) for r in processed_results if r.get("response_time_ms")]
        total_times = [r.get("total_time_ms", 0) for r in processed_results if r.get("total_time_ms")]
        successful_starts = sum(1 for r in processed_results if r.get("status_code") == 200)
        
        # Calculate final video statistics
        video_stats = [r["final_video_stats"] for r in processed_results if r.get("final_video_stats")]
        
        logger.info(f"Load test completed in {total_time:.2f} seconds")
        logger.info(f"Successful starts: {successful_starts}/{len(tasks)}")
        if api_times:
            logger.info(f"Average API response time: {sum(api_times)/len(api_times):.2f} ms")
        if total_times:
            logger.info(f"Average total time: {sum(total_times)/len(total_times):.2f} ms")
        
        if video_stats:
            total_expected = sum(stats['expected_videos'] for stats in video_stats)
            total_created = sum(stats['created_videos'] for stats in video_stats)
            overall_success_rate = total_created / total_expected if total_expected > 0 else 0.0
            avg_wait_time = sum(stats['wait_time_seconds'] for stats in video_stats) / len(video_stats)
            timeouts = sum(1 for stats in video_stats if stats['timed_out'])
            
            logger.info(f"Final Video Statistics:")
            logger.info(f"  Total expected videos: {total_expected}")
            logger.info(f"  Total created videos: {total_created}")
            logger.info(f"  Overall success rate: {overall_success_rate:.2%}")
            logger.info(f"  Average wait time: {avg_wait_time:.1f} seconds")
            logger.info(f"  Timeouts: {timeouts}/{len(video_stats)}")
        
        # Save results to CSV
        save_results_to_csv(processed_results, concurrent_requests)
        
        return processed_results

def save_results_to_csv(results: List[Dict[str, Any]], concurrency: int) -> None:
    """Save detailed results to CSV files including final video statistics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_filename = f"load_test_detailed_{concurrency}_{timestamp}.csv"
    with open(detailed_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'request_id',
            'interview_id',
            'start_status_code',
            'start_response_time_ms',
            'total_time_ms',
            'expected_videos',
            'created_videos',
            'video_success_rate',
            'video_wait_time_seconds',
            'video_timeout',
            'missing_videos',
            'deleted_blobs',
            'error'
        ])
        writer.writeheader()
        
        for result in results:
            video_stats = result.get('final_video_stats', {})
            row = {
                'request_id': result.get('request_id'),
                'interview_id': result.get('interview_id'),
                'start_status_code': result.get('status_code'),
                'start_response_time_ms': result.get('response_time_ms'),
                'total_time_ms': result.get('total_time_ms'),
                'expected_videos': video_stats.get('expected_videos', 0),
                'created_videos': video_stats.get('created_videos', 0),
                'video_success_rate': f"{video_stats.get('success_rate', 0):.2%}",
                'video_wait_time_seconds': f"{video_stats.get('wait_time_seconds', 0):.1f}",
                'video_timeout': video_stats.get('timed_out', False),
                'missing_videos': ';'.join(video_stats.get('missing_videos', [])),
                'deleted_blobs': result.get('deleted_blobs', 0),
                'error': result.get('error', '')
            }
            writer.writerow(row)
    
    # Calculate and save summary statistics
    summary_filename = f"load_test_summary_{concurrency}_{timestamp}.csv"
    with open(summary_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        
        # Calculate statistics
        api_times = [r.get('response_time_ms', 0) for r in results if r.get('response_time_ms')]
        total_times = [r.get('total_time_ms', 0) for r in results if r.get('total_time_ms')]
        video_stats = [r['final_video_stats'] for r in results if r.get('final_video_stats')]
        
        # API response statistics
        if api_times:
            writer.writerow(['API - Average Response Time (ms)', f"{sum(api_times) / len(api_times):.2f}"])
            writer.writerow(['API - Min Response Time (ms)', min(api_times)])
            writer.writerow(['API - Max Response Time (ms)', max(api_times)])
        
        # Total time statistics
        if total_times:
            writer.writerow(['Total - Average Time (ms)', f"{sum(total_times) / len(total_times):.2f}"])
            writer.writerow(['Total - Min Time (ms)', min(total_times)])
            writer.writerow(['Total - Max Time (ms)', max(total_times)])
        
        writer.writerow(['Start - Successful Requests', sum(1 for r in results if r.get('status_code') == 200)])
        writer.writerow(['Start - Failed Requests', sum(1 for r in results if r.get('status_code') != 200)])
        
        # Final video statistics
        if video_stats:
            total_expected = sum(stats['expected_videos'] for stats in video_stats)
            total_created = sum(stats['created_videos'] for stats in video_stats)
            overall_success_rate = total_created / total_expected if total_expected > 0 else 0.0
            avg_wait_time = sum(stats['wait_time_seconds'] for stats in video_stats) / len(video_stats)
            min_wait_time = min(stats['wait_time_seconds'] for stats in video_stats)
            max_wait_time = max(stats['wait_time_seconds'] for stats in video_stats)
            timeouts = sum(1 for stats in video_stats if stats['timed_out'])
            
            writer.writerow(['Video - Total Expected', total_expected])
            writer.writerow(['Video - Total Created', total_created])
            writer.writerow(['Video - Overall Success Rate (%)', f"{overall_success_rate:.2%}"])
            writer.writerow(['Video - Average Wait Time (seconds)', f"{avg_wait_time:.1f}"])
            writer.writerow(['Video - Min Wait Time (seconds)', f"{min_wait_time:.1f}"])
            writer.writerow(['Video - Max Wait Time (seconds)', f"{max_wait_time:.1f}"])
            writer.writerow(['Video - Timeouts', f"{timeouts}/{len(video_stats)}"])
            writer.writerow(['Video - Timeout Rate (%)', f"{timeouts/len(video_stats)*100:.1f}%"])
        
        # Overall statistics
        writer.writerow(['Total Requests', len(results)])
        writer.writerow(['Concurrency Level', concurrency])
        writer.writerow(['Test Timestamp', timestamp])
    
    logger.info(f"Detailed results saved to {detailed_filename}")
    logger.info(f"Summary statistics saved to {summary_filename}")

async def main() -> None:
    """Run the load tests and cleanup."""
    all_results = {}
    
    try:
        # Run tests for each concurrency level
        for concurrency in NUM_REQUESTS:
            results = await run_load_test(concurrency)
            all_results[concurrency] = results
            
            # Wait a bit between tests to let the system recover
            if concurrency != NUM_REQUESTS[-1]:
                logger.info(f"Waiting 5 seconds before next test...")
                await asyncio.sleep(5)
        
    finally:
        # Always attempt to clean up the created interviews
        await delete_interviews()

if __name__ == "__main__":
    # Set event loop policy for better performance on Windows
    if os.name == 'nt':  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())