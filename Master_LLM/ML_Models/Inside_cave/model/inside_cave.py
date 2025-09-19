import cv2
import os
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from inference_sdk import InferenceHTTPClient
from pprint import pprint
from dotenv import load_dotenv


def process_video_file(video_file,
                       workspace_name="asn-rvnzk",
                       workflow_id="custom-workflow",
                       env_path=r"D:\RockFall_ML-GenAI\.env",
                       output_path="predictions_insidecave.json",
                       conf_threshold=0.4,
                       interval_sec=2):
    """
    Process an uploaded video file (not just path), run inference on sampled frames,
    and save predictions to JSON.

    Args:
        video_file (file-like or str): Video file object (e.g. from upload) or path.
        workspace_name (str): Roboflow workspace name.
        workflow_id (str): Workflow ID.
        env_path (str): Path to .env file containing API key.
        output_path (str): Path to save JSON predictions.
        conf_threshold (float): Confidence threshold for filtering predictions.
        interval_sec (float): Seconds between frames to sample.

    Returns:
        dict: Predictions for all sampled frames.
    """
    # Load API key
    load_dotenv(env_path)
    API_KEY = os.getenv("OUTER_SURFACE_API_KEY")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=API_KEY
    )

    # If user passed an uploaded file object, save it temporarily
    if not isinstance(video_file, str):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            video_path = tmp.name
    else:
        video_path = video_file

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Error: Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_sec)

    frame_count = 0
    saved_frame_count = 0

    def process_frame(frame_id, frame):
        temp_path = f"frame_{frame_id}.jpg"
        cv2.imwrite(temp_path, frame)

        try:
            # Run workflow
            result = client.run_workflow(
                workspace_name=workspace_name,
                workflow_id=workflow_id,
                images={"image": temp_path},
                use_cache=True
            )

            # Filter predictions
            cleaned_predictions = []
            for pred in result[0]["model_predictions"]["predictions"]:
                if pred.get("confidence", 0) >= conf_threshold:
                    pred_copy = {k: v for k, v in pred.items() if k != "points"}
                    cleaned_predictions.append(pred_copy)

            return frame_id, cleaned_predictions
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Thread pool for parallel frame processing
    executor = ThreadPoolExecutor(max_workers=12)
    futures = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            futures.append(executor.submit(process_frame, saved_frame_count, frame))
            saved_frame_count += 1

        frame_count += 1

    all_predictions = {}

    for future in as_completed(futures):
        frame_id, predictions = future.result()
        all_predictions[f"frame_{frame_id}"] = predictions
        print(f"\n🔹 Predictions for frame {frame_id}:")
        pprint(predictions)

    # Save predictions
    with open(output_path, "w") as f:
        json.dump(all_predictions, f, indent=4)

    cap.release()
    executor.shutdown()

    print(f"✅ Video processing completed. Predictions saved to {output_path}")
    return all_predictions


# Example usage
if __name__ == "__main__":
    with open(r"C:\Users\KAIZEN\Downloads\vedio_testing\inner_cave\generated-video.mp4", "rb") as f:
        predictions = process_video_file(f)

    print("\n📌 Final Predictions Dictionary:")
    print(json.dumps(predictions, indent=4))
