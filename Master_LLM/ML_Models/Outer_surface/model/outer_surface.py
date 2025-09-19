import cv2
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from inference_sdk import InferenceHTTPClient
from pprint import pprint
from dotenv import load_dotenv


def process_video(video_path, workspace_name, workflow_id, env_path, output_path="predictions.json", conf_threshold=0.4, interval_sec=2):
    """
    Process a video, run inference on sampled frames, and return/save predictions.

    Args:
        video_path (str): Path to input video.
        workspace_name (str): Roboflow workspace name.
        workflow_id (str): Workflow ID.
        env_path (str): Path to .env file containing OUTER_SURFACE_API_KEY.
        output_path (str): Path to save JSON predictions.
        conf_threshold (float): Confidence threshold for filtering predictions.
        interval_sec (float): Seconds between frames to sample.

    Returns:
        dict: Predictions for all sampled frames.
    """
    # Load API key
    load_dotenv(env_path)
    OUTER_SURFACE_API_KEY = os.getenv("OUTER_SURFACE_API_KEY")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=OUTER_SURFACE_API_KEY
    )

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

            # Clean + filter predictions
            cleaned_predictions = []
            for pred in result[0]["model_predictions"]["predictions"]:
                if pred.get("confidence", 0) >= conf_threshold:
                    pred_copy = {k: v for k, v in pred.items() if k != "points"}
                    cleaned_predictions.append(pred_copy)

            return frame_id, cleaned_predictions
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Thread pool
    executor = ThreadPoolExecutor(max_workers=4)
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

if __name__ == "__main__":
    video_path = r"C:\Users\KAIZEN\Downloads\vedio_testing\outer_detect\mine.mp4"
    workspace_name = "asn-rvnzk"
    workflow_id = "custom-workflow"
    env_path = r"D:\RockFall_ML-GenAI\.env"
    output_path = "predictions.json"
    conf_threshold = 0.4
    interval_sec = 2

    # Call the video processing function
    predictions = process_video(
        video_path=video_path,
        workspace_name=workspace_name,
        workflow_id=workflow_id,
        env_path=env_path,
        output_path=output_path,
        conf_threshold=conf_threshold,
        interval_sec=interval_sec
    )

    # Optional: print final predictions
    print("\n📌 Final Predictions Dictionary:")
    print(json.dumps(predictions, indent=4))

# from inference_sdk import InferenceHTTPClient
# from pprint import pprint

# client = InferenceHTTPClient(
#     api_url="https://serverless.roboflow.com",
#     api_key="k0S14T9n9LU4cxDZLYva"
# )

# result = client.run_workflow(
#     workspace_name="asn-rvnzk",
#     workflow_id="custom-workflow",
#     images={
#         "image": r"D:\RockFall_ML-GenAI\Backend\Master_LLM\ML_Models\Outer_surface\static\images (3).jpeg"
#     },
#     use_cache=True
# )

# # Extract predictions and drop "points"
# cleaned_predictions = []
# for pred in result[0]["model_predictions"]["predictions"]:
#     pred_copy = {k: v for k, v in pred.items() if k != "points"}
#     cleaned_predictions.append(pred_copy)

# pprint(cleaned_predictions)