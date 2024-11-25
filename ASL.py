import cv2
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from collections import defaultdict

class ShotLengthAnalyzer:
    def __init__(self, video_path):
        """Initialize the ASL analyzer with a video file."""
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
    def analyze_shots(self, interval_seconds=300):
        """Analyze average shot length over time intervals."""
        shot_lengths = []
        frame_diff_threshold = 30.0
        last_shot_frame = 0
        
        intervals = defaultdict(list)
        current_frame = 0
        
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
                
            if current_frame > 0:
                # Convert to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate frame difference
                frame_diff = np.mean(np.abs(gray_frame - prev_gray_frame))
                
                if frame_diff > frame_diff_threshold:
                    shot_length = (current_frame - last_shot_frame) / self.fps
                    shot_lengths.append(shot_length)
                    
                    # Add to appropriate interval
                    interval_idx = int((current_frame / self.fps) / interval_seconds)
                    intervals[interval_idx].append(shot_length)
                    
                    last_shot_frame = current_frame
            
            prev_frame = frame.copy()
            current_frame += 1
        
        # Calculate ASL for each interval
        interval_asls = []
        for interval in sorted(intervals.keys()):
            if intervals[interval]:
                asl = np.mean(intervals[interval])
                interval_asls.append(asl)
            else:
                interval_asls.append(0)
        
        # Calculate AUC
        x = np.linspace(0, len(interval_asls), len(interval_asls))
        auc_score = auc(x, interval_asls)
        
        return shot_lengths, interval_asls, auc_score
    
    def plot_results(self, interval_asls, auc_score, interval_seconds):
        """Plot ASL analysis results."""
        plt.figure(figsize=(12, 6))
        x = np.arange(len(interval_asls)) * interval_seconds / 60  # Convert to minutes
        plt.plot(x, interval_asls)
        plt.title(f'Average Shot Length over Time (AUC: {auc_score:.2f})')
        plt.xlabel('Time (minutes)')
        plt.ylabel('ASL (seconds)')
        plt.grid(True)
        plt.show()
        
    def analyze(self, interval_seconds=300):
        """Perform complete ASL analysis."""
        print(f"Analyzing shot lengths with {interval_seconds} second intervals...")
        
        shot_lengths, interval_asls, auc_score = self.analyze_shots(interval_seconds)
        self.plot_results(interval_asls, auc_score, interval_seconds)
        
        print("\nASL Analysis Summary:")
        print(f"Overall ASL AUC Score: {auc_score:.2f}")
        print(f"Average Shot Length: {np.mean(shot_lengths):.2f} seconds")
        print(f"Total Shots: {len(shot_lengths)}")
        
        return {
            'shot_lengths': shot_lengths,
            'interval_asls': interval_asls,
            'auc_score': auc_score
        }
        
    def __del__(self):
        """Clean up video capture object."""
        self.video.release()

if __name__ == "__main__":
    analyzer = ShotLengthAnalyzer("path_to_your_video.mp4")
    results = analyzer.analyze(interval_seconds=300)