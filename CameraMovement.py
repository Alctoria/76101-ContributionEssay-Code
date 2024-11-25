import cv2
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from collections import defaultdict

class CameraMovementAnalyzer:
    def __init__(self, video_path):
        """Initialize the movement analyzer with a video file."""
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
    def analyze_movement(self, interval_seconds=300):
        """Analyze camera movements over time intervals."""
        movements = defaultdict(list)
        intervals = defaultdict(list)
        current_frame = 0
        
        # Initialize feature detector
        feature_params = dict(maxCorners=100,
                            qualityLevel=0.3,
                            minDistance=7,
                            blockSize=7)
        
        # Lucas-Kanade optical flow parameters
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        ret, old_frame = self.video.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
                
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            
            if p1 is not None:
                # Calculate movement magnitude
                good_new = p1[st==1]
                good_old = p0[st==1]
                
                # Calculate movement vectors
                movement_vectors = good_new - good_old
                movement = np.mean(np.sqrt(np.sum(movement_vectors**2, axis=1)))
                
                # Add to appropriate interval
                interval_idx = int((current_frame / self.fps) / interval_seconds)
                intervals[interval_idx].append(movement)
            
            # Update features
            old_gray = frame_gray.copy()
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            current_frame += 1
        
        # Calculate average movement for each interval
        interval_movements = []
        for interval in sorted(intervals.keys()):
            if intervals[interval]:
                avg_movement = np.mean(intervals[interval])
                interval_movements.append(avg_movement)
            else:
                interval_movements.append(0)
        
        # Calculate AUC
        x = np.linspace(0, len(interval_movements), len(interval_movements))
        auc_score = auc(x, interval_movements)
        
        return interval_movements, auc_score
    
    def plot_results(self, interval_movements, auc_score, interval_seconds):
        """Plot movement analysis results."""
        plt.figure(figsize=(12, 6))
        x = np.arange(len(interval_movements)) * interval_seconds / 60
        plt.plot(x, interval_movements)
        plt.title(f'Camera Movement over Time (AUC: {auc_score:.2f})')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Movement Magnitude')
        plt.grid(True)
        plt.show()
        
    def analyze(self, interval_seconds=300):
        """Perform complete movement analysis."""
        print(f"Analyzing camera movements with {interval_seconds} second intervals...")
        
        interval_movements, auc_score = self.analyze_movement(interval_seconds)
        self.plot_results(interval_movements, auc_score, interval_seconds)
        
        print("\nCamera Movement Analysis Summary:")
        print(f"Movement AUC Score: {auc_score:.2f}")
        print(f"Average Movement Magnitude: {np.mean(interval_movements):.2f}")
        
        return {
            'interval_movements': interval_movements,
            'auc_score': auc_score
        }
        
    def __del__(self):
        """Clean up video capture object."""
        self.video.release()

if __name__ == "__main__":
    analyzer = CameraMovementAnalyzer("path_to_your_video.mp4")
    results = analyzer.analyze(interval_seconds=300)