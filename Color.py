import cv2
import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from collections import defaultdict

class ColorRatioAnalyzer:
    def __init__(self, video_path):
        """Initialize the color analyzer with a video file."""
        self.video = cv2.VideoCapture(video_path)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        
    def analyze_colors(self, interval_seconds=300):
        """Analyze color ratios over time intervals."""
        color_ratios = defaultdict(list)
        intervals = defaultdict(list)
        current_frame = 0
        
        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
                
            # Calculate average color values for each channel
            b, g, r = cv2.split(frame)
            avg_colors = {
                'red': np.mean(r),
                'green': np.mean(g),
                'blue': np.mean(b)
            }
            
            # Calculate ratios
            total = sum(avg_colors.values())
            ratios = {k: v/total for k, v in avg_colors.items()}
            
            # Add to appropriate interval
            interval_idx = int((current_frame / self.fps) / interval_seconds)
            for color, ratio in ratios.items():
                intervals[interval_idx].append(ratios)
            
            current_frame += 1
        
        # Calculate average ratios for each interval
        interval_ratios = []
        for interval in sorted(intervals.keys()):
            if intervals[interval]:
                avg_ratio = {
                    'red': np.mean([r['red'] for r in intervals[interval]]),
                    'green': np.mean([r['green'] for r in intervals[interval]]),
                    'blue': np.mean([r['blue'] for r in intervals[interval]])
                }
                interval_ratios.append(avg_ratio)
        
        # Calculate AUC for each color channel
        x = np.linspace(0, len(interval_ratios), len(interval_ratios))
        auc_scores = {
            'red': auc(x, [r['red'] for r in interval_ratios]),
            'green': auc(x, [r['green'] for r in interval_ratios]),
            'blue': auc(x, [r['blue'] for r in interval_ratios])
        }
        
        return interval_ratios, auc_scores
    
    def plot_results(self, interval_ratios, auc_scores, interval_seconds):
        """Plot color analysis results."""
        plt.figure(figsize=(12, 6))
        x = np.arange(len(interval_ratios)) * interval_seconds / 60
        
        for color in ['red', 'green', 'blue']:
            plt.plot(x, [r[color] for r in interval_ratios], 
                    label=f'{color} (AUC: {auc_scores[color]:.2f})')
        
        plt.title('Color Ratios over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Ratio')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def analyze(self, interval_seconds=300):
        """Perform complete color analysis."""
        print(f"Analyzing color ratios with {interval_seconds} second intervals...")
        
        interval_ratios, auc_scores = self.analyze_colors(interval_seconds)
        self.plot_results(interval_ratios, auc_scores, interval_seconds)
        
        print("\nColor Analysis Summary:")
        for color, score in auc_scores.items():
            print(f"{color.capitalize()} AUC Score: {score:.2f}")
            
        return {
            'interval_ratios': interval_ratios,
            'auc_scores': auc_scores
        }
        
    def __del__(self):
        """Clean up video capture object."""
        self.video.release()

if __name__ == "__main__":
    analyzer = ColorRatioAnalyzer("path_to_your_video.mp4")
    results = analyzer.analyze(interval_seconds=300)