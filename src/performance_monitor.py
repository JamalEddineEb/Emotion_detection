import time
import psutil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.metrics = {
            'face_detection': {'time': [], 'memory': []},
            'color_conversion': {'time': [], 'memory': []},
            'face_cropping': {'time': [], 'memory': []},
            'grayscale_conversion': {'time': [], 'memory': []},
            'resize_normalize': {'time': [], 'memory': []},
            'inference': {'time': [], 'memory': []},
            'display': {'time': [], 'memory': []}
        }
    
    def get_memory_usage(self):
        return self.process.memory_info().rss / 1024 / 1024  # MB
    
    def measure(self, component):
        def decorator(func):
            def wrapper(*args, **kwargs):
                mem_before = self.get_memory_usage()
                time_start = time.time()
                
                result = func(*args, **kwargs)
                
                time_taken = time.time() - time_start
                mem_used = self.get_memory_usage() - mem_before
                
                self.metrics[component]['time'].append(time_taken)
                self.metrics[component]['memory'].append(mem_used)
                
                return result
            return wrapper
        return decorator
    
    def print_metrics(self, duration, frame_count):
        fps = frame_count / duration
        print("\nPerformance Analysis:")
        print(f"Average FPS: {fps:.2f}")
        print(f"Total frames processed: {frame_count}")
        print(f"Total duration: {duration:.2f} seconds")
        
        # Sort components by average time
        sorted_metrics = {}
        for component, measurements in self.metrics.items():
            if measurements['time']:  # Only include if we have measurements
                avg_time = np.mean(measurements['time']) * 1000  # Convert to ms
                sorted_metrics[component] = avg_time
        
        # Print sorted by time consumption
        print("\nComponents sorted by time consumption:")
        for component, avg_time in sorted(sorted_metrics.items(), key=lambda x: x[1], reverse=True):
            avg_memory = np.mean(self.metrics[component]['memory'])
            print(f"\n{component.upper()}:")
            print(f"Average time: {avg_time:.2f} ms")
            print(f"Average memory: {avg_memory:.2f} MB")