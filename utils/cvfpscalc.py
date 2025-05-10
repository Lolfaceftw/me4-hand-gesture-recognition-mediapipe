from collections import deque
import cv2 as cv
import time


class CvFpsCalc(object):
    def __init__(self, buffer_len=5):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)
        self._last_fps_calc_time = time.time()

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / different_time if different_time > 0 else 0
        fps_rounded = round(fps, 1)

        return fps_rounded

    def debug_info(self):
        # For debugging purposes
        avg_frame_time = sum(self._difftimes) / len(self._difftimes) if self._difftimes else 0
        return {
            "buffer_size": len(self._difftimes),
            "max_buffer": self._difftimes.maxlen,
            "avg_frame_time_ms": round(avg_frame_time, 2),
            "calculated_fps": round(1000.0 / avg_frame_time, 2) if avg_frame_time > 0 else 0
        }
