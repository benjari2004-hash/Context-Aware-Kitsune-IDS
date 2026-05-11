import math
import time
from collections import deque

class FeatureExtractor:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.packets = deque()

    def entropy(self, data):
        if len(data) == 0:
            return 0
        prob = [float(data.count(c)) / len(data) for c in set(data)]
        return -sum([p * math.log2(p) for p in prob])

    def extract(self, packet):
        now = time.time()

        # تنظيف الباكيتات القديمة
        while self.packets and now - self.packets[0] > self.window_size:
            self.packets.popleft()

        self.packets.append(now)

        # 1. Packet Size
        size = len(packet)

        # 2. Packet Rate
        packet_rate = len(self.packets) / self.window_size

        # 3. Entropy
        try:
            entropy_value = self.entropy(packet)
        except:
            entropy_value = 0

        return [size, packet_rate, entropy_value]