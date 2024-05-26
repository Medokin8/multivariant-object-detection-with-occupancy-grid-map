import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
import os

GRID_SIZE = 100
RESOLUTION = 0.1

# Parameters for log-odds
P_OCCUPIED = 0.9
P_FREE = 0.1
LOG_ODDS_OCCUPIED = np.log(P_OCCUPIED / (1 - P_OCCUPIED))
LOG_ODDS_FREE = np.log(P_FREE / (1 - P_FREE))
P_INIT = 0.5  # log-odds of 0.5 probability
P_MAX = 0.99
P_MIN = 0.01

class LaserListenerNode(Node):

    def __init__(self):
        super().__init__('laser_listener_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.listener_callback,
            10)
        
        self.timer = self.create_timer(1.0, self.check_for_topic)

        # Initialize log-odds grid map with log-odds corresponding to 0.5 probability
        self.grid_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)
        self.memory_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)

        # Origin of the grid map (assuming robot starts at the center)
        self.origin_x = GRID_SIZE // 2
        self.origin_y = GRID_SIZE // 2
        
        self.last_scan_time = None

    def bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if 0 <= x0 < GRID_SIZE and 0 <= y0 < GRID_SIZE:
                if (x0 != x1 or y0 != y1):
                    self.grid_map[x0, y0] -= (1 - 1 / (1 + np.exp(LOG_ODDS_FREE))) # = 0
                    self.grid_map[x0, y0] = np.clip(self.grid_map[x0, y0], P_MIN, P_MAX)
            
            if x0 == x1 and y0 == y1:
                break

            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    def listener_callback(self, msg):
        for i, range in enumerate(msg.ranges):
            if range < msg.range_min or range > msg.range_max:
                continue  # ignore out-of-range values

            angle = msg.angle_min + i * msg.angle_increment
            x = range * np.cos(angle)
            y = range * np.sin(angle)

            grid_x = int(self.origin_x + x / RESOLUTION)
            grid_y = int(self.origin_y + y / RESOLUTION)

            # Apply Bresenham's line algorithm to update free cells along the ray
            self.bresenham(self.origin_x, self.origin_y, grid_x, grid_y)

            if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                self.grid_map[grid_x, grid_y] += (1 - 1 / (1 + np.exp(LOG_ODDS_OCCUPIED))) # = 1
                self.grid_map[grid_x, grid_y] = np.clip(self.grid_map[grid_x, grid_y], P_MIN, P_MAX)
        
        self.last_scan_time = msg.header.stamp

    def check_for_topic(self):
        topics = self.get_topic_names_and_types()
        if not any('/laser_scan' in topic for topic, types in topics):
            self.get_logger().info('/laser_scan topic not found, shutting down...')
            rclpy.shutdown()

    def save_memory_map(self):
        # Convert log-odds map to probabilities
        folder_and_name = "simulated_maps/closed"
        probabilities = 1 - 1 / (1 + np.exp(self.grid_map))
        np.save(folder_and_name + ".npy", probabilities)
        self.get_logger().info('Memory map saved to memory_map.npy')
        np.savetxt(folder_and_name + ".csv", probabilities, delimiter=",")
        self.get_logger().info('Memory map saved to memory_map.csv')

def main(args=None):
    rclpy.init(args=args)
    node = LaserListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_memory_map()  # Save the memory map before shutting down
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
