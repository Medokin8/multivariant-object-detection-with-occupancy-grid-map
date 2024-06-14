import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

GRID_SIZE = 100
CELL_SIZE = 0.1
ROTATION = 0

P_INIT = 0.5
P_OCCUPIED = 0.9
P_FREE = 0.1
P_MAX = 0.99
P_MIN = 0.01
LOG_ODDS_OCCUPIED = np.log(P_OCCUPIED / (1 - P_OCCUPIED))
LOG_ODDS_FREE = np.log(P_FREE / (1 - P_FREE))

TOPIC = "/pioneer1/scan"
PATH_AND_FILE_TO_SAVE = "tmp/tmp"


class LaserListenerNode(Node):

    def __init__(self):
        super().__init__("laser_listener_node")
        self.subscription = self.create_subscription(
            LaserScan,
            TOPIC,
            self.listener_callback,
            10,
        )

        self.timer = self.create_timer(1.0, self.check_for_topic)

        self.grid_map = np.full(
            (GRID_SIZE, GRID_SIZE),
            P_INIT,
            dtype=np.float32,
        )

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
                if x0 != x1 or y0 != y1:
                    probability = np.log(
                        self.grid_map[x0, y0] / (1 - self.grid_map[x0, y0])
                    )
                    new_probability = probability + LOG_ODDS_FREE
                    self.grid_map[x0, y0] = 1 - 1 / (
                        1 + np.exp(new_probability)
                    )
                    self.grid_map[x0, y0] = np.clip(
                        self.grid_map[x0, y0],
                        P_MIN,
                        P_MAX,
                    )

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
            if (
                range < msg.range_min
                or range > msg.range_max
                or np.isnan(range)
            ):
                continue

            angle = msg.angle_min + i * msg.angle_increment
            x = range * np.cos(angle + ROTATION)
            y = range * np.sin(angle + ROTATION)

            grid_x = int(self.origin_x + x / CELL_SIZE)
            grid_y = int(self.origin_y + y / CELL_SIZE)

            self.bresenham(
                self.origin_x,
                self.origin_y,
                grid_x,
                grid_y,
            )

            if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                old_probability = np.log(
                    self.grid_map[grid_x, grid_y]
                    / (1 - self.grid_map[grid_x, grid_y])
                )
                new_probability = old_probability + LOG_ODDS_OCCUPIED
                self.grid_map[grid_x, grid_y] = 1 - 1 / (
                    1 + np.exp(new_probability)
                )
                self.grid_map[grid_x, grid_y] = np.clip(
                    self.grid_map[grid_x, grid_y],
                    P_MIN,
                    P_MAX,
                )

        self.last_scan_time = msg.header.stamp

    def check_for_topic(self):
        topics = self.get_topic_names_and_types()
        if not any(TOPIC in topic for topic, types in topics):
            self.get_logger().info(
                "/laser_scan topic not found, shutting down..."
            )
            rclpy.shutdown()

    def save_memory_map(self):
        PATH_AND_FILE_TO_SAVE
        probabilities = self.grid_map
        np.save(
            PATH_AND_FILE_TO_SAVE + ".npy",
            probabilities,
        )
        self.get_logger().info("Memory map saved to memory_map.npy")
        np.savetxt(
            PATH_AND_FILE_TO_SAVE + ".csv",
            probabilities,
            delimiter=",",
        )
        self.get_logger().info("Memory map saved to memory_map.csv")


def main(args=None):
    rclpy.init(args=args)
    node = LaserListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_memory_map()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
