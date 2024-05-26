import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np

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
        self.grid_publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.change_publisher = self.create_publisher(OccupancyGrid, 'changed_cells_grid', 10)
        
        self.timer = self.create_timer(1.0, self.publish_occupancy_grid)
        
        # Initialize log-odds grid map with log-odds corresponding to 0.5 probability
        self.grid_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)
        self.memory_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)
        self.change_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

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

    def check_for_changes(self):
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                current_prob = self.grid_map[i, j]
                previous_prob = self.memory_map[i, j]
                if previous_prob > 0.5 and current_prob < 0.5:
                    self.change_map[i, j] = 100  # Mark the cell as changed
                
        self.memory_map = np.copy(self.grid_map)


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
        
        self.check_for_changes()
        self.last_scan_time = msg.header.stamp


    def publish_occupancy_grid(self):
        if self.last_scan_time is None:
            return  # No scan data received yet

        flipped_map = np.flip(self.grid_map, axis=0)
        rotated_map = np.rot90(flipped_map, k=-1)

        # Convert log-odds map to occupancy grid
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = Header()
        occupancy_grid.header.stamp = self.last_scan_time
        occupancy_grid.header.frame_id = 'sim_lidar'

        occupancy_grid.info.resolution = RESOLUTION
        occupancy_grid.info.width = GRID_SIZE
        occupancy_grid.info.height = GRID_SIZE
        occupancy_grid.info.origin.position.x = - (GRID_SIZE // 2) * RESOLUTION
        occupancy_grid.info.origin.position.y = - (GRID_SIZE // 2) * RESOLUTION
        occupancy_grid.info.origin.position.z = 0.0
        occupancy_grid.info.origin.orientation.w = 1.0  # no rotation

        # Convert log-odds to probabilities and then to occupancy values
        probabilities = 1 - 1 / (1 + np.exp(rotated_map))
        occupancy_grid.data = (probabilities * 100).astype(np.int8).flatten().tolist()

        # Publish the occupancy grid
        self.grid_publisher.publish(occupancy_grid)
        self.get_logger().info('Occupancy grid published')

        # Publish the change map
        self.publish_changed_cells_grid()


    def publish_changed_cells_grid(self):
        flipped_change_map = np.flip(self.change_map, axis=0)
        rotated_change_map = np.rot90(flipped_change_map, k=-1)

        change_grid = OccupancyGrid()
        change_grid.header = Header()
        change_grid.header.stamp = self.last_scan_time
        change_grid.header.frame_id = 'sim_lidar'

        change_grid.info.resolution = RESOLUTION
        change_grid.info.width = GRID_SIZE
        change_grid.info.height = GRID_SIZE
        change_grid.info.origin.position.x = - (GRID_SIZE // 2) * RESOLUTION
        change_grid.info.origin.position.y = - (GRID_SIZE // 2) * RESOLUTION
        change_grid.info.origin.position.z = 0.0
        change_grid.info.origin.orientation.w = 1.0  # no rotation

        change_grid.data = rotated_change_map.flatten().tolist()

        # Publish the change grid
        self.change_publisher.publish(change_grid)
        self.get_logger().info('Changed cells grid published')


def main(args=None):
    rclpy.init(args=args)
    node = LaserListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
