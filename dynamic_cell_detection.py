import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
from scipy.ndimage import label

GRID_SIZE = 100
RESOLUTION = 0.1
DOORS_THRESHOLD = 0.49
TOLERANCE = 1e-6

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
            '/laser_scan',#'/pioneer1/scan',
            self.listener_callback,
            10)
        self.grid_publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.change_publisher = self.create_publisher(OccupancyGrid, 'detected_objects', 10)
        
        self.timer = self.create_timer(1.0, self.publish_occupancy_grid)

        # Initialize log-odds grid map with log-odds corresponding to 0.5 probability
        self.grid_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)
        self.memory_map = np.full((GRID_SIZE, GRID_SIZE), P_INIT, dtype=np.float32)
        self.change_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
        self.blobs = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)

        self.segments_list = []

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
                    probability =  np.log(self.grid_map[x0, y0]/(1-self.grid_map[x0, y0]))
                    new_probability = probability + LOG_ODDS_FREE
                    self.grid_map[x0, y0] = (1 - 1 / (1 + np.exp(new_probability)))
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

    def sanitize_map(self, map):
        map[np.isinf(map)] = 0
        map = np.nan_to_num(map)
        return map

    def segmentation(self, binary_mask):
        pattern = [[1,1,1],
                [1,1,1],
                [1,1,1]]
        
        labeled_array, num_features = label(binary_mask, pattern)
        segments = []
        
        for i in range(1, num_features + 1):
            segment = np.argwhere(labeled_array == i)
            if len(segment) >= 3:  # Filter out small segments
                segments.append(segment)
        
        return segments, labeled_array

    def remove_duplicates(self, list_of_arrays):
        seen = set()
        result = []
        for array in list_of_arrays:
            tup = tuple(map(tuple, array))
            if tup not in seen:
                seen.add(tup)
                result.append(array)
        return result

    def find_most_similar_segment(self, unique_point, source_segment, segments):
        # Compute similarity based on the number of common points
        source_points_set = set(tuple(point) for point in source_segment)
        max_similarity = 0
        most_similar_segment = None
        for segment in segments:
            common_points = source_points_set.intersection(set(tuple(point) for point in segment))
            similarity = len(common_points)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_segment = segment
        return most_similar_segment

    def divide_blobs(self, segment_list):
        included_points = set()
        output_segments = []
        single_point_segments = []

        for segment in segment_list:
            segment_points = [tuple(point) for point in segment]
            unique_points = [point for point in segment_points if point not in included_points]
            unique_segment = np.array(unique_points)
            if len(unique_segment) > 1:
                output_segments.append(unique_segment)
                included_points.update(unique_points)
            elif len(unique_segment) == 1:
                single_point_segments.append((unique_segment[0], segment))

        for unique_point, source_segment in single_point_segments:
            most_similar_segment = self.find_most_similar_segment(unique_point, source_segment, output_segments)
            if most_similar_segment is not None:
                most_similar_segment = np.vstack([most_similar_segment, unique_point])
            else:
                output_segments.append(np.array([unique_point]))
        
        return output_segments

    def check_for_changes(self):
        difference_map = np.absolute(self.grid_map - self.memory_map)
        difference_map = self.sanitize_map(difference_map)
        binary_map = np.where(difference_map - DOORS_THRESHOLD > TOLERANCE, 1, 0)
        binary_map = self.sanitize_map(binary_map)

        segments, segment_labels = self.segmentation(binary_map)
        
        for segment in segments:
            for point in segment:
                if self.change_map[point[0], point[1]] == 0:
                    self.change_map[point[0], point[1]] = 1

            flag = True
            for noted_segment in self.segments_list:
                if np.array_equal(segment, noted_segment, equal_nan=False):
                    flag = False
            
            if flag is True:
                self.segments_list.append(segment)

        # segments, segment_labels = self.segmentation(self.change_map)
        self.segments_list = self.remove_duplicates(self.segments_list)
        detected_objects = self.divide_blobs(self.segments_list)

        for segment_id, segment in enumerate(detected_objects, start=1):
            for (x, y) in segment:
                self.blobs[x, y] = segment_id

        print(len(detected_objects))
        self.memory_map = np.copy(self.grid_map)

    def listener_callback(self, msg):
        for i, range in enumerate(msg.ranges):
            if range < msg.range_min or range > msg.range_max or np.isnan(range):
                continue  # ignore out-of-range values

            angle = msg.angle_min + i * msg.angle_increment
            rotation = 0 #np.pi*3/4
            x = range * np.cos(angle+rotation)
            y = range * np.sin(angle+rotation)

            grid_x = int(self.origin_x + x / RESOLUTION)
            grid_y = int(self.origin_y + y / RESOLUTION)

            # Apply Bresenham's line algorithm to update free cells along the ray
            self.bresenham(self.origin_x, self.origin_y, grid_x, grid_y)

            if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                old_probability =  np.log(self.grid_map[grid_x, grid_y]/(1-self.grid_map[grid_x, grid_y]))
                new_probability = old_probability + LOG_ODDS_OCCUPIED
                self.grid_map[grid_x, grid_y] = (1 - 1 / (1 + np.exp(new_probability)))
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
        # probabilities = 1 - 1 / (1 + np.exp(rotated_map))
        # occupancy_grid.data = (probabilities * 100).astype(np.int8).flatten().tolist()

        # Publish the occupancy grid
        # self.grid_publisher.publish(occupancy_grid)
        occupancy_grid.data = (rotated_map * 100).astype(np.int8).flatten().tolist()
        self.grid_publisher.publish(occupancy_grid)
        self.get_logger().info('Occupancy grid published')

        # Publish the change map
        self.publish_changed_cells_grid()


    def publish_changed_cells_grid(self):
        if self.last_scan_time is None:
            return  # No scan data received yet
        
        flipped_change_map = np.flip(self.blobs, axis=0)
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

        change_grid.data = (rotated_change_map* 2).astype(np.int8).flatten().tolist()
        # change_grid.data = flipped_change_map.flatten().tolist()
        # change_grid.data = rotated_change_map.flatten().tolist()

        # Publish the change grid
        self.change_publisher.publish(change_grid)
        self.get_logger().info('Detected objects published')


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
