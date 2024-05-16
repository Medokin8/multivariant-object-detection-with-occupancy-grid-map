import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import csv
import os

class LaserListenerNode(Node):

    def __init__(self):
        super().__init__('laser_listener_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/laser_scan',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        csv_file = os.path.join(os.path.expanduser(os.getcwd()), 'laser_scan_data.csv')

        # Extract data from the LaserScan message
        data = {
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,
            'range_min': msg.range_min,
            'range_max': msg.range_max,
            'ranges': list(msg.ranges),
            'intensities': list(msg.intensities)
        }

        # Write data to CSV
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data.keys())
            writer.writerow(data.values())

        self.get_logger().info(f'Data written to {csv_file}')
        
        # Shutdown the node after writing the file
        self.get_logger().info('Shutting down node...')
        rclpy.shutdown()

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