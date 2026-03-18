"""Check point cloud filtering statistics."""

import numpy as np
import os

def check_point_cloud_filtering(data_path, num_samples=20):
    """Check how many points are filtered out during BEV projection."""

    print("="*60)
    print("Point Cloud Filtering Analysis")
    print("="*60)

    # LiDAR projection parameters
    x_min, x_max = 0, 80
    y_min, y_max = -40, 40
    z_min, z_max = -2, 6

    total_points = []
    filtered_points = []
    z_filtered_points = []
    xy_filtered_points = []

    # Find some lidar files
    train_dir = os.path.join(data_path, 'train')
    sequences = os.listdir(train_dir)[:5]  # Check first 5 sequences

    count = 0
    for seq in sequences:
        seq_path = os.path.join(train_dir, seq)
        samples = os.listdir(seq_path)

        for sample in samples:
            if count >= num_samples:
                break

            sample_path = os.path.join(seq_path, sample)
            lidar_file = os.path.join(sample_path, 'os1.npy')

            if not os.path.exists(lidar_file):
                continue

            # Load point cloud
            pc = np.load(lidar_file)

            if pc.ndim != 2 or pc.shape[1] != 9:
                continue

            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]

            total_pts = len(x)
            total_points.append(total_pts)

            # Check XY filtering
            xy_mask = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
            xy_kept = xy_mask.sum()
            xy_filtered_points.append(xy_kept)

            # Check Z filtering
            z_mask = (z >= z_min) & (z < z_max)
            z_kept = z_mask.sum()
            z_filtered_points.append(z_kept)

            # Check combined filtering
            combined_mask = xy_mask & z_mask
            combined_kept = combined_mask.sum()
            filtered_points.append(combined_kept)

            print(f"\nSample {count}:")
            print(f"  Total points: {total_pts}")
            print(f"  After XY filter: {xy_kept} ({xy_kept/total_pts*100:.1f}%)")
            print(f"  After Z filter: {z_kept} ({z_kept/total_pts*100:.1f}%)")
            print(f"  After combined filter: {combined_kept} ({combined_kept/total_pts*100:.1f}%)")

            # Check Z distribution
            print(f"  Z range: [{z.min():.2f}, {z.max():.2f}]")
            print(f"  Z within [-2, 6]: {((z >= -2) & (z <= 6)).sum()} points")

            count += 1

        if count >= num_samples:
            break

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"\nAverage points per sample:")
    print(f"  Original: {np.mean(total_points):.0f}")
    print(f"  After XY filter: {np.mean(xy_filtered_points):.0f} ({np.mean(xy_filtered_points)/np.mean(total_points)*100:.1f}%)")
    print(f"  After Z filter: {np.mean(z_filtered_points):.0f} ({np.mean(z_filtered_points)/np.mean(total_points)*100:.1f}%)")
    print(f"  After combined filter: {np.mean(filtered_points):.0f} ({np.mean(filtered_points)/np.mean(total_points)*100:.1f}%)")

    # Check if Z filtering is the bottleneck
    z_loss_ratio = 1 - np.mean(z_filtered_points) / np.mean(total_points)
    xy_loss_ratio = 1 - np.mean(xy_filtered_points) / np.mean(total_points)

    print(f"\nFiltering impact:")
    print(f"  XY filter removes: {xy_loss_ratio*100:.1f}% of points")
    print(f"  Z filter removes: {z_loss_ratio*100:.1f}% of points")

    if z_loss_ratio > 0.5:
        print(f"\n⚠️  WARNING: Z filter removes >50% of points!")
        print(f"  Consider adjusting z_range in lidar_info.py")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python check_point_cloud_filtering.py <data_path>")
        print("Example: python check_point_cloud_filtering.py /data/kradar/processed")
        sys.exit(1)

    check_point_cloud_filtering(sys.argv[1])
