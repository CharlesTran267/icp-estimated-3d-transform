import json
import cv2
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
import open3d as o3d
import copy


def draw_registration_result(source, target, transformation=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
    return source_temp


def get_points_inside_contours(contour_points, xyz):
    mask = np.zeros_like(image[:, :, 0])

    # Fill the contour region in the mask with white color (255)
    cv2.drawContours(mask, [contour_points], 0, 255, cv2.FILLED)

    # Get the indices of points inside the contour (where the mask is white)
    indices = np.where(mask == 255)

    cv2.imwrite('mask.png', mask)

    # Access the original coordinates from your array of (x, y, z) coordinates
    coordinates_inside_contour = xyz[indices]
    return coordinates_inside_contour


def remove_outliers(coordinates, threshold=1.5):
    # Calculate the IQR for each dimension
    q1 = np.percentile(coordinates, 25, axis=0)
    q3 = np.percentile(coordinates, 75, axis=0)
    iqr = q3 - q1
    
    # Define the lower and upper bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    # Find the indices of the outliers
    outliers = np.any((coordinates < lower_bound) | (coordinates > upper_bound), axis=1)

    # Remove the outliers from the array
    cleaned_coordinates = coordinates[~outliers]
    
    return cleaned_coordinates


def nomalize_coordinate(coordinates):
    # Set x min, y min, z min to 0
    # Set x max to 10 and keep the ratio of x, y, z
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)

    new_x = ((x - x_min)/(x_max - x_min)) *10
    new_y = ((y - y_min)/(x_max - x_min)) *10
    new_z = ((z - z_min)/(x_max - x_min)) *10

    transformed_coordinates = np.column_stack((new_x, new_y, new_z))
    return transformed_coordinates


def remove_point_beyond_depth(coordinates, depth):
    # Remove points that has z value greater than depth
    z = coordinates[:, 2]
    indices = np.where(z < depth)
    cleaned_coordinates = coordinates[indices]
    return cleaned_coordinates


# Load the image, xyz coordinates and the detected contours
detected_result_path = '/home/archuser/switch-pose-estimation/detected_result/image (1).json'
xyz_path = '/home/archuser/switch-pose-estimation/rgb-xyz/xyz/image (1).npy'
image_path = '/home/archuser/switch-pose-estimation/rgb-xyz/rgb/image (1).png'
mesh_model_path = '/home/archuser/switch-pose-estimation/D2SW_P2D.STL'

xyz = np.load(xyz_path)
image = cv2.imread(image_path)
with open(detected_result_path) as f:
    detected = json.load(f)
    # Select this object as this one has the closest shape to the model
    selected_idx = 2
    contour_points = np.array(detected["contours"][selected_idx])


# Get the points inside the contour
coordinates_inside_contour = get_points_inside_contours(contour_points, xyz)
coordinates_inside_contour = remove_outliers(coordinates_inside_contour)


# Normalize the coordinates and convert to point cloud
normalized_detected_array = nomalize_coordinate(coordinates_inside_contour)
detected_pcl = o3d.geometry.PointCloud()
detected_pcl.points = o3d.utility.Vector3dVector(normalized_detected_array)


# Load mesh model, convert to point cloud, normalize the point cloud and just keep the front part
mesh = o3d.io.read_triangle_mesh(mesh_model_path)
model_pcl = mesh.sample_points_uniformly(100000)
model_pcl_array = np.asarray(model_pcl.points)
normalized_model_array = nomalize_coordinate(model_pcl_array)
THICK_NESS = 0.1
front_model_array = remove_point_beyond_depth(normalized_model_array, THICK_NESS)
model_pcl.points = o3d.utility.Vector3dVector(front_model_array)

source = detected_pcl
target = model_pcl
threshold = 0.02

# Draw original position
draw_registration_result(source, target)

# Draw the initial alignment
trans_init = np.identity(4)
draw_registration_result(source, target, trans_init)


# Evaluate the initial alignment
print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(
    source, target, threshold, trans_init)
print(evaluation)


# Point-to-point ICP for refinement
print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
draw_registration_result(source, target, reg_p2p.transformation)







