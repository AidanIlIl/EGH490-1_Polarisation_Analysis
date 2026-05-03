import numpy as np

def main():
    # Example usage
    angle = plane_azimuth_angle(zen_cam=[70,5,0], az_light=[0,55,0], zen_light=[30,60,90])
    print("Angle between the plane and vertical azimuth:", angle)

#function to correct the stokes vector for the scattering plane azimuth angle offset (angle between the plane of light target and observer relative to surface normal)
def stokes_scatter_plane_correct(I,Q,U,V,azimuth_plane):
    #convert azimuth to radians
    chi = np.radians(azimuth_plane)

    stokes = np.array([I,Q,U,V])

    # Rotation matrix for Stokes parameters
    R = np.array([
        [1, 0              ,                0, 0],
        [0,  np.cos(2 * chi), np.sin(2 * chi), 0],
        [0, -np.sin(2 * chi), np.cos(2 * chi), 0],
        [0,                0,               0, 1]
    ])

    #Functionally, the rotation matrix is equivalent to the following:
    # I_corrected = I
    # Q_corrected = Q*np.cos(2*azimuth_plane) - U*np.sin(2*azimuth_plane)
    # U_corrected = Q*np.sin(2*azimuth_plane) + U*np.cos(2*azimuth_plane)
    # V_corrected = V

    # Rotate the Stokes vector
    stokes = R @ stokes

    # Return the corrected Stokes parameters as seperate values I,Q,U,V
    return stokes[0], stokes[1], stokes[2], stokes[3]
    
def plane_azimuth_angle(zen_cam, az_light, zen_light, r_cam=1, r_light=1, az_cam=180):
    # Convert degrees to radians
    zen_light = np.radians(zen_light)
    az_light = np.radians(az_light)
    zen_cam = np.radians(zen_cam)
    az_cam = np.radians(az_cam)

    # Convert spherical coordinates to rectangular coordinates
    x_light = r_light * np.sin(zen_light) * np.cos(az_light)
    y_light = r_light * np.sin(zen_light) * np.sin(az_light)
    z_light = r_light * np.cos(zen_light)

    x_cam = r_cam * np.sin(zen_cam) * np.cos(az_cam)
    y_cam = r_cam * np.sin(zen_cam) * np.sin(az_cam)
    z_cam = r_cam * np.cos(zen_cam)

    # Calculate the normal vector of the plane formed by light and camera vectors
    normal_x = y_light * z_cam - z_light * y_cam
    normal_y = z_light * x_cam - x_light * z_cam
    normal_z = x_light * y_cam - y_light * x_cam

    # Normalize the normal vector
    normal_magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= normal_magnitude
    normal_y /= normal_magnitude
    normal_z /= normal_magnitude

    # Vertical azimuth direction (assumed as z-axis aligned vector)
    vertical_azimuth_vector = np.array([0, 0, 1])

    # Calculate dot product between normal vector and vertical azimuth vector
    dot_product = normal_z  # Since vertical azimuth vector is [0, 0, 1], only z component matters

    # Compute angle between the plane and the vertical azimuth
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    #angle of the plane is the compliment of 0-90 with respect to the normal
    angle = 90 - angle

    return np.round(angle, 4) #round to 4 decimal places

def phase_angle(zen_cam, az_light, zen_light, r_cam = 1, r_light = 1,  az_cam = 180):
    # convert degrees to radians 
    zen_light = np.radians(zen_light)
    az_light = np.radians( az_light)
    zen_cam = np.radians(zen_cam)
    az_cam = np.radians(az_cam)

    # convert spherical coordinates to rectangular coords
    x_light = r_light * np.sin(zen_light) * np.cos(az_light)
    y_light = r_light * np.sin(zen_light) * np.sin(az_light)
    z_light = r_light * np.cos(zen_light)

    x_cam  = r_cam * np.sin(zen_cam) *np.cos(az_cam)
    y_cam = r_cam * np.sin(zen_cam) * np.sin(az_cam)
    z_cam = r_cam * np.cos(zen_cam)

    # dot product 
    dot_product = x_light * x_cam + y_light * y_cam + z_light * z_cam
    mag1 = np.sqrt(x_light**2 + y_light**2 + z_light**2)
    mag2 = np.sqrt(x_cam**2 + y_cam**2+z_cam**2)
    product_mag = mag1*mag2	
    
    phase = np.degrees(np.arccos(dot_product/product_mag))

    #if phase angle is zero, the result calculated is NaN, so we set it to 0
    if np.isnan(phase):
        phase = 0; 

    return np.round(phase,4)

if __name__ == '__main__':
    main()

# import numpy as np

# def phase_angle(zen_cam, az_light, zen_light, r_cam = 1, r_light = 1,  az_cam = 180):
#     # convert degrees to radians 
#     zen_light = np.radians(zen_light)
#     az_light = np.radians( az_light)
#     zen_cam = np.radians(zen_cam)
#     az_cam = np.radians(az_cam)

#     # convert spherical coordinates to rectangular coords
#     x_light = r_light * np.sin(zen_light) * np.cos(az_light)
#     y_light = r_light * np.sin(zen_light) * np.sin(az_light)
#     z_light = r_light * np.cos(zen_light)

#     x_cam  = r_cam * np.sin(zen_cam) *np.cos(az_cam)
#     y_cam = r_cam * np.sin(zen_cam) * np.sin(az_cam)
#     z_cam = r_cam * np.cos(zen_cam)

#     # dot product 
#     dot_product = x_light * x_cam + y_light * y_cam + z_light * z_cam
#     mag1 = np.sqrt(x_light**2 + y_light**2 + z_light**2)
#     mag2 = np.sqrt(x_cam**2 + y_cam**2+z_cam**2)
#     product_mag = mag1*mag2	
    
#     phase = np.degrees(np.arccos(dot_product/product_mag))
#     return np.round(phase,4)

# def vector_average(dolp,aolp_rad):
#     """
#     input: 
#     dolp: array of dolp values 
#     aolp: array of aolp values in radians

#     output - average dolp & aolp over region    
#     """
#     # validate input
#     if aolp_rad.max() > np.pi or aolp_rad.min() < 0:
#         Exception(f"aolp values must be in radians, in [0, pi] (max is {aolp_rad.max()}, min is {aolp_rad.min()})")

#     if dolp.max() > 1 or dolp.min() < 0:
#         Exception(f"dolp values must be in [0, 1] (max is {dolp.max()}, min is {dolp.min()})")

#     #determine standard deviation of DoLP
#     # sd_dolp = np.std(dolp)

#     #convert to cartesian values
#     x_vals = dolp*np.cos(aolp_rad)
#     y_vals = dolp*np.sin(aolp_rad)

#     # calculate x & y mean
#     x_avg = np.mean(x_vals)
#     y_avg = np.mean(y_vals)

#     # convert back to polar corods
#     avg_dolp = np.sqrt(x_avg**2 + y_avg**2)
#     avg_aolp = np.arctan2(y_avg, x_avg)

#     if avg_aolp < 0: print(f'avg_aolp is negative: {avg_aolp}')
#     if avg_aolp > np.pi: print(f'avg_aolp is greater than pi: {avg_aolp}')

#     # avg_aolp must be in [0, pi]
#     if avg_aolp < 0:        avg_aolp   += np.pi
#     if avg_aolp > np.pi:    avg_aolp   -= np.pi

#     return avg_dolp, avg_aolp #, sd_dolp

# def main():
#     angles_matrix = np.array([[30, 60, 90], [120, 150, 180]])
#     magnitudes_matrix = np.array([[1, 2, 3], [4, 5, 6]])

#     # Calculate the average
#     avg_magnitude_matrix, avg_angle_matrix = vector_average(angles_matrix, magnitudes_matrix)
#     print(avg_magnitude_matrix, avg_angle_matrix)

#     phase = phase_angle(30,180,60)
#     print(phase)

# if __name__ == '__main__':
    
#     camera_zenith = 45
#     azimuth_goal_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]  
#     zenith_goal_positions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]

#     print(f"camera_az | camera_zen | light_az | light_zen | phase")

#     for i in range(len(azimuth_goal_positions)):
#         for j in range(len(zenith_goal_positions)):
#             phase = phase_angle(camera_zenith,azimuth_goal_positions[i],zenith_goal_positions[j])
#             print(f"180 | {camera_zenith} | {azimuth_goal_positions[i]} | {zenith_goal_positions[j]} | {phase} ")
            