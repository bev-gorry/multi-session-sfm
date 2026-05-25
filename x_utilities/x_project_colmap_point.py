def project_point_colmap(xyz_world, image, camera):
    R = image.qvec2rotmat()
    t = image.tvec.reshape(3, 1)

    xyz_cam = R @ xyz_world.reshape(3, 1) + t
    X, Y, Z = xyz_cam.flatten()

    if Z <= 0:
        return None

    params = camera.params

    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]

    elif camera.model == "PINHOLE":
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]
    
    elif camera.model == "SIMPLE_RADIAL":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
        k1 = params[3]

        # normalized coordinates
        x = X / Z
        y = Y / Z

        r2 = x*x + y*y
        r4 = r2 * r2

        # radial distortion
        x_dist = x * (1 + k1*r2)
        y_dist = y * (1 + k1*r2)

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        return u, v

    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params

        # normalized coordinates
        x = X / Z
        y = Y / Z

        r2 = x*x + y*y
        r4 = r2 * r2

        # radial + tangential distortion
        x_dist = x * (1 + k1*r2 + k2*r4) + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y * (1 + k1*r2 + k2*r4) + p1*(r2 + 2*y*y) + 2*p2*x*y

        u = fx * x_dist + cx
        v = fy * y_dist + cy

        return u, v

    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported")

    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    return u, v