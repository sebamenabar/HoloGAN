import tensorflow as tf


def generate_transform_matrix(transform_params, in_size=64, out_size=64):
    """ Generate inverse of transformation matrix represented by transform params,
    so that we can get the source point which a target point looks at, and not where
    the source points would be mapped
    Application order:
        - translate center to in voxel
        - y rotation
        - x rotation
        - z rotation (temp disabled)
        - scaling
        - translation
        - inverse translate center to out voxel

    Arguments:
        transform_params {(batch_size, 9)} -- (
            azimuth angle - y axis pointing up - in radians,
            elevation angle - x axis poiting sideways - in radians,
            roll (?) angle - z axis pointing forward - in radians,
            scale x -- 1 for no scaling -- < 1 reduce size -- > 1 increase size,
            scale y -- 1 for no scaling -- < 1 reduce size -- > 1 increase size,
            scale z -- 1 for no scaling -- < 1 reduce size -- > 1 increase size,
            translation x,
            traslation y,
            translation z,
        )

    Keyword Arguments:
        in_size {int} -- [description] (default: {64})
        out_size {int} -- [description] (default: {64})
    """

    batch_size = transform_params.shape[0]
    ones = tf.ones((batch_size, 1, 1))
    zeros = tf.zeros((batch_size, 1, 1))

    azimuth = tf.reshape(transform_params[:, 0], (batch_size, 1, 1))
    elevation = tf.reshape(transform_params[:, 1], (batch_size, 1, 1))
    # z_rot = tf.reshape(transform_params[:, 2], (batch_size, 1, 1))
    # scale_x = tf.reshape(transform_params[:, 3], (batch_size, 1, 1))
    # scale_y = tf.reshape(transform_params[:, 4], (batch_size, 1, 1))
    # scale_z = tf.reshape(transform_params[:, 5], (batch_size, 1, 1))
    # tx = tf.reshape(transform_params[:, 6], (batch_size, 1, 1))
    # ty = tf.reshape(transform_params[:, 7], (batch_size, 1, 1))
    # tz = tf.reshape(transform_params[:, 8], (batch_size, 1, 1))

    _Ry = tf.concat(
        [
            tf.concat([tf.cos(azimuth), zeros, -tf.sin(azimuth), zeros], axis=2),
            tf.concat([zeros, ones, zeros, zeros], axis=2),
            tf.concat([tf.sin(azimuth), zeros, tf.cos(azimuth), zeros], axis=2),
            tf.concat([zeros, zeros, zeros, ones], axis=2),
        ],
        axis=1,
    )
    # Batch Rotation X matrixes
    _Rx = tf.concat(
        [
            tf.concat([tf.cos(elevation), tf.sin(elevation), zeros, zeros], axis=2),
            tf.concat([-tf.sin(elevation), tf.cos(elevation), zeros, zeros], axis=2),
            tf.concat([zeros, zeros, ones, zeros], axis=2),
            tf.concat([zeros, zeros, zeros, ones], axis=2),
        ],
        axis=1,
    )

    # S = tf.concat(
    #     [
    #         tf.concat([scale_x, zeros, zeros, zeros], axis=2),
    #         tf.concat([zeros, scale_y, zeros, zeros], axis=2),
    #         tf.concat([zeros, zeros, scale_z, zeros], axis=2),
    #         tf.concat([zeros, zeros, zeros, ones], axis=2),
    #     ],
    #     axis=1,
    # )

    # T = tf.concat(
    #     [
    #         tf.concat([ones, zeros, zeros, tx], axis=2),
    #         tf.concat([zeros, ones, zeros, ty], axis=2),
    #         tf.concat([zeros, zeros, ones, tz], axis=2),
    #         tf.concat([zeros, zeros, zeros, ones], axis=2),
    #     ],
    #     axis=1,
    # )

    # Co = tf.constant(
    #     [
    #         [1, 0, 0, -in_size * 0.5],
    #         [0, 1, 0, -in_size * 0.5],
    #         [0, 0, 1, -in_size * 0.5],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # Co = tf.tile(
    #     tf.reshape(Co, (1, 4, 4)), [batch_size, 1, 1]
    # )

    # Dn = tf.constant(
    #     [
    #         [1, 0, 0, out_size * 0.5],
    #         [0, 1, 0, out_size * 0.5],
    #         [0, 0, 1, out_size * 0.5],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # Dn = tf.tile(
    #     tf.reshape(Dn, (1, 4, 4)), [batch_size, 1, 1]
    # )

    # M = tf.matmul(
    #     tf.matmul(
    #         tf.matmul(
    #             tf.matmul(tf.matmul(Dn, T), S),
    #             Rx,
    #         ),
    #         Ry,
    #     ),
    #     Co,
    # )
    # M * (x, y, z) = Dn * T * S * Rx * Ry * Co * (x, y, z) = (x', y' z')
    # (x, y, z) = (Co)^-1 * (Ry)^-1 * (Rx)^-1 * (S)^-1 * (T)^-1 * (Dn)^-1 * (x', y' z')
    # Instead of calculating the inverse of the whole transformation M,
    # we can analitically (also differentiably)
    # compute the inverse of each individual transformation.

    Ry_inv = tf.transpose(_Ry, perm=[0, 2, 1])  # Inverse of rotation is its transpose
    Rx_inv = tf.transpose(_Rx, perm=[0, 2, 1])

    # Inverse of scale is 1/scale
    scale_x_inv = tf.reshape(1 / transform_params[:, 3], (batch_size, 1, 1))
    scale_y_inv = tf.reshape(1 / transform_params[:, 4], (batch_size, 1, 1))
    scale_z_inv = tf.reshape(1 / transform_params[:, 5], (batch_size, 1, 1))
    # Inverse of translation is the negative translation
    tx_inv = tf.reshape(-transform_params[:, 6], (batch_size, 1, 1))
    ty_inv = tf.reshape(-transform_params[:, 7], (batch_size, 1, 1))
    tz_inv = tf.reshape(-transform_params[:, 8], (batch_size, 1, 1))

    S_inv = tf.concat(
        [
            tf.concat([scale_x_inv, zeros, zeros, zeros], axis=2),
            tf.concat([zeros, scale_y_inv, zeros, zeros], axis=2),
            tf.concat([zeros, zeros, scale_z_inv, zeros], axis=2),
            tf.concat([zeros, zeros, zeros, ones], axis=2),
        ],
        axis=1,
    )

    T_inv = tf.concat(
        [
            tf.concat([ones, zeros, zeros, tx_inv], axis=2),
            tf.concat([zeros, ones, zeros, ty_inv], axis=2),
            tf.concat([zeros, zeros, ones, tz_inv], axis=2),
            tf.concat([zeros, zeros, zeros, ones], axis=2),
        ],
        axis=1,
    )

    Co_inv = tf.constant(
        [
            [1, 0, 0, in_size * 0.5],
            [0, 1, 0, in_size * 0.5],
            [0, 0, 1, in_size * 0.5],
            [0, 0, 0, 1],
        ]
    )
    Co_inv = tf.tile(tf.reshape(Co_inv, (1, 4, 4)), [batch_size, 1, 1])

    Dn_inv = tf.constant(
        [
            [1, 0, 0, -out_size * 0.5],
            [0, 1, 0, -out_size * 0.5],
            [0, 0, 1, -out_size * 0.5],
            [0, 0, 0, 1],
        ]
    )
    Dn_inv = tf.tile(tf.reshape(Dn_inv, (1, 4, 4)), [batch_size, 1, 1])

    # (x, y, z) = (Co)^-1 * (Ry)^-1 * (Rx)^-1 * (S)^-1 * (T)^-1 * (Dn)^-1 * (x', y' z')
    M_inv = tf.matmul(
        tf.matmul(
            tf.matmul(tf.matmul(tf.matmul(Co_inv, Ry_inv), Rx_inv), S_inv,), T_inv,
        ),
        Dn_inv,
    )
    M_inv = M_inv[
        :, 0:3, :
    ]  # Ignore the homogenous coordinate so the results are 3D vectors

    return M_inv


def tf_voxel_meshgrid(height, width, depth, homogeneous=False):
    # Because 'ij' ordering is used for meshgrid, z_t and x_t are swapped (Think about order in 'xy' VS 'ij'
    z_t, y_t, x_t = tf.meshgrid(
        tf.range(depth, dtype=tf.float32),
        tf.range(height, dtype=tf.float32),
        tf.range(width, dtype=tf.float32),
        indexing="ij",
    )
    # Reshape into a big list of slices one after another along the X,Y,Z direction
    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))
    z_t_flat = tf.reshape(z_t, (1, -1))

    # Vertical stack to create a (3,N) matrix for X,Y,Z coordinates
    grid = tf.concat([x_t_flat, y_t_flat, z_t_flat], axis=0)
    if homogeneous:
        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([grid, ones], axis=0)
    return grid


def tf_repeat(x, n_repeats):
    # Repeat X for n_repeats time along 0 axis
    # Return a 1D tensor of total number of elements
    rep = tf.ones(shape=[1, n_repeats], dtype="int32")
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])


def tf_interpolate(voxel, x, y, z, out_size):
    """
    Trilinear interpolation for batch of voxels
    :param voxel: The whole voxel grid
    :param x,y,z: indices of voxel
    :param output_size: output size of voxel
    :return:
    """
    batch_size = tf.shape(voxel)[0]
    height = tf.shape(voxel)[1]
    width = tf.shape(voxel)[2]
    depth = tf.shape(voxel)[3]
    n_channels = tf.shape(voxel)[4]

    x = tf.cast(x, "float32")
    y = tf.cast(y, "float32")
    z = tf.cast(z, "float32")

    out_height = out_size[1]
    out_width = out_size[2]
    out_depth = out_size[3]
    out_channel = out_size[4]

    zero = tf.zeros([], dtype="int32")
    max_y = tf.cast(height - 1, "int32")
    max_x = tf.cast(width - 1, "int32")
    max_z = tf.cast(depth - 1, "int32")

    # do sampling
    x0 = tf.cast(tf.floor(x), "int32")
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), "int32")
    y1 = y0 + 1
    z0 = tf.cast(tf.floor(z), "int32")
    z1 = z0 + 1

    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)
    z0 = tf.clip_by_value(z0, zero, max_z)
    z1 = tf.clip_by_value(z1, zero, max_z)

    # A 1D tensor of base indicies describe First index for each shape/map in the whole batch
    # tf.range(batch_size) * width * height * depth : Element to repeat. Each selement in the list is incremented by width*height*depth amount
    # out_height * out_width * out_depth: n of repeat. Create chunks of out_height*out_width*out_depth length with the same value created by tf.rage(batch_size) *width*height*dept
    base = tf_repeat(
        tf.range(batch_size) * width * height * depth,
        out_height * out_width * out_depth,
    )

    # Find the Z element of each index

    base_z0 = base + z0 * width * height
    base_z1 = base + z1 * width * height
    # Find the Y element based on Z
    base_z0_y0 = base_z0 + y0 * width
    base_z0_y1 = base_z0 + y1 * width
    base_z1_y0 = base_z1 + y0 * width
    base_z1_y1 = base_z1 + y1 * width

    # Find the X element based on Y, Z for Z=0
    idx_a = base_z0_y0 + x0
    idx_b = base_z0_y1 + x0
    idx_c = base_z0_y0 + x1
    idx_d = base_z0_y1 + x1
    # Find the X element based on Y,Z for Z =1
    idx_e = base_z1_y0 + x0
    idx_f = base_z1_y1 + x0
    idx_g = base_z1_y0 + x1
    idx_h = base_z1_y1 + x1

    # use indices to lookup pixels in the flat image and restore
    # channels dim
    voxel_flat = tf.reshape(voxel, [-1, n_channels])
    voxel_flat = tf.cast(voxel_flat, "float32")
    Ia = tf.gather(voxel_flat, idx_a)
    Ib = tf.gather(voxel_flat, idx_b)
    Ic = tf.gather(voxel_flat, idx_c)
    Id = tf.gather(voxel_flat, idx_d)
    Ie = tf.gather(voxel_flat, idx_e)
    If = tf.gather(voxel_flat, idx_f)
    Ig = tf.gather(voxel_flat, idx_g)
    Ih = tf.gather(voxel_flat, idx_h)

    # and finally calculate interpolated values
    x0_f = tf.cast(x0, "float32")
    x1_f = tf.cast(x1, "float32")
    y0_f = tf.cast(y0, "float32")
    y1_f = tf.cast(y1, "float32")
    z0_f = tf.cast(z0, "float32")
    z1_f = tf.cast(z1, "float32")

    # First slice XY along Z where z=0
    wa = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z1_f - z)), 1)
    wb = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z1_f - z)), 1)
    wc = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z1_f - z)), 1)
    wd = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z1_f - z)), 1)
    # First slice XY along Z where z=1
    we = tf.expand_dims(((x1_f - x) * (y1_f - y) * (z - z0_f)), 1)
    wf = tf.expand_dims(((x1_f - x) * (y - y0_f) * (z - z0_f)), 1)
    wg = tf.expand_dims(((x - x0_f) * (y1_f - y) * (z - z0_f)), 1)
    wh = tf.expand_dims(((x - x0_f) * (y - y0_f) * (z - z0_f)), 1)

    output = tf.add_n(
        [wa * Ia, wb * Ib, wc * Ic, wd * Id, we * Ie, wf * If, wg * Ig, wh * Ih]
    )
    return output


def tf_3d_transform(voxel_array, transform_params, in_size=64, out_size=64):
    batch_size = voxel_array.shape[0]
    n_channels = voxel_array.shape[4]
    M = generate_transform_matrix(transform_params, in_size, out_size)
    grid = tf_voxel_meshgrid(out_size, out_size, out_size, homogeneous=True)
    grid = tf.tile(
        tf.reshape(grid, (1, grid.shape[0], grid.shape[1]),), [batch_size, 1, 1],
    )
    grid_transform = tf.matmul(M, grid)
    x_s_flat = tf.reshape(grid_transform[:, 0, :], [-1])
    y_s_flat = tf.reshape(grid_transform[:, 1, :], [-1])
    z_s_flat = tf.reshape(grid_transform[:, 2, :], [-1])
    input_transformed = tf_interpolate(
        voxel_array,
        x_s_flat,
        y_s_flat,
        z_s_flat,
        [batch_size, out_size, out_size, out_size, n_channels],
    )
    target = tf.reshape(
        input_transformed, [batch_size, out_size, out_size, out_size, n_channels]
    )

    return target


def transform_voxel_to_match_image(tensor):
    tensor = tf.transpose(tensor, [0, 2, 1, 3, 4])
    tensor = tensor[:, ::-1, :, :, :]
    return tensor
