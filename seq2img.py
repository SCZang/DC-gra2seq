import numpy as np
import cv2

def seq_3d_to_5d(stroke, max_len=250):
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1
    return result

def seq_5d_to_3d(big_stroke):
    l = 0 # the total length of the drawing
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
        if l == 0:
            l = len(big_stroke) # restrict the max total length of drawing to be the length of big_stroke
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result # stroke-3

def canvas_size_google(sketch):
    vertical_sum = np.cumsum(sketch[1:], axis=0)
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]
    start_y = -ymin - sketch[0][1]
    # sketch[0] = sketch[0] - sketch[0]
    return [int(start_x), int(start_y), int(h), int(w)]

def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=float)
    return sketch_rescale.astype("int16")

def make_graph_(sketch, seed, seed_id, graph_num=30, graph_picture_size=128,
                random_color=False, mask_prob=0.0, train=True):
    tmp_img_size = 640
    thickness = int(tmp_img_size * 0.025)
    # preprocess
    sketch = scale_sketch(sketch, (tmp_img_size, tmp_img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1

    # graph (graph_num, 3, graph_size, graph_size)
    graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size), dtype='uint8')  # must uint8

    adj_matrix = np.eye(graph_num, dtype=float) * 0.5  # (graph_num, graph_num)
    for index in range(graph_num):
        if index == 0:
            adj_matrix[0, 0] += 0.5
            continue
        # adj_matrix[index][(index + graph_num - 3) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 2) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num) % graph_num] = 0.5
        adj_matrix[index][(index + graph_num + 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num + 2) % graph_num] = 0.2
        # adj_matrix[index][(index + graph_num + 3) % graph_num] = 0.2
    adj_matrix[:, 0] += 0.5


    # canvas (h, w, 3)
    canvas = np.zeros((max(h, w) + 2 * (thickness + 1), max(h, w) + 2 * (thickness + 1)), dtype='uint8')
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (255, 255, 255)
    pen_now = np.array([start_x, start_y])
    store_pen_location = pen_now
    first_zero = False

    # generate canvas.
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) != 0:  # next stroke
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (255, 255, 255)
        pen_now += delta_x_y
        store_pen_location = np.vstack((store_pen_location, pen_now))
    # canvas_first = cv2.resize(canvas, (graph_picture_size, graph_picture_size))
    # graphs[0] = canvas_first

    # generate patch pixel picture from canvas
    # make canvas larger, enlarge canvas 100 pixels boundary
    _h, _w = canvas.shape  # (h, w, c)
    boundary_size = int(graph_picture_size * 1.5)
    top_bottom = np.zeros((boundary_size, _w), dtype=canvas.dtype)
    left_right = np.zeros((boundary_size * 2 + _h, boundary_size), dtype=canvas.dtype)
    canvas = np.concatenate((top_bottom, canvas, top_bottom), axis=0)
    canvas = np.concatenate((left_right, canvas, left_right), axis=1)
    # cv2.imwrite(f"./google_large.png", canvas)
    # processing.
    pen_now = np.array([start_x + boundary_size, start_y + boundary_size])
    first_zero = False


    # Create masked canvas
    mask_id = []
    graph_count = 0
    tmp_count = 0
    _move = graph_picture_size // 2
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move]

            if graph_count + 1 > graph_num - 1:
                break

            if train == 'train':
                applied_seed = np.random.uniform(0, 1)
            else:
                applied_seed = seed[seed_id]
                seed_id += 1

            if tmpRec.shape[0] != graph_picture_size or tmpRec.shape[1] != graph_picture_size:
                # print(f'this sketch is broken: broken stroke: ', index)
                pass
            elif applied_seed < mask_prob:
                canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move] = 0
                # cv2.rectangle(canvas,
                #               tuple(pen_now - np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               tuple(pen_now + np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               color=(255, 255, 255), thickness=1)
                mask_id.append(graph_count)


            graph_count += 1
        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y
    # cv2.imwrite("./google_large_rec.png", 255 - canvas)
    # id = np.array(id)
    # print(id)

    canvas_first = cv2.resize(canvas[boundary_size:boundary_size + _h, boundary_size:boundary_size + _w], (graph_picture_size, graph_picture_size))
    graphs[0] = canvas_first


    # generate patches.
    # strategies:
    # 1. get box at the head of one stroke
    # 2. in a long stroke, we get box in
    pen_now = np.array([start_x + boundary_size, start_y + boundary_size])
    first_zero = False
    graph_count = 0
    tmp_count = 0
    # num_strokes = math.floor(len(sketch) / (graph_num - 1))  # zsc: number of strokes for creating a single lattice
    _move = graph_picture_size // 2
    location_of_pen = []
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move]
            if graph_count + 1 > graph_num - 1:
                break
            if tmpRec.shape[0] != graph_picture_size or tmpRec.shape[1] != graph_picture_size:
                # print(f'this sketch is broken: broken stroke: ', index)
                pass
            else:
                graphs[graph_count + 1] = tmpRec
                location_of_pen.append([pen_now[1], pen_now[0]])
                # cv2.rectangle(canvas,
                #               tuple(pen_now - np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               tuple(pen_now + np.array([graph_picture_size // 2, graph_picture_size // 2])),
                #               color=(255, 255, 255), thickness=1)

            graph_count += 1
            # cv2.line(canvas, tuple(pen_now), tuple(pen_now + np.array([1, 1])), color=(0, 0, 255), thickness=3)

        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y

    for idx_i in range(graph_num):
        adj_matrix[idx_i] /= np.sum(adj_matrix[idx_i])

    graphs_tensor = np.zeros([graph_num, graph_picture_size, graph_picture_size, 1])
    # cv2.imwrite("./google_large_rec.png", 255 - canvas)
    # exit(0)
    for index in range(graph_num):
        # graphs_tensor[index] = patch_trans(graphs[index])
        graphs_tensor[index] = np.expand_dims(graphs[index] / 255 * 2 - 1, axis=2)

    # mask block
    # mask_list = [x for x in range(graph_count)]
    # mask_list.remove(0)  # remove global, prevent be masked
    # mask_number = int(mask_prob * graph_count)
    # mask_index_list = random.sample(mask_list, mask_number)
    for mask_index in mask_id:
        graphs[mask_index + 1, :] = 0
        adj_matrix[mask_index + 1, :] = 0
        adj_matrix[:, mask_index + 1] = 0
    if graph_count + 1 < graph_num:
        adj_matrix[graph_count + 1 + 1:, :] = 0
        adj_matrix[:, graph_count + 1 + 1:] = 0

    return store_pen_location, graphs_tensor, adj_matrix, graph_count, mask_id, seed_id