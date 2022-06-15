import os
import pickle

import cv2
from PIL import Image
import numpy as np
import pickle


def plot_pose(img, result, scale=(1.0, 1.0)):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)
    PINK = (203, 192, 255)
    unshown_pts = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    l_pair = [  # v3 web
        # coco pose
        #
        (5, 7),# R shoulder - R elbow
        (7, 9),# R elbow - R wrist

        (6, 8), # M shoulder - L shoulder
        (8, 10),# L shoulder - L elbow
        (5, 6),# L elbow - L wrist

        # ====================================
        # =========feet ===================

        # ========================================
        # =========face===========================
        (23,24),#contour
        (24,25),
        (25,26),
        (26,27),
        (27,28),
        (28,29),
        (29,30),
        (30,31),
        (31,32),
        (32,33),
        (33,34),
        (34,35),
        (35,36),
        (36,37),
        (37,38),
        (38,39),

        (40,41),#R eye brows
        (41,42),
        (42,43),
        (43,44),
        (45,46),#L eye brows
        (46,47),
        (47,48),
        (48,49),

        (50,51), #nose
        (51,52),
        (52, 53),
        (54,55),
        (55,56),
        (56,57),
        (57,58),

                (59,60),# R eye
                (60,61),
                (61,62),
                (62,63),
                (63,64),
                (64,59),
                (65,66),#L eye
                (66,67),
                (67,68),
                (68,69),
                (69,70),
                (70,65),

                (71,72),#out mouth
                (72,73),
                (73,74),
                (74,75),
                (75,76),
                (76,77),
                (77,78),
                (78,79),
                (79,80),
                (80,81),
                (81,82),
                (82,71),

                (83,84),#in mouth
                (84,85),
                (85,86),
                (86,87),
                (87,88),
                (88,89),
                (89,90),
                (90,83),
        #==================================================
        #=============hands================================
                (9,91),#L wrist - L hand
                (91,92),# L thumb
                (92,93),
                (93,94),
                (94,95),
                (91,96),#L index finger
                (96,97),
                (97,98),
                (98,99),
                (91,100), #L mid finger
                (100,101),
                (101,102),
                (102,103),
                (91,104), # L ring finger
                (104,105),
                (105,106),
                (106,107),
                (91,108), #L little finger
                (108,109),
                (109,110),
                (110,111),

                (10,112), # R wrist - R hand
                (112,113),#R thumb
                (113,114),
                (114,115),
                (115,116),
                (112,117),#R index finger
                (117,118),
                (118,119),
                (119,120),
                (112,121),#R mid finger
                (121,122),
                (122,123),
                (123,124),
                (112,125),#R ring finger
                (125,126),
                (126,127),
                (127,128),
                (112,129), #R little finger
                (129,130),
                (130,131),
                (131,132),
    ]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
               YELLOW, ORANGE, PINK, RED, PINK, RED, PINK, RED]
    p_color = [GREEN, BLUE, BLUE, BLUE, BLUE, YELLOW, ORANGE, YELLOW, ORANGE,
               YELLOW, ORANGE, PINK, RED, PINK, RED, PINK]

    part_line = {}
    kp_preds = result[:, :2]
    kp_scores = result[:, 2]

    # Draw keypoints
    for n in range(kp_scores.shape[0]):
        if kp_scores[n] <= 0.1 or n in unshown_pts:
            continue
        cor_x, cor_y = int(round(kp_preds[n, 0] * scale[0])), int(round(kp_preds[n, 1] * scale[1]))
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 2, (0,255,255), -1)
    # Draw limbs
    for start_p, end_p in l_pair:
        if start_p in part_line and end_p in part_line:
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, (0,255,255), 2)
    return img

# 测试单张图像 和 单个pkl文件使用
def imageCrop(pkl_pth, img_pth):
    f_pkl = open(pkl_pth, 'rb')
    data = pickle.load(f_pkl)
    print("data.size:", data.shape)

    # 存储pose移动距离的矩阵
    dist = np.zeros(data.shape, dtype=float) # [frame_num, 133, 3]

    cor_x = data[4, 53, :][0]
    cor_y = data[4, 53, :][1]
    img = cv2.imread(img_pth) # [512, 512, 3]
    cv2.circle(img, (int(cor_x), int(cor_y)), 2, (0,256,256), -1)
    # (0,0)图像左上角
    # cv2.circle(img, (0, 0), 2, (0, 255, 255), -1)
    cv2.imwrite("/root/LC/SSLData/test.png", img)
    ################################
    theory_left_dist = img.shape[0] // 2
    theory_top_dist = 150
    left_dist = cor_x
    top_dist = cor_y
    img2 = Image.open(img_pth)
    crop_comp_dist_left = abs(left_dist - theory_left_dist)
    crop_comp_dist_top = abs(top_dist - theory_top_dist)
    max_width = int(img2.size[0] + crop_comp_dist_left)
    max_height = int(img2.size[1] + crop_comp_dist_top)
    # new_img = Image.new('RGB', (max_dist, max_dist), (70, 136, 203)) # (70, 136, 203)
    new_img = Image.new('RGB', (max_width, max_height), color="red")

    if left_dist < theory_left_dist:
        if top_dist < theory_top_dist: # complement left and top
            # RGB image alignment
            new_img.paste(img2, (int(crop_comp_dist_left), int(crop_comp_dist_top)))
            crop_img = new_img.crop((0, 0, img2.size[0], img2.size[0]))
            crop_img.save("/root/LC/SSLData/test_left_top_crop.png")
            # joint coordinate alignment
            left_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_left)
            top_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_top)
            dist[4, :, 0:1] = left_dist_arr
            dist[4, :, 1:2] = top_dist_arr

            # joint可视化
            # data[4, :, :] = data[4, :, :] + dist[4, :, :]
            # img_adj = cv2.imread("/root/LC/SSLData/test_left_top_crop.png")
            # img_joint_vis = plot_pose(img_adj, data[4, :, :])
            # cv2.imwrite("/root/LC/SSLData/img_joint_vis.png", img_joint_vis)

        else: # complement left and bottom
            # RGB image alignment
            new_img.paste(img2, (int(crop_comp_dist_left), int(-crop_comp_dist_top)))
            crop_img = new_img.crop((0, 0, img2.size[0], img2.size[0]))
            crop_img.save("/root/LC/SSLData/test_left_bottom_crop.png")
            # joint coordinate alignment
            left_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_left)
            top_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_top)
            dist[4, :, 0:1] = left_dist_arr
            dist[4, :, 1:2] = top_dist_arr

            # joint可视化
            # data[4, :, :] = data[4, :, :] + dist[4, :, :]
            # img_adj = cv2.imread("/root/LC/SSLData/test_left_bottom_crop.png")
            # img_joint_vis = plot_pose(img_adj, data[4, :, :])
            # cv2.imwrite("/root/LC/SSLData/img_joint_vis.png", img_joint_vis)

    else: # left_dist > theory_left_dist:
        if top_dist < theory_top_dist: # complement right and top
            # RGB image alignment
            new_img.paste(img2, (int(-crop_comp_dist_left), int(crop_comp_dist_top)))
            crop_img = new_img.crop((0, 0, img2.size[0], img2.size[0]))
            crop_img.save("/root/LC/SSLData/test_right_top_crop.png")
            # joint coordinate alignment
            left_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_left)
            top_dist_arr  =np.full((dist.shape[1], 1), crop_comp_dist_top)
            dist[4, :, 0:1] = left_dist_arr
            dist[4, :, 1:2] = top_dist_arr

            # joint可视化
            # data[4, :, :] = data[4, :, :] + dist[4, :, :]
            # img_adj = cv2.imread("/root/LC/SSLData/test_right_top_crop.png")
            # img_joint_vis = plot_pose(img_adj, data[4, :, :])
            # cv2.imwrite("/root/LC/SSLData/img_joint_vis.png", img_joint_vis)

        else: # complement right and bottom
            # RGB image alignment
            new_img.paste(img2, (int(-crop_comp_dist_left), int(-crop_comp_dist_top)))
            crop_img = new_img.crop((0, 0, img2.size[0], img2.size[0]))
            crop_img.save("/root/LC/SSLData/test_right_bottom_crop.png")
            # joint coordinate alignment
            left_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_left)
            top_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_top)
            dist[4, :, 0:1] = left_dist_arr
            dist[4, :, 1:2] = top_dist_arr

            # joint可视化
            # data[4, :, :] = data[4, :, :] + dist[4, :, :]
            # img_adj = cv2.imread("/root/LC/SSLData/test_right_bottom_crop.png")
            # img_joint_vis = plot_pose(img_adj, data[4, :, :])
            # cv2.imwrite("/root/LC/SSLData/img_joint_vis.png", img_joint_vis)


def img_pose_normlizaion(inp_RGB_dir, inp_pkl_pth, out_RGB_dir, out_pkl_pth, is_new_joint_visualization=True):
    img_name_list = sorted(os.listdir(inp_RGB_dir))
    for name in img_name_list:
        print("processing:", out_RGB_dir+"/"+name)
        num = int(name.split(".")[0])
        f_pkl = open(inp_pkl_pth, 'rb')
        data = pickle.load(f_pkl)
        # the pose moving distant
        dist = np.zeros(data.shape, dtype=float)  # [frame_num, 133, 3]
        img_pth = os.path.join(inp_RGB_dir, name)
        img = Image.open(img_pth)

        cor_x = data[num, 53, :][0]
        cor_y = data[num, 53, :][1]
        theory_left_dist = img.size[0] // 2
        theory_top_dist = 150
        left_dist = cor_x
        top_dist = cor_y

        crop_comp_dist_left = abs(left_dist - theory_left_dist)
        crop_comp_dist_top = abs(top_dist - theory_top_dist)
        max_width = int(img.size[0] + crop_comp_dist_left)
        max_height = int(img.size[1] + crop_comp_dist_top)
        new_img = Image.new('RGB', (max_width, max_height), (70, 136, 203))

        if left_dist < theory_left_dist:
            if top_dist < theory_top_dist:  # complement left and top
                # RGB image alignment
                new_img.paste(img, (int(crop_comp_dist_left), int(crop_comp_dist_top)))
                crop_img = new_img.crop((0, 0, img.size[0], img.size[1]))
                crop_img.save(os.path.join(out_RGB_dir, name))
                # joint coordinate alignment
                left_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_left)
                top_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_top)
                dist[num, :, 0:1] = left_dist_arr
                dist[num, :, 1:2] = top_dist_arr

                print("complement left and top")


            else:  # complement left and bottom
                # RGB image alignment
                new_img.paste(img, (int(crop_comp_dist_left), int(-crop_comp_dist_top)))
                crop_img = new_img.crop((0, 0, img.size[0], img.size[1]))
                crop_img.save(os.path.join(out_RGB_dir, name))
                # joint coordinate alignment
                left_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_left)
                top_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_top)
                dist[num, :, 0:1] = left_dist_arr
                dist[num, :, 1:2] = top_dist_arr

                print("complement left and bottom")

        else:  # left_dist > theory_left_dist:
            if top_dist < theory_top_dist:  # complement right and top
                # RGB image alignment
                new_img.paste(img, (int(-crop_comp_dist_left), int(crop_comp_dist_top)))
                crop_img = new_img.crop((0, 0, img.size[0], img.size[1]))
                crop_img.save(os.path.join(out_RGB_dir, name))
                # joint coordinate alignment
                left_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_left)
                top_dist_arr = np.full((dist.shape[1], 1), crop_comp_dist_top)
                dist[num, :, 0:1] = left_dist_arr
                dist[num, :, 1:2] = top_dist_arr
                print("complement right and top")

            else:  # complement right and bottom
                # RGB image alignment
                new_img.paste(img, (int(-crop_comp_dist_left), int(-crop_comp_dist_top)))
                crop_img = new_img.crop((0, 0, img.size[0], img.size[1]))
                crop_img.save(os.path.join(out_RGB_dir, name))
                # joint coordinate alignment
                left_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_left)
                top_dist_arr = np.full((dist.shape[1], 1), -crop_comp_dist_top)
                dist[num, :, 0:1] = left_dist_arr
                dist[num, :, 1:2] = top_dist_arr
                print("complement right and bottom")

        data[num, :, :] = data[num, :, :] + dist[num, :, :]
        pickle.dump(data, open(out_pkl_pth, 'wb'))

        if is_new_joint_visualization:
            img_adj = cv2.imread(os.path.join(out_RGB_dir, name))
            img_joint_vis = plot_pose(img_adj, data[num, :, :])
            cv2.imwrite(os.path.join(out_RGB_dir, name), img_joint_vis)

if __name__ == '__main__':
    # f = open("/root/SSL/data/data_sorted_pose_pkl/051/00/p06_n061_66320_66640/0.pkl", 'rb')
    # data = pickle.load(f)
    # print(data)
    # imageCrop("/root/LC/SSLData/data_sorted_pose_pkl/013/00/p04_n046_59320_60080/0.pkl", "/root/LC/SSLData/data_sorted_posevis_pkl/013/00/p04_n046_59320_60080/4.png")
    # imageCrop("/root/LC/SSLData/data_sorted_pose_pkl/060/00/p02_n079_62360_62560/0.pkl","/root/LC/SSLData/data_sorted_posevis_pkl/060/00/p02_n079_62360_62560/4.png")
    # imageCrop("/root/LC/SSLData/data_sorted_pose_pkl/046/00/p01_n023_33560_34240/0.pkl",
    #           "/root/LC/SSLData/data_sorted_posevis_pkl/046/00/p01_n023_33560_34240/4.png")
    # imageCrop("/root/LC/SSLData/data_sorted_pose_pkl/036/00/p05_n006_114360_115040/0.pkl",
    #           "/root/LC/SSLData/data_sorted_posevis_pkl/036/00/p05_n006_114360_115040/4.png")

    input_pth_RGB = "/root/LC/SSLData/data_sorted_posevis_pkl/"
    input_pth_pkl = "/root/LC/SSLData/data_sorted_pose_pkl/"
    output_vis_pths = "/root/LC/SSLData/data_sorted_posevis_pkl_norm/"
    output_pth_pkl = "/root/LC/SSLData/data_sorted_pose_pkl_norm/"

    is_new_joint_visualization = False

    id_list = sorted(os.listdir(input_pth_RGB))
    for id in id_list:
        inp_pth_RGB_id = os.path.join(input_pth_RGB, id)
        inp_pth_RGB_id00 = inp_pth_RGB_id + '/00/'
        inp_pth_PKL_id = os.path.join(input_pth_pkl, id)
        inp_pth_PKL_id00 = inp_pth_PKL_id + '/00/'
        out_pth_PKL_id = os.path.join(output_pth_pkl, id)
        out_pth_PKL_id00 = out_pth_PKL_id + '/00/'
        out_RGB_pth_id = os.path.join(output_vis_pths, id)
        out_RGB_pth_id00 = out_RGB_pth_id + '/00/'

        name_list = sorted(os.listdir(inp_pth_RGB_id00))

        for img_name in name_list:
            inp_RGB_pth_list = os.path.join(inp_pth_RGB_id00, img_name)
            inp_pkl_pth = os.path.join(inp_pth_PKL_id00, img_name) + '/0.pkl'

            out_RGB_pth_list = os.path.join(out_RGB_pth_id00, img_name)
            isExists = os.path.exists(out_RGB_pth_list)
            if not isExists:
                os.makedirs(out_RGB_pth_list)

            out_pkl_pth = os.path.join(out_pth_PKL_id00, img_name)
            isExists = os.path.exists(out_pkl_pth)
            if not isExists:
                os.makedirs(out_pkl_pth)
            out_pkl_pth = out_pkl_pth + '/0.pkl'
            img_pose_normlizaion(inp_RGB_pth_list, inp_pkl_pth, out_RGB_pth_list, out_pkl_pth, is_new_joint_visualization=True)

