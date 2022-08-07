
import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling import models

from modeling.models.SLR_pose3D_inference import SLR_pose3D_inference
from modeling.models.SLR_Pose_inference import SLR_Pose_inference
from modeling.models.i3d_mlp_inference import i3d_mlp_inference

from tool import config_loader, generate_RGB_pkl
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def cuda_dist(x, y, metric='euc'):
    # x = torch.from_numpy(x).cuda()
    # y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=2)  # n p c
        y = F.normalize(y, p=2, dim=2)  # n p c
    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, i, ...]
        _y = y[:, i, ...]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                1).transpose(0, 1) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin



def extract_npy(dir_path, total = 2000):
    file_list = sorted(os.listdir(dir_path))
    # print(file_list)
    i = 1
    npy_list = []
    name_list = []
    for name in file_list:
        name_list.append(name)
        if i > total:
            break
        i = i + 1
        if i%100 == 0:
            print('-current load-',i)
        # print(name)
        dir_each = os.path.join(dir_path, name)
        frame_list = os.listdir(dir_each)
        frame_list.sort()
        for npy in frame_list:
            frame_each = os.path.join(dir_each, npy)
            f = open(frame_each, 'rb')
            data = pickle.load(f)
            npy_list.append(data)

    return npy_list,name_list

def extract_npy_i(dir_path,name):
    dir_each = os.path.join(dir_path, name)
    frame_list = os.listdir(dir_each)
    frame_list.sort()
    for npy in frame_list:
        frame_each = os.path.join(dir_each, npy)
        f = open(frame_each, 'rb')
        data = pickle.load(f)

    return data,name

class BasePosejointTransform2D():
    def __init__(self,):
        self.selected = {
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
    '39': np.concatenate(([9,13,16,18,20,14,17,19,21], [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], #39
                    [39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]), axis=0)
    }
    max_body_true = 1
    max_frame = 150
    num_channels = 3
    num_channels
    def __call__(self, x):
        x = x[:,self.selected['27'],:]
        x = x[:,:,:,np.newaxis]
        x = np.transpose(x,(2,0,1,3))
        return x 
class BasePosejointTransform3D():
    def __init__(self,):
        self.selected = {
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0), #27
    '39': np.concatenate(([9,13,16,18,20,14,17,19,21], [24,25,26,27,28,29,30,31,32,33,34,35,36,37,38], #39
                    [39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]), axis=0)
    }
    max_body_true = 1
    max_frame = 150
    num_channels = 3
    num_channels
    def __call__(self, x):
        x = x[:,self.selected['39'],:]
        x = x[:,:,:,np.newaxis]
        x = np.transpose(x,(2,0,1,3))
        return x 
class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        # if np.size(x,2)!=720 or np.size(x,3)!=1280:
        #     x = np.resize(x,(np.size(x,0),np.size(x,1),720,1280))

        # x = x[:,:,:,220:940]
        # x = x[:,:,:,160:880]
        input_size = 512
        output_size = 256
        bin_size = input_size // output_size
        small_image = x.reshape((len(x),3, output_size, bin_size, 
                                            output_size, bin_size)).max(5).max(3)

        return (small_image - self.mean) / self.std
def loadtopk(path, tk=0):
    f = open(path+'video_id_'+str(tk)+'.pkl', 'rb')
    video_id = pickle.load(f)
    print(path+'video_id_'+str(tk)+'.pkl',len(video_id))

    f = open(path+'video_name_'+str(tk)+'.pkl', 'rb')
    video_name = pickle.load(f)

    f = open(path+'video_starttime_'+str(tk)+'.pkl', 'rb')
    video_starttime = pickle.load(f)

    f = open(path+'video_endtime_'+str(tk)+'.pkl', 'rb')
    video_endtime = pickle.load(f)

    f = open(path+'video_dist_'+str(tk)+'.pkl', 'rb')
    video_dist = pickle.load(f)


    video_dist, video_id, video_starttime, video_endtime, video_name  = zip(*sorted(zip(video_dist, video_id, video_starttime, video_endtime, video_name)))
    # print(video_id)
    video_dist, video_id, video_starttime, video_endtime, video_name = list(video_dist), list(video_id), list(video_starttime), list(video_endtime), list(video_name)
    return video_dist, video_id, video_starttime, video_endtime, video_name
    
def generate_pkl(dir_path, save_path):
    file_list = sorted(os.listdir(dir_path))
    print(file_list)
    count = 1
    for name in file_list:
        dir_each = os.path.join(dir_path, name)
        frame_list = os.listdir(dir_each)
        frame_list.sort()
        print(count, name,len(frame_list))
        all_imgs = []
        for frame in frame_list:
            frame_each = os.path.join(dir_each, frame)
            # print(frame_each)
            img = cv2.imread(frame_each)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img.transpose(2,0,1)
            if np.size(img,1)!=720 or np.size(img,2)!=1280:
                img = np.resize(img,(np.size(img,0),720,1280))
            all_imgs.append(torch.from_numpy(img))
            if len(all_imgs)%100 ==0:
                print('-len(frame) = ',len(all_imgs))

            # print(img.shape)
        # print('-----------1---------------')
        # all_imgs = np.asarray(all_imgs)
        all_imgs = torch.stack(all_imgs).numpy()
        # print(all_imgs.shape)
        # print('-----------2---------------')
        save_path_each = os.path.join(save_path, name)
        isExists = os.path.exists(save_path_each)
        print(all_imgs.shape)
        if not isExists:
            os.makedirs(save_path_each)
            all_imgs_pkl = os.path.join(save_path_each, '0.pkl')
            pickle.dump(all_imgs, open(all_imgs_pkl, 'wb')) 
        count = count + 1

if __name__ == "__main__":
    Delete = [
          [65,8,39],#20
          [59,10,43],#21
          [53,31,49],#22
          [66,13,45],#23
          [39,1,34],#24
          [48,10,43],#25
          [77,14,62],#26
          [56,16,47],#27
          [47,4,41],#28
          [45,5,38],#29
          [48,8,43],#30
          [48,6,41],#31
          [54,7,47],#32
          [47,4,46],#33
          [51,4,40],#34
          [74,5,41],#35
          [56,2,50],#36
          [60,3,54],#37
          [63,5,51],#38
          [50,4,44] #39
          ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str,
                        default='./config/SLR_Pose.yaml', help="path of config file")
                        
    parser.add_argument('--model_path1', type=str,
                        default="./output/SLR_Pose_all_adjustlr-00240.pt", help="path of model param")
    parser.add_argument('--model_path2', type=str,
                        default="./output/SLR_pose3D_all-00190.pt", help="path of model param")
    parser.add_argument('--model_path3', type=str,
                        default="./output/i3d_mlp-00100.pt", help="path of model param")
    # parser.add_argument('--model_path3', type=str,
                        # default="/fuxi_cv/LC/1StagePretrainModel/lr0.1_normRGB256/checkpoints/i3d_mlp-00200.pt", help="path of model param")
    # parser.add_argument("--predictions_save_path", default='/root/SSL/predictions_track2_ensemble_3D_2D_adjustlr_test_all/SLR_16frames_240_190_manualBEIBEIv2/', type=str)
    parser.add_argument("--predictions_save_path", default='./predictions/', type=str)
    parser.add_argument("--topk_save_path", default='./topk', type=str)

    parser.add_argument('--gallery_pkl2D', type=str,
                        default='/fuxi_cv/SSL/original_2stage_data/OSLWL_QUERY_TEST_SET_posepkl/', help="path of gallery pkl")    
    parser.add_argument('--val_pkl2D', type=str,
                        default="/fuxi_cv/SSL/original_2stage_data/OSLWL_Test_Set_posepkl/", help="path of val pkl")    

    parser.add_argument('--gallery_pkl3D', type=str,
                        default='/data/SSL/SSL/data/OSLWL_QUERY_TEST_SET_sorted_3Dposepkl_v2/', help="path of gallery pkl")    
    parser.add_argument('--val_pkl3D', type=str,
                        default='/data/SSL/SSL/data/OSLWL_Query_Set-and-OSLWL_Test_Set_3Dposepkl_v2/', help="path of val pkl")

    parser.add_argument('--gallery_pklRGB', type=str,
                        default='/fuxi_cv/SSL/original_2stage_data/OSLWL_QUERY_TEST_SET_sorted_frames_Resize_normpkl/', help="path of gallery pkl")  
    parser.add_argument('--val_pklRGB', type=str,
                        default='/fuxi_cv/SSL/original_2stage_data/OSLWL_Test_Set_Frame_Resize_normpkl/', help="path of val pkl")    


    parser.add_argument('--window_min', type=int, default=16)
    parser.add_argument('--window_max', type=int, default=17)
    parser.add_argument('--window_stride', type=int, default=1)
    parser.add_argument('--maxsearch', type=int, default=1)
    parser.add_argument('--margin', type=int, default=0.5)
    parser.add_argument("--skip_gallery", default=True, type=bool)
    parser.add_argument('--length_of_each_gallery', type=int, default=16)
    parser.add_argument("--delete_gallery", default=True, type=bool)
    parser.add_argument("--topk", default=[0, 5], type=list)
    
    
    

    args = parser.parse_args()
    print('model_path =',args.model_path1)
    print('window_min = ',args.window_min, ' window_max =', args.window_max, ' window_stride =', args.window_stride)
    print('maxsearch =',args.maxsearch)
    print('margin =',args.margin)
    print('delete_gallery =',str(args.delete_gallery))
    print('skip_gallery =', args.skip_gallery)
    if args.skip_gallery:
        print('-length_of_each_gallery-',args.length_of_each_gallery)
    print('topk =', args.topk)
    #-------------------- load model config--------------------------
    cfgs = config_loader(args.cfgs) # load model config
    model_cfg = cfgs['model_cfg']
    #---------------------------------------------------------------
    #-------------------- create model------------------------------
    model1 = SLR_Pose_inference()#SLR_pose3D_inference()
    model1.build_network(cfgs['model_cfg1'])
    #----------------------load model-------------------------------
    checkpoint1 = torch.load(args.model_path1)
    model_state_dict1 = checkpoint1['model']
    model1.load_state_dict(model_state_dict1, strict=True)
    #--------------------------------------------------------------
    #----------------------model eval------------------------------
    model1.cuda()
    model1.eval()
    #-------------------------------------------------------------
    model2 = SLR_pose3D_inference()#SLR_pose3D_inference_2level()#
    model2.build_network(cfgs['model_cfg2'])
    #----------------------load model-------------------------------
    checkpoint2 = torch.load(args.model_path2)
    model_state_dict2 = checkpoint2['model']
    model2.load_state_dict(model_state_dict2, strict=True)
    #--------------------------------------------------------------
    #----------------------model eval------------------------------
    model2.cuda()
    model2.eval()


    #---------------------------------------------------------------
    #-------------------- create model------------------------------
    model3 = i3d_mlp_inference()
    model3.build_network(cfgs['model_cfg'])

    #----------------------load model-------------------------------
    checkpoint3 = torch.load(args.model_path3)
    model_state_dict3 = checkpoint3['model']
    model3.load_state_dict(model_state_dict3, strict=True)

    # #----------------------model eval------------------------------
    model3.cuda()
    model3.eval()
    # #-------------------------------------------------------------

    #---------------------------------------------------------------
    # print(model)
    with torch.no_grad():
        #-----------------------new dict -------------------------------
        Dict = dict()
        Posetransformer2D = BasePosejointTransform2D()
        Posetransformer3D = BasePosejointTransform3D()
        PosetransformerRGB = BaseRgbTransform()
        topklist = args.topk
        for tk in range(len(topklist)):
            print('---------------------------------------')
            print(tk,'-th',' topK = ','top',topklist[tk])
            print('---------------------------------------')
            if tk == 0:
                # gallery_feature_listall = []
                gallery_list2D, gallery_name_list2D = extract_npy(args.gallery_pkl2D)
                gallery_list3D, gallery_name_list3D = extract_npy(args.gallery_pkl3D)
                gallery_listRGB, gallery_name_listRGB = extract_npy(args.gallery_pklRGB)
                gallery_feature_list = []
                print('length of the gallery list = ',len(gallery_list2D))
                for i in range(len(gallery_list2D)):
                    gallery_list2D[i] = torch.from_numpy(Posetransformer2D(gallery_list2D[i])).float().unsqueeze(0).cuda()
                    gallery_list3D[i] = torch.from_numpy(Posetransformer3D(gallery_list3D[i])).float().unsqueeze(0).cuda()
                    gallery_listRGB[i] = torch.from_numpy(PosetransformerRGB(gallery_listRGB[i])).float().unsqueeze(0).cuda()
                    gallery_listRGB[i] = gallery_listRGB[i].permute(0,2,1,3,4)

                    if args.delete_gallery:
                        #assert gallery_list2D[i].size(2) == Delete[i][0]
                        #assert gallery_list3D[i].size(2) == Delete[i][0]
                        gallery_list2D[i] = gallery_list2D[i][:,:,Delete[i][1]-1:Delete[i][2]-1,:,:]
                        gallery_list3D[i] = gallery_list3D[i][:,:,Delete[i][1]-1:Delete[i][2]-1,:,:]
                        gallery_listRGB[i] = gallery_listRGB[i][:,:,Delete[i][1]-1:Delete[i][2]-1,:,:]
                    print(gallery_list2D[i].shape,gallery_list3D[i].shape,gallery_listRGB[i].shape)
                    if args.skip_gallery:
                        listskip2D = []
                        for l in range(0, gallery_list2D[i].size(2), gallery_list2D[i].size(2)//args.length_of_each_gallery):
                            if len(listskip2D) >= args.length_of_each_gallery:
                                break
                            listskip2D.append(l)
                        gallery_list2D[i] = gallery_list2D[i][:,:,listskip2D,:,:]
                    # print('2D-',i,'2D-',gallery_name_list2D[i],gallery_list2D[i].shape)

                    if args.skip_gallery:
                        listskip3D = []
                        for l in range(0, gallery_list3D[i].size(2), gallery_list3D[i].size(2)//args.length_of_each_gallery):
                            if len(listskip3D) >= args.length_of_each_gallery:
                                break
                            listskip3D.append(l)
                        gallery_list3D[i] = gallery_list3D[i][:,:,listskip3D,:,:]
                    # print('3D-',i,'3D-',gallery_name_list3D[i],gallery_list3D[i].shape)

                    if args.skip_gallery:
                        listskipRGB = []
                        for l in range(0, gallery_listRGB[i].size(2), gallery_listRGB[i].size(2)//args.length_of_each_gallery):
                            if len(listskipRGB) >= args.length_of_each_gallery:
                                break
                            listskipRGB.append(l)
                        gallery_listRGB[i] = gallery_listRGB[i][:,:,listskipRGB,:,:]
                    # print('-',i,'-',gallery_name_list[i],gallery_listRGB[i].shape)

                    output1 = model1(gallery_list2D[i])
                    output2 = model2(gallery_list3D[i])
                    output3 = model3(gallery_listRGB[i])['inference_feat']['embeddings']
                    output3 = output3.view(output3.size(0),4,-1)
                    # output3 = model3(gallery_list2D[i])
                    # print("model 1 output['inference_feat']['embeddings'] shape :", output1['inference_feat']['embeddings'].shape)
                    # print("model 2 output['inference_feat']['embeddings'] shape :", output2['inference_feat']['embeddings'].shape)
                    # print("model 3 output['inference_feat']['embeddings'] shape :", output3.shape)
                    output = torch.cat((output1['inference_feat']['embeddings'], output2['inference_feat']['embeddings'], output3) , dim=1)
                    # output = torch.cat((output1['inference_feat']['embeddings'], output2['inference_feat']['embeddings'], output3['inference_feat']['embeddings']) , dim=1)
                    # print(output.shape)
                    gallery_feature_list.append(output)
                
                gallery_feature_list = torch.cat(gallery_feature_list,dim=0)
            else:
                video_dist, video_id, video_starttime, video_endtime, video_name = loadtopk(args.topk_save_path,tk-1)
                video_id = np.array(video_id)
                gallery_feature_list = []
                #for cid in range(20):#val
                for cid in range(20,40):#test
                    id_index = list(np.where(video_id == cid))[0]
                    # print(id_index)
                    temp_list = []
                    for ii in range(topklist[tk]):
                        # print(ii)
                        gallery_npy2D, gallery_name2D = extract_npy_i(args.val_pkl2D,video_name[id_index[ii]])
                        gallery_npy3D, gallery_name3D = extract_npy_i(args.val_pkl3D,video_name[id_index[ii]])
                        gallery_npyRGB, gallery_nameRGB = extract_npy_i(args.val_pklRGB,video_name[id_index[ii]])

                        gallery_npy2D = torch.from_numpy(Posetransformer2D(gallery_npy2D)).float().unsqueeze(0).cuda()
                        gallery_npy3D = torch.from_numpy(Posetransformer3D(gallery_npy3D)).float().unsqueeze(0).cuda()
                        gallery_npyRGB = torch.from_numpy(PosetransformerRGB(gallery_npyRGB)).float().unsqueeze(0).cuda()
                        gallery_npyRGB = gallery_npyRGB.permute(0,2,1,3,4)

                        gallery_npy2D = gallery_npy2D[:,:,video_starttime[id_index[ii]]//40:(video_endtime[id_index[ii]])//40+1,:,:]
                        gallery_npy3D = gallery_npy3D[:,:,video_starttime[id_index[ii]]//40:(video_endtime[id_index[ii]])//40+1,:,:]
                        gallery_npyRGB = gallery_npyRGB[:,:,video_starttime[id_index[ii]]//40:(video_endtime[id_index[ii]])//40+1,:,:]
                        # print(video_name[id_index[ii]],gallery_npy.shape,gallery_name)
                        output1 = model1(gallery_npy2D)['inference_feat']['embeddings']
                        output2 = model2(gallery_npy3D)['inference_feat']['embeddings']
                        output3 = model3(gallery_npyRGB)['inference_feat']['embeddings']
                        output3 = output3.view(output3.size(0),4,-1)
                        # output3 = model3(gallery_npy2D)['inference_feat']['embeddings']
                        output = torch.cat((output1, output2, output3), dim = 1)
                        # output = torch.cat((output1, output2), dim = 1)
                        temp_list.append(output)
                    outputlist = torch.cat(temp_list, dim=0)
                    outputlist = torch.mean(outputlist,dim=0).unsqueeze(0)
                    # print(outputlist.shape)
                    # while count < 10:
                    gallery_feature_list.append(outputlist)
                gallery_feature_list = torch.cat(gallery_feature_list,dim=0) 
                print('-gallery_feature_list-',gallery_feature_list.shape)      

            print('-----------------------start evaluation-----------------------------')
            
            file_list = sorted(os.listdir(args.val_pkl2D))
            video_name = []
            video_dist = []
            video_id = []
            video_starttime = []
            video_endtime = []      
            video_feat = []   
            print('len(file_list) = ',len(file_list))
            # for i in range(0,100):
            for i in range(len(file_list)):
                probe_npy2D, probe_name = extract_npy_i(args.val_pkl2D,file_list[i])
                #print('probe_npy2D shape is: ', probe_npy2D.shape)
                probe_npy3D, probe_name = extract_npy_i(args.val_pkl3D,file_list[i])
                probe_npyRGB, probe_name = extract_npy_i(args.val_pklRGB,file_list[i])
            # for to in range(5):
                # if to != 1:
                #     continue
                # print(Top1[to])
                # probe_npy, probe_name = extract_npy_i(args.val_pkl,Top1[to][0])
                classid = int(probe_name.split('_')[1].split('s')[1])
                probe_npy2D = torch.from_numpy(Posetransformer2D(probe_npy2D)).float().unsqueeze(0).cuda()
                probe_npy3D = torch.from_numpy(Posetransformer3D(probe_npy3D)).float().unsqueeze(0).cuda()
                probe_npyRGB = torch.from_numpy(PosetransformerRGB(probe_npyRGB)).float().unsqueeze(0).cuda()
                probe_npyRGB = probe_npyRGB.permute(0,2,1,3,4)
                #print('probe_npy2D =',probe_npy2D.shape,'probe_name =',probe_name,'classid =',classid)
                #print('probe_npy3D =',probe_npy3D.shape,'probe_name =',probe_name,'classid =',classid)
                #if probe_npy2D.shape[2] != probe_npy3D.shape[2]:
                   #print('error')
                #probe_npy = torch.cat((probe_npy2D, probe_npy3D), dim = 3)
                #print('probe_npy =',probe_npy.shape,'probe_name =',probe_name,'classid =',classid)
                feat2D = probe_npy2D
                feat3D = probe_npy3D
                featRGB = probe_npyRGB

                windows_idlist = []
                windows_distlist = []
                windows_starttimelist = []
                windows_endtimelist = [] 
                windows_featlist = []

                for cw in range(args.window_min, args.window_max, args.window_stride):
                    # print('-cw-',cw)
                    current_window = min(probe_npy2D.size(2), cw)
                    list_windows_feat2D = []
                    list_windows_feat3D = []
                    list_windows_featRGB = []
                    for w in range(probe_npy2D.size(2)-current_window + 1):
                        list_windows_feat2D.append(feat2D[:,:,w:w+current_window,:,:])
                        list_windows_feat3D.append(feat3D[:,:,w:w+current_window,:,:])
                        list_windows_featRGB.append(featRGB[:,:,w:w+current_window,:,:])
                        #print('list_windows_feat3D shape is: ' , list_windows_feat3D[w].shape)
                        windows_idlist.append(classid)
                        windows_starttimelist.append(np.int64(w*40))
                        windows_endtimelist.append(np.int64((w+current_window-1)*40))
                        # st = max(0, w-3)
                        # ed = min(w+current_window-1+2, 99)
                        # print(w,st,ed)
                        # windows_starttimelist.append(np.int64(st*40))
                        # windows_endtimelist.append(np.int64((ed)*40))
                    torchlist_windows_feat2D = torch.cat(list_windows_feat2D, dim=0)
                    torchlist_windows_feat3D = torch.cat(list_windows_feat3D, dim=0)
                    torchlist_windows_featRGB = torch.cat(list_windows_featRGB, dim=0)
                    # print('torchlist_windows_feat2D =',torchlist_windows_feat2D.shape)
                    # print('torchlist_windows_feat3D =',torchlist_windows_feat3D.shape)
                    # print('torchlist_windows_featRGB =',torchlist_windows_featRGB.shape)
                    output_windows_feat1 = model1(torchlist_windows_feat2D)
                    output_windows_feat2 = model2(torchlist_windows_feat3D)
                    # output_windows_feat3 = model3(torchlist_windows_featRGB)

                    if torchlist_windows_featRGB.size(0) > 50:
                        retval = []
                        logitslist = []
                        for ret in range(torchlist_windows_featRGB.size(0)//50 + 1):
                            # print(ret)
                            tempm = model3(torchlist_windows_featRGB[ret*50:(ret+1)*50,:,:,:,:])
                            retval.append(tempm['inference_feat']['embeddings'])

                        output_windows_feat3 = torch.cat(retval,dim=0)
                        # logitslist =  torch.cat(logitslist,dim=0)
                    else:
                        output_windows_feat3 = model3(torchlist_windows_featRGB)['inference_feat']['embeddings']

                    output_windows_feat1 = output_windows_feat1['inference_feat']['embeddings']
                    output_windows_feat2 = output_windows_feat2['inference_feat']['embeddings']
                    # output_windows_feat3 = output_windows_feat3['inference_feat']['embeddings']
                    output_windows_feat3 = output_windows_feat3.view(output_windows_feat3.size(0),4,-1)
                    # print('output_windows_feat1 =',output_windows_feat1.shape)
                    # print('output_windows_feat2 =',output_windows_feat2.shape)
                    # print('output_windows_feat3 =',output_windows_feat3.shape)


                    output_windows_feat = torch.cat((output_windows_feat1, output_windows_feat2, output_windows_feat3), dim = 1)
                    # windows_dist = cuda_dist(output_windows_feat, gallery_feature_list[classid-20:classid+1-20,:], metric='cos').cpu().numpy()#test


                    # print('-output_windows_feat-',output_windows_feat.shape,'-gallery_feature_list[classid-20:classid+1-20,:]-',gallery_feature_list[classid-20:classid+1-20,:].shape)
                    output_windows_feats1 = output_windows_feat[:,:120,:]
                    output_windows_feats2 = output_windows_feat[:,120:,:].view(len(output_windows_feat),1,-1)
                    # print('-output_windows_feats1-',output_windows_feats1.shape,'-output_windows_feats2-',output_windows_feats2.shape)
                    gallerys1 = gallery_feature_list[classid-20:classid+1-20,:][:,:120,:]
                    gallerys2 = gallery_feature_list[classid-20:classid+1-20,:][:,120:,:].view(1,1,-1)
                    # print('-gallerys1-',gallerys1.shape,'-gallerys2-',gallerys2.shape)
                    # print('-windows_dist-',windows_dist.shape,output_windows_feat.shape,gallery_feature_list[classid:classid+1,:].shape)
                    windows_dist1 = cuda_dist(output_windows_feats1, gallerys1, metric='cos').cpu().numpy()#test
                    windows_dist2 = cuda_dist(output_windows_feats2, gallerys2, metric='cos').cpu().numpy()#test
                    # print('-windows_dist1-',windows_dist1.shape,'-windows_dist2-',windows_dist2.shape)
                    # print('-windows_dist1-',windows_dist1[:5],'-windows_dist2-',windows_dist2[:5])
                    windows_dist = windows_dist1 + windows_dist2
                    windows_distlist = windows_distlist + list(windows_dist[:,0])

                    # for l in range(len(gallery_feature_listall)):
                    #     tempdist = cuda_dist(output_windows_feat, gallery_feature_listall[l][classid-20:classid+1-20,:], metric='cos').cpu().numpy()
                    #     windows_distlist = windows_distlist + list(tempdist[:,0])
                    # print('-windows_distlist-',len(windows_distlist))
                    # windows_idx = windows_dist.sort(1)[1].cpu().numpy()
                    # print('-windows_idlist-', len(windows_idlist))
                    # for wfl in range(len(windows_idlist)):
                    #     t_feat = output_windows_feat[wfl:wfl+1].cpu()
                    #     windows_featlist.append(t_feat)
                    # print('-windows_featlist-', len(windows_featlist),windows_featlist[0].shape)
                    # print(len(windows_distlist),len(windows_idlist),len(windows_starttimelist),len(windows_endtimelist))
                    
                    # while len(windows_distlist) != len(windows_idlist):
                    #     windows_idlist = windows_idlist + windows_idlist[-len(output_windows_feat):]
                    #     windows_starttimelist = windows_starttimelist + windows_starttimelist[-len(output_windows_feat):]
                    #     windows_endtimelist = windows_endtimelist + windows_endtimelist[-len(output_windows_feat):]

                    # print(len(windows_distlist),len(windows_idlist),len(windows_starttimelist),len(windows_endtimelist))

                    windows_distlist, windows_idlist, windows_starttimelist, windows_endtimelist  = zip(*sorted(zip(windows_distlist, windows_idlist,windows_starttimelist,windows_endtimelist)))
                    windows_distlist, windows_idlist, windows_starttimelist, windows_endtimelist, = list(windows_distlist), list(windows_idlist), list(windows_starttimelist), list(windows_endtimelist)
                    # for wd in range(len(windows_distlist)):
                        # print(windows_distlist[wd], windows_starttimelist[wd]//40,windows_endtimelist[wd]//40)
                    # print()

                    # current_sign = []
                    # current_sign.append([idlist[mi],starttimelist[mi],endtimelist[mi]])
                Dict[probe_name] = [[windows_idlist[0],windows_starttimelist[0],windows_endtimelist[0]]]
                print('--',i+1,'--',probe_name,len(windows_distlist),[[windows_idlist[0],windows_starttimelist[0],windows_endtimelist[0]]], windows_starttimelist[0]//40,windows_endtimelist[0]//40)
                # print(windows_distlist[0], windows_distlist[-1])
                video_name.append(probe_name)
                video_dist.append(windows_distlist[0])
                video_id.append(windows_idlist[0])
                video_starttime.append(windows_starttimelist[0])
                video_endtime.append(windows_endtimelist[0])
                # video_feat.append(windows_featlist[0])
                # print(video_name, video_dist, video_id)


            with open(args.topk_save_path+'video_name_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_name, a, 3)
            with open(args.topk_save_path+'video_dist_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_dist, a, 3)
            with open(args.topk_save_path+'video_id_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_id, a, 3)
            with open(args.topk_save_path+'video_starttime_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_starttime, a, 3)
            with open(args.topk_save_path+'video_endtime_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_endtime, a, 3)
            # with open(args.topk_save_path+'video_feat_'+str(tk)+'.pkl', 'wb') as a:
            #     pickle.dump(video_feat, a, 3)
            with open(args.predictions_save_path+'predictions_pose_cos_skip_'+str(args.skip_gallery)+'_delete'+str(args.delete_gallery)+'_top'+str(tk)+'_track2_min_'+str(args.window_min)+'_max_'+str(args.window_max)+'_stride'+str(args.window_stride)+'_search'+str(args.maxsearch)+'_m'+str(args.margin)+'.pkl', 'wb') as f:
                pickle.dump(Dict, f, 3)
            # with open(args.predictions_save_path+'predictions_poseFv3_top'+str(tk)+'_track2_min'+str(args.window_min)+'_max'+str(args.window_max)+'_stride'+str(args.window_stride)+'_search'+str(args.maxsearch)+'_m'+str(args.margin)+'.pkl', 'wb') as f:
            #     pickle.dump(Dict, f, 3)
