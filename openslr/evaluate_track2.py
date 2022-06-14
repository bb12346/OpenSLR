
import os
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
# import sys
# sys.path.append('/root/OpenSLR/openslr/modeling/models')
from modeling import models
from modeling.models.SLR_Pose_inference import SLR_Pose_inference

from tool import config_loader, generate_RGB_pkl
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def k_reciprocal_neigh( initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking(original_dist, query_num, k1=3, k2=3, lambda_value=0.3):
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    # print('starting re_ranking')

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

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

class BasePosejointTransform():
    def __init__(self,):
        self.selected = {
    '27': np.concatenate(([0,5,6,7,8,9,10], 
                    [91,95,96,99,100,103,104,107,108,111],[112,116,117,120,121,124,125,128,129,132]), axis=0) #27
    }
    max_body_true = 1
    max_frame = 150
    num_channels = 3
    def __call__(self, x):
        x = x[:,self.selected['27'],:]
        x = x[:,:,:,np.newaxis]
        x = np.transpose(x,(2,0,1,3))
        return x 
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
    Top1 = [
        ['p03_s00_n1005',45,57],#0
        ['p03_s01_n1419',84,95],#1 
        ['p03_s02_n1304',83,99],#2
        ['p03_s03_n1024', 75,88],#3
        ['p03_s04_n1475',58,71],#4
        ['p03_s05_n1351',19,38],#5
        ['p03_s06_n1344',89,99],#6
        ['p03_s07_n018',86,99],#7
        ['p09_s08_n1392',23,40],#8
        ['p09_s09_n608',28,52],#9
        ['p03_s10_n1030',0,34],#10
        ['p10_s11_n521', 61,79],#11
        ['p03_s12_n345',55,67],#12
        ['p03_s13_n1039',45,55],#13
        ['p03_s14_n1107', 36,57],#14
        ['p03_s15_n1163', 72,84],#15
        ['p03_s16_n1216', 7,16],#16
        ['p03_s17_n1124',70,83],#17
        ['p03_s18_n1129', 81,91],#18
        ['p03_s19_n127', 80,94] #19
    ]
    Delete = [
        [58,20,52],#0
        [59,25,51],#1
        [81,32,67],#2
        [60,18,54],#3
        [59,27,52],#4
        [69,20,57],#5
        [72,23,60],#6
        [62,19,48],#7
        [69,18,51],#8
        [74,23,63],#9
        [68,13,52],#10
        [52, 6,42],#11
        [72,25,67],#12
        [67,18,55],#13
        [59, 6,51],#14
        [51, 6,38],#15
        [60, 11,42],#16
        [75,12,63],#17
        [45, 8,36],#18
        [67, 6,53] #19
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgs', type=str,
                        default='/root/OpenSLR/config/SLR_Pose.yaml', help="path of config file")
    parser.add_argument('--model_path', type=str,
                        default='/root/OpenSLR/output/SLR/SLR_Pose/SLR_Pose/checkpoints/SLR_Pose-02500.pt', help="path of model param")
    parser.add_argument('--gallery_pkl', type=str,
                        default='/root/SSL/data/OSLWL_QUERY_VAL_SET_sorted_posepkl', help="path of gallery pkl")    
    parser.add_argument('--val_pkl', type=str,
                        default='/root/SSL/data/OSLWL_VAL_SET_VIDEOS_sorted_posepkl', help="path of val pkl")    

    parser.add_argument('--window_min', type=int, default=16)
    parser.add_argument('--window_max', type=int, default=17)
    parser.add_argument('--window_stride', type=int, default=1)
    parser.add_argument('--maxsearch', type=int, default=1)
    parser.add_argument('--margin', type=int, default=0.5)
    parser.add_argument("--skip_gallery", default=True, type=bool)
    parser.add_argument('--length_of_each_gallery', type=int, default=16)
    parser.add_argument("--delete_gallery", default=True, type=bool)
    # parser.add_argument("--topk", default=[0, 10, 10, 10, 10], type=list)
    parser.add_argument("--topk", default=[0, 5, 5, 5, 5], type=list)
    # parser.add_argument("--topk", default=[0], type=list)
    parser.add_argument("--topk_save_path", default='/root/OpenSLR/topk_track1/', type=str)
    parser.add_argument("--predictions_save_path", default='/root/OpenSLR/predictions_track1/', type=str)
    

    args = parser.parse_args()
    print('model_path =',args.model_path)
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
    model = SLR_Pose_inference()
    model.build_network(cfgs['model_cfg'])
    #----------------------load model-------------------------------
    checkpoint = torch.load(args.model_path)
    model_state_dict = checkpoint['model']
    model.load_state_dict(model_state_dict, strict=True)
    #--------------------------------------------------------------
    #----------------------model eval------------------------------
    model.cuda()
    model.eval()
    #---------------------------------------------------------------
    # print(model)
    with torch.no_grad():
        #-----------------------new dict -------------------------------
        Dict = dict()
        Posetransformer = BasePosejointTransform()
        topklist = args.topk
        for tk in range(len(topklist)):
            print('---------------------------------------')
            print(tk,'-th',' topK = ','top',topklist[tk])
            print('---------------------------------------')
            if tk == 0:
                gallery_list, gallery_name_list = extract_npy(args.gallery_pkl)
                gallery_feature_list = []
                print('length of the gallery list = ',len(gallery_list))
                for i in range(len(gallery_list)):
                    gallery_list[i] = torch.from_numpy(Posetransformer(gallery_list[i])).float().unsqueeze(0).cuda()
                    if args.delete_gallery:
                        assert gallery_list[i].size(2) == Delete[i][0]
                        gallery_list[i] = gallery_list[i][:,:,Delete[i][1]-1:Delete[i][2]-1,:,:]

                    if args.skip_gallery:
                        listskip = []
                        for l in range(0, gallery_list[i].size(2), gallery_list[i].size(2)//args.length_of_each_gallery):
                            if len(listskip) >= args.length_of_each_gallery:
                                break
                            listskip.append(l)
                        gallery_list[i] = gallery_list[i][:,:,listskip,:,:]
                    print('-',i,'-',gallery_name_list[i],gallery_list[i].shape)

                    output = model(gallery_list[i])
                    gallery_feature_list.append(output['inference_feat']['embeddings'])
                gallery_feature_list = torch.cat(gallery_feature_list,dim=0) 
                print('gallery_feature_list shape=',gallery_feature_list.shape)
                # gallery_feature_list = []
                # for t in range(len(Top1)):
                #     # print(Top1[t])
                #     gallery_npy, gallery_name = extract_npy_i(args.val_pkl,Top1[t][0])
                #     # print(gallery_npy.shape,gallery_name)
                #     gallery_feature_list.append(torch.from_numpy(Posetransformer(gallery_npy)).float().unsqueeze(0).cuda())
                #     gallery_feature_list[t] = gallery_feature_list[t][:,:,Top1[t][1]:Top1[t][2],:,:]
                #     # print(gallery_feature_list[t].shape,gallery_name)
                #     output = model(gallery_feature_list[t])
                #     gallery_feature_list[t] = output['inference_feat']['embeddings']
                # gallery_feature_list = torch.cat(gallery_feature_list,dim=0) 
                # print('-gallery_feature_list-',gallery_feature_list.shape)  
            else:
                video_dist, video_id, video_starttime, video_endtime, video_name = loadtopk(args.topk_save_path,tk-1)
                video_id = np.array(video_id)
                gallery_feature_list = []
                for cid in range(20):
                    id_index = list(np.where(video_id == cid))[0]
                    # print(id_index)
                    temp_list = []
                    for ii in range(topklist[tk]):
                        # print(ii)
                        gallery_npy, gallery_name = extract_npy_i(args.val_pkl,video_name[id_index[ii]])
                        gallery_npy = torch.from_numpy(Posetransformer(gallery_npy)).float().unsqueeze(0).cuda()
                        gallery_npy = gallery_npy[:,:,video_starttime[id_index[ii]]//40:(video_endtime[id_index[ii]])//40+1,:,:]
                        # print(video_name[id_index[ii]],gallery_npy.shape,gallery_name)
                        output = model(gallery_npy)['inference_feat']['embeddings']
                        temp_list.append(output)
                    outputlist = torch.cat(temp_list, dim=0)
                    outputlist = torch.mean(outputlist,dim=0).unsqueeze(0)
                    # print(outputlist.shape)
                    # while count < 10:
                    gallery_feature_list.append(outputlist)
                gallery_feature_list = torch.cat(gallery_feature_list,dim=0) 
                print('-gallery_feature_list-',gallery_feature_list.shape)      

            print('-----------------------start evaluation-----------------------------')
            
            file_list = sorted(os.listdir(args.val_pkl))
            video_name = []
            video_dist = []
            video_id = []
            video_starttime = []
            video_endtime = []      
            video_feat = []   
            print('len(file_list) = ',len(file_list))
            # for i in range(0,100):
            for i in range(len(file_list)):
                probe_npy, probe_name = extract_npy_i(args.val_pkl,file_list[i])
            # for to in range(5):
                # if to != 1:
                #     continue
                # print(Top1[to])
                # probe_npy, probe_name = extract_npy_i(args.val_pkl,Top1[to][0])
                classid = int(probe_name.split('_')[1].split('s')[1])
                probe_npy = torch.from_numpy(Posetransformer(probe_npy)).float().unsqueeze(0).cuda()
                # print('probe_npy =',probe_npy.shape,'probe_name =',probe_name,'classid =',classid)

                feat = probe_npy

                windows_idlist = []
                windows_distlist = []
                windows_starttimelist = []
                windows_endtimelist = [] 
                windows_featlist = []

                for cw in range(args.window_min, args.window_max, args.window_stride):
                    # print('-cw-',cw)
                    current_window = min(probe_npy.size(2), cw)
                    list_windows_feat = []
                    for w in range(probe_npy.size(2)-current_window + 1):
                        list_windows_feat.append(feat[:,:,w:w+current_window,:,:])
                        windows_idlist.append(classid)
                        windows_starttimelist.append(np.int64(w*40))
                        windows_endtimelist.append(np.int64((w+current_window-1)*40))
                    torchlist_windows_feat = torch.cat(list_windows_feat, dim=0)
                    # print('torchlist_windows_feat =',torchlist_windows_feat.shape)
                    output_windows_feat = model(torchlist_windows_feat)
                    output_windows_feat = output_windows_feat['inference_feat']['embeddings']
                    # print('-output_windows_feat-',output_windows_feat.shape,gallery_feature_list[classid:classid+1,:].shape)
                    windows_dist = cuda_dist(output_windows_feat, gallery_feature_list[classid:classid+1,:], metric='cos').cpu().numpy()
                    # print('-windows_dist-',windows_dist.shape,output_windows_feat.shape,gallery_feature_list[classid:classid+1,:].shape)

                    # Rk_martix = torch.cat((gallery_feature_list[classid:classid+1,:],output_windows_feat),axis=0)
                    # # Rk_martix = torch.cat((gallery_feature_list,output_windows_feat),axis=0)
                    # print('-Rk_martix-',Rk_martix.shape)
                    # dist_RK = cuda_dist(Rk_martix, Rk_martix, metric='euc').cpu().numpy()
                    # print('-dist_RK-',dist_RK.shape)
                    # for kk in range(50):
                    #     re_rank = re_ranking(dist_RK, gallery_feature_list[classid:classid+1,:].shape[0], k1=1, k2=kk, lambda_value=0.3)
                    #     # re_rank = re_ranking(dist_RK, gallery_feature_list.shape[0], k1=1, k2=30, lambda_value=0.3)
                    #     idxa = np.argsort(re_rank, axis=1)
                    #     # idxa = np.argsort(re_rank[classid:classid+1,:], axis=1)
                    #     # print('-idxa-',idxa.shape)
                    #     print('-idxa-',kk, idxa[0,:5])

                    windows_distlist = windows_distlist + list(windows_dist[:,0])
                    # print('-windows_distlist-',len(windows_distlist))
                    # windows_idx = windows_dist.sort(1)[1].cpu().numpy()
                    # print('-windows_idlist-', len(windows_idlist))
                    for wfl in range(len(windows_idlist)):
                        t_feat = output_windows_feat[wfl:wfl+1].cpu()
                        windows_featlist.append(t_feat)
                    # print('-windows_featlist-', len(windows_featlist),windows_featlist[0].shape)
                    windows_distlist, windows_idlist, windows_starttimelist, windows_endtimelist, windows_featlist  = zip(*sorted(zip(windows_distlist, windows_idlist,windows_starttimelist,windows_endtimelist,windows_featlist)))
                    windows_distlist, windows_idlist, windows_starttimelist, windows_endtimelist, windows_featlist = list(windows_distlist), list(windows_idlist), list(windows_starttimelist), list(windows_endtimelist), list(windows_featlist)
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
                video_feat.append(windows_featlist[0])
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
            with open(args.topk_save_path+'video_feat_'+str(tk)+'.pkl', 'wb') as a:
                pickle.dump(video_feat, a, 3)
            with open(args.predictions_save_path+'predictions_pose_cos_skip_'+str(args.skip_gallery)+'_delete'+str(args.delete_gallery)+'_top'+str(tk)+'_track2_min_'+str(args.window_min)+'_max_'+str(args.window_max)+'_stride'+str(args.window_stride)+'_search'+str(args.maxsearch)+'_m'+str(args.margin)+'.pkl', 'wb') as f:
                pickle.dump(Dict, f, 3)
            # with open(args.predictions_save_path+'predictions_poseFv3_top'+str(tk)+'_track2_min'+str(args.window_min)+'_max'+str(args.window_max)+'_stride'+str(args.window_stride)+'_search'+str(args.maxsearch)+'_m'+str(args.margin)+'.pkl', 'wb') as f:
            #     pickle.dump(Dict, f, 3)
