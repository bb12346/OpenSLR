import cv2
import os
import argparse



def extract_framesROI(video_path_folder):
    '''
    '''
    list_video_frame = []
    list_video_time = []
    cap = cv2.VideoCapture()
    if not cap.open(video_path_folder):
        print("can not open the video")
        exit(1)

    count = 1
    while True:
        _, frame = cap.read()
        # print(frame)
        if frame is None:
            break
        row, col, _ = frame.shape
        # print(row, col)
            
        list_video_frame.append(frame)
        this_time = str(cap.get(cv2.CAP_PROP_POS_MSEC)).split('.')[0]
        if count !=1:
            int_this_time = int(this_time)
            # if int_this_time ==0:
            if int_this_time == 0 or int_this_time%40 != 0:
                int_last_time = int(last_time)
                int_this_time = int_last_time + 40
                this_time = str(int_this_time)
            
        list_video_time.append(this_time)
        last_time = this_time

        count += 1
    cap.release()
    return list_video_frame, list_video_time


def generate_frame(video_path, save_path):
    total_count = 0 
    video_list  = sorted(os.listdir(video_path))
    # txt_path = ' '
    print(video_list)
    print('len(txt_list) =',len(video_list))
    for folder_name in video_list:
        total_count = total_count + 1
        folder_name = folder_name.split('.')[0]
        print('-----------------'+folder_name+' start --------------')
        print(video_path+folder_name+'.mp4')
        # print(txt_path+folder_name+'.txt')
        # list_GTtxt = []
        # with open(txt_path+folder_name+'.txt', 'r') as file:
        #     for line in file.readlines():                
        #         line = line.strip()                             
        #         list_GTtxt.append(line)
        # list_GTtxt = sorted(list_GTtxt)
        video_path_folder = video_path+folder_name+'.mp4'
        list_video_frame, list_video_time = extract_framesROI(video_path_folder = video_path_folder)
        # print('-len(list_video_frame)-',len(list_video_frame),'-len(list_video_time)-',len(list_video_time))
        print(list_video_time)

        save_path_each = save_path+folder_name+'/'
        isExists = os.path.exists(save_path_each)
        if not isExists:
            os.makedirs(save_path_each)
            for i in range(len(list_video_time)):
                frame_id ='%03d' % (i+1)
                cv2.imwrite(save_path_each+frame_id+'.png', list_video_frame[i])    

        # for i in range(len(list_GTtxt)):
        #     total_count = total_count + 1
        #     class_id, start_time, end_time = list_GTtxt[i].split(',')
        #     str_class_id ='%03d' % (int(class_id) + 1)
        #     print('class_id + 1 = ', str_class_id, ' start_time = ', start_time, 'end_time = ', end_time)
        #     start_index = list_video_time.index(start_time)
        #     end_index = list_video_time.index(end_time)
        #     # print('-start_index-',start_index)
        #     # print('-end_index-',end_index)
        #     # print(list_video_time[start_index],list_video_time[end_index])
        #     # print(list_video_time[start_index:end_index+1])
        #     currentframe = list_video_frame[start_index:end_index+1]
        #     currenttime = list_video_time[start_index:end_index+1]

        #     # save_path_each = save_path+str(int(class_id) + 1)+'/'+folder_name+'_'+str(start_time)+'_'+str(end_time)+'/'
        #     # save_path_each = save_path+str_class_id+'/'+folder_name+'/'+str(start_time)+'_'+str(end_time)+'/'
        #     save_path_each = save_path+str_class_id+'/00/'+folder_name+'_'+str(start_time)+'_'+str(end_time)+'/'
        #     isExists = os.path.exists(save_path_each)
        #     if not isExists:
        #         os.makedirs(save_path_each)
        #         for i in range(len(currenttime)):
        #             cv2.imwrite(save_path_each+str(i)+'_'+str(currenttime[i])+'.png', currentframe[i])
        # print('-----------------'+folder_name+' end --------------')
    print('total count = ',total_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--video_path", default='/data/SSL/PUBLIC/OSLWL_QUERY_VAL_SET/', type=str, help="video_path."
    # )
    parser.add_argument(
        "--video_path", default='/fuxi_cv/SSL/original_2stage_data/OSLWL_Query_Set-and-OSLWL_Test_Set/OSLWL_Test_set/OSLWL_QUERY_TEST_SET/', type=str, help="video_path."
    )
    parser.add_argument(
        "--save_path", default='/root/SSL/data/OSLWL_QUERY_TEST_SET_frame/', type=str, help="save_path."
    )
    args = parser.parse_args()

    generate_frame(args.video_path, args.save_path)