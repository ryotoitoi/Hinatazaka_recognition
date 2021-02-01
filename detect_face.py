import glob
import os
import cv2
'''
dataディレクトリから画像を読み込んで顔を切り取り、faceディレクトリに保存
'''

names=['akari','shiho','kyoko','miho','miku']
out_dir = './face'
os.makedirs(out_dir,exist_ok=True)
image_size=224

for i in range(len(names)):
    in_dir = os.path.join(".","data",names[i],"*.jpg")
    in_jpg=glob.glob(in_dir)
    os.makedirs(out_dir+f'/{names[i]}',exist_ok=True)
    print(in_jpg)
    # print(len(in_jpg))
    for num in range(len(in_jpg)):
        image=cv2.imread(in_jpg[num])
        if image is None:
            print('No open:',num)
            continue
        
        image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
        # 顔認識の実行
        face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))       
        #顔が１つ以上検出された時
        if len(face_list) > 0:
            for rect in face_list:
                x,y,width,height=rect
                image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
                if image.shape[0]<64:
                    continue
                image = cv2.resize(image,(image_size,image_size))
                #保存
                fileName=os.path.join(out_dir+"/"+names[i],str(num)+".jpg")
                cv2.imwrite(str(fileName),image)
                print(str(num)+".jpgを保存しました.")
        #顔が検出されなかった時
        else:
            print("no face")
            continue
        print(image.shape)        

