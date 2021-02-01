# 2割をテストデータに移行
import shutil,random,glob,os
names=['akari','shiho','kyoko','miho','miku']
os.makedirs('./test',exist_ok=True)

for name in names:
    in_dir='./face/'+name+'/*'
    in_jpg=glob.glob(in_dir)
    img_file_name_list=os.listdir('./face/'+name+'/')
    # img_file_name_listをシャッフルして、そのうち2割をtest_imageディレクトリに入れる
    random.shuffle(in_jpg)
    os.makedirs('./test/'+name,exist_ok=True)
    for t in range(len(in_jpg)//5):
        shutil.move(str(in_jpg[t]),'./test/'+name)