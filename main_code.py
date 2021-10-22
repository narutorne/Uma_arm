import pyautogui as pg
import numpy as np
import time
import requests
import os
import cv2
import glob
import csv
import random
import pyperclip
import sys
import matplotlib.pyplot as plt
import shutil
import win32gui
import win32con
import win32api

#エミュレータ全画面スクショ
def screenshot_all():    
    screenshot = pg.screenshot(region = (0, 0, 326, 553)) #BlueStacks用のスクショサイズ
    screenshot.save('screen.PNG') #スクショのファイル名

#エミュレータを最前面に出す
def foreground():
    print("hello")
    hwnd = win32gui.FindWindow(None, 'BlueStacks 5')
    win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,0,0,0,0,win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    print(left, top, right, bottom)

#Template画像とのマッチング処理の実行部分
def image_recognize(moto): # moto:判定元

    #ウィンドウ探索をパスする場合は判定値:1だけ返す
    if moto == "pass":
        return 1, 0, 0
    
    #スコアの閾値を決めて判定する
    def judgement(num):
        if 0.98 < num:
            return 1
        else:
            return 0
    
    #ファイルパス作成用の文字列
    tem = "./Template/"

    screenshot_all()
    image = cv2.imread('screen.PNG')
    template = cv2.imread(tem + moto)
    
    # 画像マッチング処理
    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    # 最も類似度が高い位置を取得する。
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    # 類似度から、判定結果を求める（上で設定した式を使用）
    Judg = judgement(maxVal)
    #print(moto, "Value :", maxVal)

    return Judg, maxLoc, template.shape #認識判定、最大評価値、認識画像のサイズを返す

#Template画像を探索し、見つけた場合はクリックする
def image_matching_and_click(window, button):

    for num_w in range(60):
        #メッセージウィンドウの探索
        Judg_w, maxLoc_w, shape_w = image_recognize(window)  
        if Judg_w == 1:
            #print("w_finded.")
            #print(" maxLoc_w", maxLoc_w)
            #print(" shape_w", shape_w)
            
            time.sleep(0.3)

            #ボタンの探索
            Judg_b, maxLoc_b, shape_b = image_recognize(button)  
            if Judg_b == 1:
                #print("b_finded.")
                #print(" maxLoc_b", maxLoc_b)
                #print(" shape_b", shape_b)

                #ボタンの中央まで移動          
                pg.moveTo(maxLoc_b[0] + shape_b[1] / 2, maxLoc_b[1] + shape_b[0] / 2, duration = 0.5)
                pg.click()
                break
                
        if num_w > 50:
            print("Timeout Error!")
            sys.exit()
                      
        time.sleep(0.5)
        
#10連ガチャ結果のレアリティ部分10か所のスクショ
def screenshot_10times(): 
    dirs = glob.glob("./rarelity/*.PNG")
    screenshot = pg.screenshot(region = (35, 91, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(1))
    screenshot = pg.screenshot(region = (122, 91, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(2))
    screenshot = pg.screenshot(region = (209, 91, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(3))
    screenshot = pg.screenshot(region = (78, 188, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(4))
    screenshot = pg.screenshot(region = (166, 188, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(5))
    screenshot = pg.screenshot(region = (35, 286, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(6))
    screenshot = pg.screenshot(region = (122, 286, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(7))
    screenshot = pg.screenshot(region = (209, 286, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(8))
    screenshot = pg.screenshot(region = (78, 384, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(9))
    screenshot = pg.screenshot(region = (166, 384, 13, 10))
    screenshot.save('./rarelity/{:04d}.PNG'.format(10))

#pyautoguiのテンプレートマッチングによる
#ガチャ結果 → SSRの枚数カウント 
def count_SSR():
    im_SSR = cv2.imread("./Template/im_SSR.png")
    cnt_SSR = 0
    for sc_num in range(5):
        sc = cv2.imread("./sc_save/{}.PNG".format(sc_num+1))
        
        #plt.subplot(1,2,1)
        #plt.imshow(im_SSR)
        #plt.subplot(1,2,2)
        #plt.imshow(sc)
        #print(sc)
        # 画像マッチング処理
        result = cv2.matchTemplate(sc, im_SSR, cv2.TM_CCORR_NORMED)
        #print(result)   
        #print("img.shape", sc.shape)  # (380, 694, 3)
        #print("template.shape", im_SSR.shape)  # (30, 31, 3)
        #print("result.shape", result.shape)  
        # 最も類似度が高い位置を取得する。
        #for i in result:
        #    for j in i:
        #        if j > 0.985:
        #            print(j)
        #閾値以上を抽出
        pos_y, pos_x = np.where(result >= 0.985)
        print(pos_x)
        print(pos_y)

        last_x = 0
        last_y = 0
        pos_nx = []
        pos_ny = []

        for i in range(len(pos_x)):
            length = np.sqrt((pos_x[i] - last_x)**2 + (pos_y[i] - last_y)**2)
            #print("length :", length)
            if length > 5:
                pos_nx.append(pos_x[i])
                pos_ny.append(pos_y[i])
            last_x = pos_x[i]
            last_y = pos_y[i]
        print(pos_nx)
        print(pos_ny)
        cnt_SSR += len(pos_nx)
        # 描画する。
        #dst = hikaku.copy()
        #for x, y in zip(pos_nx, pos_ny):
        #    cv2.rectangle(
        #        dst,
        #        (x, y),
        #        (x + im_SSR.shape[1], y + im_SSR.shape[0]),
        #        color=(0, 0, 0),
        #        thickness=4,
        #    )
        #plt.imshow(dst)
    return cnt_SSR

#k-means法を用いた判別
#ガチャ結果 → SSRの枚数カウント
def count_SSR_kmeans(rarelity_cnt):
    
    from PIL import Image
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import KMeans

    #Numpy配列に変換する
    def img_to_matrix(img):
        img_array = np.asarray(img)
        return img_array

    #配列を平坦化する
    def flatten_img(img_array):
        s = img_array.shape[0] * img_array.shape[1] * img_array.shape[2]
        img_width = img_array.reshape(1, s)
        #print(img_width)
        return img_width[0]

    screenshot_10times() 
    #pickleファイルをロード
    import pickle
    #pickleファイルのファイルパスを指定
    with open('./uma_kmeans.pkl', 'rb') as fp:
        clf = pickle.load(fp)

    dataset = []
    img_paths = []

    #imgファイルのパスを指定
    for file in glob.glob("./rarelity/*.PNG"):
        img_paths.append(file)

    #print(img_paths)
    img_paths.sort()
    #print(img_paths)

    #print("Image number:", len(img_paths))
    #print("Image list make done.")
    #print(img_paths)

    for i in img_paths:
        img = Image.open(i)
        img = img_to_matrix(img)
        img = flatten_img(img)
        dataset.append(img)

    #print(type(dataset))
    dataset = np.array(dataset)
    #print(dataset)
    #print(dataset.shape)
    #print("Dataset make done.")

    #Kmeans
    for i in range(10):
        y = clf.predict(dataset[i].reshape(1, -1))
        #print(y)
        rarelity_cnt[y[0]] += 1
    print(rarelity_cnt) #[R, SR, SSR]

    
#リセマラの動作実行 本体
def resemara_action():
    #チュートリアルを終えてホーム画面に入ったところからスタート
    p = "pass"
    foreground()
    #ホーム画面プレゼントの受け取り
    image_matching_and_click(p, "b_present.png")
    image_matching_and_click(p , "b_ikkatsu.png")
    time.sleep(1)
    image_matching_and_click(p , "b_tojiru.png")
    time.sleep(1)    
    image_matching_and_click(p , "b_tojiru.png")
    time.sleep(1)
    image_matching_and_click(p , "b_tojiru.png")
    #ガチャ画面へ
    image_matching_and_click(p, "b_home_gacha.png")
    time.sleep(3)
    pg.moveTo(275, 350, duration = 0.5)
    pg.click()
    image_matching_and_click("w_gacha.png", "b_10ren.png")
    
    #ガチャ結果のカウント　初期化
    rarelity_cnt = [0, 0, 0]  #[SSR, SR, R]

    for i in range(5):
        image_matching_and_click(p, "b_gachahiku.png")
        for j in range(15):
            judg, x, y = image_recognize("b_mouikkai.png")
            if judg == 1:
                break
            image_matching_and_click(p, "b_skipmark.png")
            time.sleep(2)
        screenshot = pg.screenshot(region = (10, 40, 280, 500)) #BlueStacks用のスクショサイズ
        screenshot.save('./sc_save/{}.PNG'.format(i+1))

        count_SSR_kmeans(rarelity_cnt)

        print(i)
        if i == 4:
            print("all draw.")
            break
        image_matching_and_click(p, "b_mouikkai.png")
    time.sleep(1)
    image_matching_and_click(p, "b_modoru.png")
    time.sleep(5)
    image_matching_and_click("w_gacha.png", "b_home.png")
    

    #SSRの枚数判定
    num_SSR = rarelity_cnt[2]
    #num_SSR = count_SSR() #pyautoguiのテンプレートマッチング
    print("num_SSR : ", num_SSR)
    if num_SSR >= 4:
        print("SSRが4枚以上！")
    
        #アカウント連携パスワードの設定
        time.sleep(10)
        pg.moveTo(265, 65, duration = 1)
        time.sleep(10)
        image_matching_and_click(p, "b_home_menu.png")
        time.sleep(1)
        image_matching_and_click(p, "b_datarenkei.png")
        image_matching_and_click(p, "b_datarenkei2.png")
        time.sleep(1)
        image_matching_and_click(p, "b_settei.png")
        time.sleep(1)
        image_matching_and_click(p, "b_settei2.png")
        time.sleep(2)
        image_matching_and_click(p, "b_passform.png")
        time.sleep(1)
        pg.write("Pass2580", interval = 0.25)
        image_matching_and_click(p, "b_whiteOK.png")
        image_matching_and_click(p, "b_passform2.png")
        time.sleep(1)
        pg.write("Pass2580", interval = 0.25)
        image_matching_and_click(p, "b_whiteOK.png")
        image_matching_and_click(p, "b_check.png")
        image_matching_and_click(p, "b_ok.png")
        time.sleep(1)
        screenshot = pg.screenshot(region = (5, 190, 285, 210)) #BlueStacks用のスクショサイズ
        screenshot.save('./sc_save/{}.png'.format("ID"))
        image_matching_and_click(p, "b_tojiru.png")

        #ディレクトリの作成・移動
        dirs = glob.glob("./sc_save/r*")
        os.makedirs("./sc_save/r{:04d}".format(len(dirs)+1))
        files = glob.glob("./sc_save/*.png")
        #print(files)
        for file in files:
            shutil.move(file, "./sc_save/r{:04d}".format(len(dirs)+1))
    
    #データの削除
    time.sleep(10)
    pg.moveTo(265, 65, duration = 1)
    time.sleep(10)
    pg.click()
    #image_matching_and_click(p, "b_home_menu.png")
    pg.moveTo(280, 280, duration = 0.5)
    pg.dragRel(0, 30, duration = 1)
    image_matching_and_click(p, "b_title.png")
    time.sleep(5)
    image_matching_and_click(p, "b_menumark.png")
    image_matching_and_click(p, "b_delete.png")
    image_matching_and_click(p, "b_userdelete.png")
    image_matching_and_click(p, "b_userdelete.png")
    image_matching_and_click(p, "b_tojiru.png")
    pg.sleep(5)
    pg.click()
    pg.sleep(5)
    pg.click()
    pg.sleep(3)
    #リセット後、タイトルから
    image_matching_and_click("w_kiyaku.png", "b_doui.png")
    time.sleep(1)
    image_matching_and_click("w_privacy.png", "b_doui.png")
    time.sleep(1)
    image_matching_and_click("w_signin.png", "b_cancel.png")
    time.sleep(2)
    image_matching_and_click("w_renkei.png", "b_ato.png")
    time.sleep(1)
    image_matching_and_click("w_chutrial.png", "b_skip.png")
    #トレーナー名入力
    pg.sleep(1)
    image_matching_and_click("w_trainer.png", "b_name.png")
    pg.sleep(2)
    pg.write("Shirayuki", interval = 0.25)
    image_matching_and_click("w_trainer.png", "b_whiteOK.png")
    image_matching_and_click("w_trainer.png", "b_touroku.png")
    image_matching_and_click("w_touroku.png", "b_ok.png")
    #ガチャピックアップ
    image_matching_and_click(p, "b_skipmark.png")
    pg.click()
    time.sleep(2)
    pg.click()
    image_matching_and_click("w_startdash.png", "b_skipmark.png")
    image_matching_and_click(p, "b_skipmark.png")
    image_matching_and_click("w_oshirase.png", "b_tojiru.png")
    time.sleep(1)
    image_matching_and_click(p, "b_tojiru.png")
    time.sleep(1)
    image_matching_and_click(p, "b_tojiru.png")
    #ロビー画面に到着
    time.sleep(5)

args = sys.argv

#引数で与えた数字の回数だけリセマラ実行する
for times in range(1, int(args[1])+1):
    print(times, ": riset marason start!")
    resemara_action()
    print(times, ": riset marason end!")
