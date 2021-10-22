# Uma_arm
the program : auto reset marason

Pythonの勉強のために作成した、
ソーシャルゲームのリセマラプログラムです。

Bluestackというエミュレータでゲームを起動し、  
Pyautoguiというライブラリを用いてマウスカーソルを操作したり、  
あらかじめ用意しておいたボタン画像をテンプレートマッチング → クリックして画面を操作しています。

Pyautoguiのテンプレートマッチングでは、ガチャ結果のレアリティの判定には精度が低く、  
（判定基準のアイコンが小さかったり、光るエフェクトがあったため）  
その解決法として、教師なし学習のK-means法を利用しました。  
  
K-means法によるクラスタリングの結果  
![img2](https://user-images.githubusercontent.com/58933271/138414486-d26f00d9-8031-43b8-aad6-4598e7d942ca.png)
