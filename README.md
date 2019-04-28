# Kmeans-pp numpy implementation

## 執行方法 - 
### python Kmeans.py --dataset=() --K=() --init=() --maxiter=() --seed=()  


--dataset : 可以是 iris 或 abalone      (required)

--K : iris建議K=3                      (required)

--init : 可輸入kmeans++ , 預設為空      (optional)

--maxiter : 可輸入最多運算次數, 預設20   (optional)

--seed : 可輸入random種子 , 預設None    (optional)

## Kmeans++ Algorithm
* 實作方法範例

![Alt text](/images/kmeans++_example.png)

0. 隨機選擇一資料點作為初始質心

1. 計算各資料點到已有質心的最短距離 <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ {D(x)}^2 $$" style="border:none;">

2. total = 所有距離加總值

3. 各資料點機率 = 資料點 <img src="http://chart.googleapis.com/chart?cht=tx&chl= $$ {D(x)}^2 $$" style="border:none;"> 除以 total

--> 所算出之機率 = 該資料點被選作下一個質心的機率 (所有機率加總和為1) , 以此機率表選出新質心

4. 持續執行步驟1~3 , 直到質心數 = K

## 使用kmeans++作為初始化質心的影響
![Alt text](/images/iris_kmenas++.png) ![Alt text](/images/iris_origin.png)
* 上圖為使用kmeans++ , 下圖為一般Kmeans , 方形點為質心更新過程 , 使用kmeans++初始質心位置較接近最終質心

## 使用kmeans++對最終收斂SSE的影響
![Alt text](/images/iris_SSE.png) ![Alt text](/images/abalone_SSE.png)
* 圖中為使用不同random seed更新20epochs後, 分別在Iris和Abalone Dataset上最終收斂的Sum of Square Error(SSE)
* 使用kmeans++平均可收斂在較佳的SSE
