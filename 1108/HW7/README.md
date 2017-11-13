# Homework 7

### Image Processing - Filtering
#### 建立以下四種Filter並附上原圖及處理後的結果圖：

#### 1. Gaussian Blur
> 產生一個Gaussian Filter套用至原圖，做出Gaussian Blur的效果

#### 2. Motion Blur
> 產生一個任意方向的Motion Filter套用至原圖，做出Motion Blur的效果

#### 3. Sharpening
> 產生一個Sharpening Filter（將兩個標準差不同的Gaussian Filter相減，或是參考投影片p.21直接給定一個3×3的Filter），做出Sharpen的效果

#### 4. Edege Detection
> 將輸入的影像灰階化（RGB三個值平均）為二維矩陣，套用Sobel Mask產生Ix及Iy，將Ix^2 + Iy^2產生G，將G前n%轉成黑色（0），剩下的轉為白色（255）
> 
> ![Sobel Mask](https://saush.files.wordpress.com/2011/04/filters.png)
> 
> ↑Sobel Mask
