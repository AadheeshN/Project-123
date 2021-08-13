# Import Necessary Modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import PIL.ImageOps
import os, ssl, time

# Fetch Data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("https://raw.githubusercontent.com/whitehatjr/datasets/master/C%20122-123/labels.csv")['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
print("Number of Classes (Letters) =", nclasses)

# Check Data
samples_per_class = 5
figure = plt.figure(figsize = (nclasses * 2, (1 + samples_per_class * 2)))
idx_cls = 0

for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_class, replace = False)
  i = 0
  for idx in idxs:
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(samples_per_class, nclasses, plt_idx)
    p = sns.heatmap(np.reshape(X[idx], (22,30)), cmap = plt.cm.gray, xticklabels = False, yticklabels = False, cbar = False)
    p = plt.axis("off")
    i += 1
  idx_cls +=1

# Training and Testing Data 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

# Logistic Regression
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

# Selecting Default Camera
cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()
        # Gray color box in the center of the camera screen
        gray = cv2.cvtColor(frame, cv2.COLOR_COLOR_BGR2GRAY)

        # Create the box
        height, width = gray.shape
        upper_left = (int(width/2 - 56), int(height/2 - 56))
        bottom_right = (int(width/2 + 56), int(height/2 + 56))

        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
    
        # Region of Interest
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        # Converting the image captured with the help of cv2 to PIL format
        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert("L")
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)

        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)

        max_pixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        
        test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted class is: ", test_pred)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()