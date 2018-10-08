import os
from flask import Flask, request, render_template, send_from_directory, url_for
from werkzeug import secure_filename
from PIL import Image
import numpy as np
import pandas as pd
from sklearn import svm
import warnings; warnings.simplefilter('ignore')  # Fix NumPy issues.
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
import pickle
import translate as tr
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from skimage import io
from sklearn.datasets import load_sample_image
#anaconda prompt: activate flask

UPLOAD_FOLDER = os.path.basename('uploads')
ROUND_NUM = 3

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def toRound(arg):
    return round(arg, ROUND_NUM)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload_file():
    phone = request.form['phone']
    translate = None
    init_flag = int(request.form['initialize'])
    if 'Kvalue' in request.form:
        Kvalue = int(request.form['Kvalue'])
    else:
        Kvalue = None
    mc = None
    list_num =  int(request.form['Kvalue'])
    if 'image' in request.files:
        file = request.files['image']
        filename = secure_filename(file.filename)
        f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(f)
    else:
        filename = request.form['hidden']
    change_color_list=[]
    colornum = 0
    if 'colornum' in request.form:
        colornum = int(request.form['colornum'])
    #print(colornum)
    for i in range(colornum):
        #print(request.form[str(i)])
        change_color_list.append(request.form[str(i)])

    """
    while (request.form[str(index)]):
        index+=1
    """
    open_filename = 'uploads/' + filename
    Kmeans_filename = 'uploads/' + 'K_means.jpg'
    changed_filename = 'uploads/' + 'changed_file.jpg'

    """i
    K - means on Image code
    """
    sample = io.imread(open_filename)
    x,y,z = sample.shape
    data = sample / 255.0 # use 0...1 scale
    data = data.reshape(x * y, z)
    #data.shape

    kmeans = MiniBatchKMeans(Kvalue)
    kmeans.fit(data)
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
    new_colors = kmeans.cluster_centers_[kmeans.predict(data)]

    sample_recolored = new_colors.reshape(sample.shape)

    io.imsave(Kmeans_filename, sample_recolored)

    target_colors = new_colors * 255
    target_colors = np.int32(target_colors)

    unique_rows, unique_counts = np.unique(target_colors, axis=0, return_counts=True)
    unique_rows, unique_counts = np.unique(target_colors, axis=0, return_counts=True)
    arr = np.c_[ unique_rows, unique_counts ]
    arr = arr[arr[:,3].argsort()][::-1]
    df = pd.DataFrame(arr,columns=['R','G','B','counts'])
    Kmeans_im = Image.open(Kmeans_filename, 'r') # image that has lots of color
    
    """
    if color change button is pressed
    """
    changed_target_colors = 0 #initially not existing
    changed_index=0 #initially not existing
    changed_im = 0 # initially not existing
    changed_df = 0 # initially not existing
    if colornum is not 0 and init_flag is 0:
        changed_target_colors = np.copy(target_colors)
        for i in change_color_list:
            changed_RED=int(i.split(',')[0])
            changed_GREEN=int(i.split(',')[1])
            changed_BLUE=int(i.split(',')[2])
            np.place(changed_target_colors,changed_target_colors==arr[:,:-1][changed_index],[changed_RED,changed_GREEN,changed_BLUE])
            changed_index+=1

        unique_rows, unique_counts = np.unique(changed_target_colors, axis=0, return_counts=True)
        arr = np.c_[ unique_rows, unique_counts ]
        arr = arr[arr[:,3].argsort()][::-1]
        changed_df = pd.DataFrame(arr,columns=['R','G','B','counts'])

        changed_target_colors = np.float64(changed_target_colors)
        changed_target_colors /= 255
        changed_target_colors = changed_target_colors.reshape(sample.shape)
        io.imsave(changed_filename, changed_target_colors)
        changed_im = Image.open(changed_filename, 'r')


    """
    image processing code
    """
    im = Image.open(open_filename, 'r') # image that has lots of color
    x, y = im.size

    # select appropriate model
    if phone == 'gn5movie':
        clf = joblib.load('model/power_svm_gn5_movie.pkl')
    elif phone == 'pxldefault':
        clf = joblib.load('model/power_svm_pxl_default.pkl')
    elif phone == 'pxlpicture':
        clf = joblib.load('model/power_svm_pxl_picture.pkl')

    predicted_power = tr.PredictedPower(im, clf)
    kmeans_predicted_power = tr.PredictedPower(Kmeans_im,clf)
    if changed_im is 0:
        end_pixels = tr.toEndPixels(Kmeans_im)
    else:
        end_pixels = tr.toEndPixels(changed_im)

    R, G, B = np.mean(end_pixels, axis=0)

    """
    for the Clustered Color Palette
    """
    # pick n most used colors
    color_dict={}
    for index,row in df.iterrows():
        index = str(row['R'])+','+str(row['G'])+','+str(row['B'])
        color_dict[index]=row['counts']

    sorted_color_list = sorted(color_dict.items(), key=lambda x:x[1], reverse=True)

    colorUsage = []
    for i in range(min(len(color_dict),int(list_num))):
        most = sorted_color_list[i]
        most_rgb = list(map(int, most[0].split(',')))
        mR = most_rgb[0]
        mG = most_rgb[1]
        mB = most_rgb[2]
        most_power = clf.predict([[mR/255, mG/255, mB/255]]) * most[1] / (x * y)
        most_ratio = (most_power / kmeans_predicted_power * 100).round(ROUND_NUM)
        most_name = (("0x%0.2X" % mR)[2:] + ("0x%0.2X" % mG)[2:] + ("0x%0.2X" % mB)[2:]).lower()
        most_per = toRound(sorted_color_list[i][1] / (x * y) * 100)

        most_power = most_power.round(ROUND_NUM)

        colorUsage.append([i, most_rgb, most_power, most_ratio[0], most_name, most_per])

    """
    for the changed color Palette
    """
    changed_colorUsage=[]
    # pick n most used colors
    if changed_im is not 0:
        color_dict={}
        for index,row in changed_df.iterrows():
            index = str(row['R'])+','+str(row['G'])+','+str(row['B'])
            color_dict[index]=row['counts']

        sorted_color_list = sorted(color_dict.items(), key=lambda x:x[1], reverse=True)

        for i in range(min(len(color_dict),int(list_num))):
            most = sorted_color_list[i]
            most_rgb = list(map(int, most[0].split(',')))
            mR = most_rgb[0]
            mG = most_rgb[1]
            mB = most_rgb[2]
            most_power = clf.predict([[mR/255, mG/255, mB/255]]) * most[1] / (x * y)
            most_ratio = (most_power / kmeans_predicted_power * 100).round(ROUND_NUM)
            most_name = (("0x%0.2X" % mR)[2:] + ("0x%0.2X" % mG)[2:] + ("0x%0.2X" % mB)[2:]).lower()
            most_per = toRound(sorted_color_list[i][1] / (x * y) * 100)

            most_power = most_power.round(ROUND_NUM)

            changed_colorUsage.append([i, most_rgb, most_power, most_ratio[0], most_name, most_per])


    # RGBP
    # To send rounded R, G, B, Power value to index.html
    # create parameter Red Green Blue, then send R,G,B value
    RGBP = [toRound(R), toRound(G), toRound(B), toRound(predicted_power)]

    # find similar colors

    # similar_color_list = [[[similar colors], num], [], ... []]
    # ex) [[[[0, 0, 0], [0, 2, 0]], 5], [[[255, 255, 255], [248, 255, 255]], 14]]

    similar_color_list = []
    similar_color_num = []
    simColorUsage = []
    #colorUsage.sort(key=lambda x:x[2], reverse=True)
    #simColorUsage.sort(key=lambda x:x[2], reverse=True)

    '''
    Recommended Image
    '''
    if Kvalue is None:
        trImage = im
    else:
        if changed_im is 0:
            trImage = Kmeans_im 
        else:
            trImage = changed_im

    trImage.save("uploads/translated_image.jpg")

    trPower = tr.PredictedPower(trImage, clf)
    trRate = (predicted_power - trPower) / predicted_power * 100

    trInfo = [toRound(trPower), toRound(trRate)]

    list_max = len(sorted_color_list)
    simlist_max = len(similar_color_list)
    check = [phone, translate, list_num, list_max, simlist_max, mc, Kvalue]

    return render_template('index.html', check = check, filename = filename, RGBP = RGBP, colorUsage = colorUsage, simColorUsage = simColorUsage, trInfo = trInfo, changed_colorUsage = changed_colorUsage)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=13000)
