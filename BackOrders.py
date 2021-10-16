from flask import Flask, request, url_for, render_template
import numpy as np
import pickle
import pandas as pd
import os
import datetime
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__,template_folder="Template",static_folder="Static")

PT = pickle.load(open("transformer.pkl","rb"))
model = pickle.load(open("rf_importances.pkl","rb"))

@app.route('/')
def Index():
    return render_template('HomePage.html')

@app.route('/Results')
def Results():

    forecast_6_month = float(request.args.get("forecast_6_month"))
    sales_prev_6_month = float(request.args.get("sales_prev_6_month"))
    past_6_month_perf = float(request.args.get("past_6_month_perf"))
    national_inv = int(request.args.get("national_inv"))
    in_transit = int(request.args.get("in_transit"))
    local_bo_qty = int(request.args.get("local_bo_qty"))
    min_stock = int(request.args.get("min_stock"))
    lead = int(request.args.get("lead"))
    deck_risk = request.args.get("deck_risk")
    ppap_risk = request.args.get("ppap_risk")

    if str(deck_risk).lower() == "Yes":
        deck_risk = 1
    else:
        deck_risk = 0

    if str(ppap_risk).lower() == "Yes":
        ppap_risk = 1
    else:
        ppap_risk = 0

    ['sales_6_month',
 'forecast_6_month',
 'min_bank',
 'perf_6_month_avg',
 'lead_time',
 'national_inv',
 'in_transit_qty',
 'x1_Yes',
 'local_bo_qty',
 'x3_Yes']

    num_arr = np.array([sales_prev_6_month,forecast_6_month,min_stock,past_6_month_perf,lead,national_inv,in_transit,local_bo_qty])
    num_arr = PT.transform(num_arr.reshape(1,-1))[0]
    num_arr_1 = num_arr[:-1]
    num_arr_2 = num_arr[-1]
    arr = np.append(num_arr_1,deck_risk)
    arr = np.append(arr,num_arr_2)
    arr = np.append(arr,ppap_risk)
    result = model.predict(arr.reshape(1,-1))
    
    try:
        result = model.predict(arr.reshape(1,-1))

        if result == 0:
            result = "Not a Backorder"

        else:
            result = "A Backorder"
        with open("Logging/Logging.txt","a") as logger:
            date = str(datetime.datetime.fromtimestamp(round(datetime.datetime.utcnow().timestamp())))
            logger.write(f"{date} INFO-Prediction successful. Prediction made: {result}\n")
    except BaseException as logger:
        with open("Logging/Logging.txt","a") as logger:
            date = str(datetime.datetime.fromtimestamp(round(datetime.datetime.utcnow().timestamp())))
            logger.write(f"{date} INFO-Prediction successful. Prediction made: {result}\n")
            
    return render_template("Results.html",result=result)


if __name__ == "__main__":
    app.run(debug=True)
    
