
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cmb_eval import cmb_eval
import os
import pickle
import gzip

def rf_classifer(moda, fold_name, train, train_data, test_data):

    X_train = train_data.values[:,:-1]
    y_train = train_data.values[:,-1].astype(int)

    X_test = test_data.values[:,:-1]
    y_test = test_data.values[:,-1].astype(int) 

    # from sklearn.utils import compute_sample_weight
    # sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

    current_dir = os.path.join(os.path.dirname(__file__))  # Folder
    weight_dir = os.path.join(current_dir, 'weights', moda)
    os.makedirs(weight_dir, exist_ok=True)
    weight_name = fold_name + '.pickle.gz'
    weight_path = os.path.join(weight_dir, weight_name)

    if train == True:
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)#, sample_weight=sample_weights)
        with gzip.open(weight_path, 'wb') as f:
            pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)  # Using highest protocol for efficiency
    else:
        with gzip.open(weight_path, 'rb') as f:
            clf = pickle.load(f)


    y_pred = clf.predict(X_test) 

    acc_dict, prec_dict, recal_dict, f1_dict = cmb_eval(y_pred, y_test)

    return clf, acc_dict, prec_dict, recal_dict, f1_dict