import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV



def xgb_train(train_full_data,test_full_data):
    X_df_train = train_full_data.copy()
    X_df_test = test_full_data.copy()
    del X_df_train['counter_volume']
    del X_df_train['date']
    del X_df_test['date']
    X_df_test['holiday_SS'] =  0
#     X_df_test['month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12'] = 0
    X_df_test = X_df_test.reindex(columns=list(X_df_test.columns)+['month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12'], fill_value=0)
    X_df_test = X_df_test[list(X_df_train.columns)]
    X_train,y_train = np.array(X_df_train),np.array(train_full_data.counter_volume)
    X_test = np.array(X_df_test)
    xgb_clf = xgb.XGBRegressor() 
#     parameters = {'n_estimators': [500, 1000, 1500], 'max_depth':[2,3,4,5]}
#     grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=5, n_jobs=-1)
#     print("parameters:")
#     pprint.pprint(parameters)
    xgb_clf.fit(X_train, y_train)
#     best_parameters=grid_search.best_estimator_.get_params()
#     for param_name in sorted(parameters.keys()):
#         print("\t%s: %r" % (param_name, best_parameters[param_name]))
    y_pred = xgb_clf.predict(X_test)
    feature_imp = xgb_clf.feature_importances_
    return y_pred,feature_imp

y_pred,feature_imp = xgb_train(train_full_data4,test_full_data4)

### 保存结果并提交
y_pred_end = []
for i in list(y_pred):
    if i < 0:
        y_pred_end.append(0)
    else:
        y_pred_end.append(i)
# test_data['counter_volume'] = list(y_pred)t

test_data['counter_volume'] = y_pred_end

test_data.to_csv('result.csv',encoding='utf-8',index=False)

import xlab
     
xlab.ftcamp.submit('/clever/result.csv')  # file_path为所需提交的文件绝对路径
