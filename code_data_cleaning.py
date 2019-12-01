import pandas as pd
import numpy as np

### 得到唯一的节假日特征列
def get_multi_full(dataframe):
    date=list(pd.date_range(start=dataframe[0],end=dataframe[2]).astype('str'))
    frame=pd.DataFrame({'date':date},index=list(range(len(date))))
    frame['holiday_type']=dataframe[1]
    return frame
def get_full_data(input_data):
    data=input_data.copy()
    holiday=pd.read_csv('data/da0030/LV35_T_TDW_BOPR_OPR_T80_WKD_PARM_INF_20190130_000AH000.DAT',sep='|',header=None)
    holiday.loc[:,1]=holiday.loc[:,1].map(lambda x:x[1:])
    holiday=holiday.rename({0:'begin',1:'holiday_type',2:'end'},axis=1)
    holiday=holiday.drop([3,4],axis=1)
    holiday['end']=holiday['end'].map(lambda x:x[1:])

    single_holiday=holiday[holiday['begin']==holiday['end']]
    single_holiday=single_holiday.drop('end',axis=1).rename({'begin':'date'},axis=1)

    multi_holiday_index=set(holiday.index)-set(single_holiday.index)
    multi_holiday=holiday[holiday.index.isin(multi_holiday_index)].reset_index(drop=True)
    raw_multi=multi_holiday.apply(get_multi_full,axis=1)
    multi_holiday=pd.concat(list(raw_multi),axis=0).reset_index(drop=True)
    # ----这里得到了完整的节假日列表------
    full_holiday=pd.concat([single_holiday,multi_holiday],axis=0)

    full_holiday.date=full_holiday.date.astype('datetime64')
    data.date=data.date.astype('datetime64')


    ##### 处理日期对应多个节假日的情况
    ##查看节假日对应情况 
    # full_holiday.sort_values('date').groupby('date').count().holiday_type.value_counts()
    ##获得每天对应的不同节假日个数
    count_holiday = full_holiday.sort_values('date').groupby('date').count()

    # full_holiday.drop_duplicates(keep='first',inplace=True)
    ##获得只对应一个节假日的日期
    nor_ss_holiday = count_holiday[count_holiday['holiday_type'] <= 1]
    nor_ss_date = pd.merge(full_holiday,nor_ss_holiday,on = 'date')
    ##获得对应一个以上节假日的日期
    ss_holiday = count_holiday[count_holiday['holiday_type'] > 1]
    ss_date = pd.merge(full_holiday,ss_holiday,on = 'date')
    ##将对应多个节假日的日期命为SS（放假的工作日）
    ss_date['holiday_type_x'] = list(set(holiday.holiday_type))[1]
    ##拼接两表，形成一一对应的日期—节假日表
    new_full_holiday = pd.concat([nor_ss_date,ss_date])
    ##去重，改列名
    new_full_holiday.drop_duplicates(inplace=True)
    del new_full_holiday['holiday_type_y']
    new_full_holiday.rename(columns={'holiday_type_x':'holiday_type'},inplace=True) 
    # ----用left join合并------得到带节假日特征的总表
    full_data=pd.merge(data,new_full_holiday,how='left',on='date')
    def getweekday(frame):
        return frame.weekday()

    weekday_series=full_data.date.map(getweekday)

    full_data=pd.concat([full_data,pd.get_dummies(weekday_series,prefix='weekday')],axis=1)
    def getmonth(frame):
        return frame.month

    month_series=full_data.date.map(getmonth)
    month_series=pd.DataFrame({'month':list(month_series)})
    full_data=pd.concat([full_data,month_series],axis=1)
    full_data = pd.concat([full_data,pd.get_dummies(full_data['holiday_type'],prefix = 'holiday')],axis=1).drop(['holiday_type'],axis=1)
#     full_data = pd.merge(full_data,full_holiday.groupby('date').count(),how = 'left',on = 'date')
    
    
    return full_data


######缺失值处理
### 0填充+均值填充
# import pandas as pd
# # 读取训练数据
# data = pd.read_csv('train.csv', encoding='utf-8')

def zero_avg(data):
    t1 = data[data['type'] == 'typ1'].index
    t2 = data[data['type'] == 'typ2'].index
    t3 = data[data['type'] == 'typ3'].index
    t4 = data[data['type'] == 'typ4'].index
    t5 = data[data['type']=='typ5'].index
    t6 = data[data['type']=='typ6'].index

    typ4_dic = {'VTM_volume':0,'ATM_volume':data.loc[t4].ATM_volume.mean()}
    typ5_dic = {'VTM_volume':data.loc[t5].VTM_volume.mean(),'ATM_volume':data.loc[t5].ATM_volume.mean()}
    typ6_dic = {'VTM_volume':data.loc[t6].VTM_volume.mean(),'ATM_volume':data.loc[t6].ATM_volume.mean()}

    data.loc[t1] =  data.loc[t1].fillna(0)
    data.loc[t2] =  data.loc[t2].fillna(0)
    data.loc[t3] =  data.loc[t3].fillna(0)
    data.loc[t4] = data.loc[t4].fillna(value = typ4_dic)
    data.loc[t5] = data.loc[t5].fillna(value = typ5_dic)
    data.loc[t6] = data.loc[t6].fillna(value = typ6_dic)


### 0填充
# import pandas as pd
# # 读取训练数据
# data = pd.read_csv('train.csv', encoding='utf-8')
def zero(data):
    data = data.fillna(0)


### 0填充+dropna
# import pandas as pd
# # 读取训练数据
# data = pd.read_csv('train.csv', encoding='utf-8')
def zero_drop(data):
    t1 = data[data['type'] == 'typ1'].index
    t2 = data[data['type'] == 'typ2'].index
    t3 = data[data['type'] == 'typ3'].index
    t4 = data[data['type'] == 'typ4'].index
    t5 = data[data['type']=='typ5'].index
    t6 = data[data['type']=='typ6'].index

    typ4_dic = {'VTM_volume':0}
    data.loc[t1] =  data.loc[t1].fillna(0)
    data.loc[t2] =  data.loc[t2].fillna(0)
    data.loc[t3] =  data.loc[t3].fillna(0)
    data.loc[t4] = data.loc[t4].fillna(value = typ4_dic)
    data = data.dropna()





#### 读取原始数据，训练集+测试集
train_data = pd.read_csv('data/da0030/train.csv', encoding='utf-8').drop(['VTM_volume','ATM_volume'],axis=1)
test_data = pd.read_csv('data/da0030/predict.csv',encoding='utf-8')
train_full_data = get_full_data(train_data)
test_full_data = get_full_data(test_data)


### 转换id，month数据类型，便于onehot
train_full_data.id = train_full_data.id.astype('object')
test_full_data.id = test_full_data.id.astype('object')

train_full_data.month = train_full_data.month.astype('object')
test_full_data.month = test_full_data.month.astype('object')





### 获得按id按type按month分类的平均值—counter_volume
avg_id_type_month = train_full_data.groupby(['id','type','month'])['counter_volume'].mean().reset_index()
avg_id_type_month = avg_id_type_month.rename({'counter_volume':'avg_y'},axis=1)

# ### 获得按id按type分类的平均值—VTM and ATM
# train_data = pd.read_csv('data/da0030/train.csv', encoding='utf-8')
# zero_avg(train_data)
# avg_VTM = train_data.groupby(['id','type'])['VTM_volume'].mean().reset_index()
# avg_ATM = train_data.groupby(['id','type'])['ATM_volume'].mean().reset_index()

# ### 获得按id按type按month分类的
# def get_id_type_month_aver():
#     month_aver_data = pd.read_csv('data/da0030/train.csv', encoding='utf-8').fillna(0)
#     month_aver_data.date=month_aver_data.date.astype('datetime64')
#     month_aver_data['month']=month_aver_data['date'].map(lambda x :x.month)
#     counter_eachmonth=month_aver_data.groupby(['id','type','month'])['VTM_volume', 'ATM_volume','counter_volume'].mean().reset_index()
#     counter_eachmonth=counter_eachmonth.rename({ 'VTM_volume':'VTM_volume_month', 'ATM_volume':'ATM_volume_month', 'counter_volume':'counter_volume_month'},axis=1)
#     return counter_eachmonth

# avg_month_id_type_TM = get_id_type_month_aver()

#### join表增加特征列
train_full_data1 = pd.merge(train_full_data,avg_id_type_month,how = 'left',on = ['id','type','month'])
test_full_data1 = pd.merge(test_full_data,avg_id_type_month,how = 'left',on = ['id','type','month'])

# train_full_data2 = pd.merge(train_full_data1,avg_ATM,how = 'left',on = ['id','type'])
# test_full_data2 = pd.merge(test_full_data1,avg_ATM,how = 'left',on = ['id','type'])

# train_full_data3 = pd.merge(train_full_data2,avg_VTM,how = 'left',on = ['id','type'])
# test_full_data3 = pd.merge(test_full_data2,avg_VTM,how = 'left',on = ['id','type'])

# train_full_data4 = pd.merge(train_full_data3,avg_month_id_type_TM,how = 'left',on = ['id','type','month'])
# test_full_data4 = pd.merge(test_full_data3,avg_month_id_type_TM,how = 'left',on = ['id','type','month'])

train_full_data4 = pd.get_dummies(train_full_data1)
test_full_data4 = pd.get_dummies(test_full_data1)
