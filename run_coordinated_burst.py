from features import df_features
from model import load_data
from work_flow import prepare_xy,make_holdout_split
from net import eval_params,train_net,eval_net
from sklearn.metrics import classification_report
x,y=prepare_xy(df_features)
train_pool, test_holdout = make_holdout_split(df_features, "coordinated_burst", legit_sample_size=2000)
x_train, y_train = prepare_xy(train_pool)
x_test, y_test = prepare_xy(test_holdout)
model,criterion,train_df,test_df=eval_params(x_train, y_train, x_test, y_test)
model=train_net(y_train, train_df, model, criterion, epochs=10)
all_labels,all_preds=eval_net(test_df, model, criterion)
print(classification_report(all_labels, all_preds))