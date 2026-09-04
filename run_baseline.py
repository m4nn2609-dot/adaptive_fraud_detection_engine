from sklearn.model_selection import train_test_split
from features import df_features
from model import load_data
from work_flow import prepare_xy
from net import eval_params,train_net,eval_net
from sklearn.metrics import classification_report
x,y=prepare_xy(df_features)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y)
model,criterion,train_df,test_df=eval_params(x_train, y_train, x_test, y_test)
model=train_net(y_train, train_df, model, criterion, epochs=10)
all_labels,all_preds=eval_net(test_df, model, criterion)
print(classification_report(all_labels, all_preds))



