from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,ConfusionMatrixDisplay,classification_report
import numpy as np
import matplotlib.pyplot as plt

def eval(X_test,y_test,model,factorised_to_original):
    print('hit')
    # https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    y_pred = model.predict(X_test)
    print("************************** EVALS *****************************")
    #https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    y_predb = np.argmax(y_pred,axis=1)
    print("Precision score", precision_score(y_test, y_predb,average='micro'))
    print("Recall Score",recall_score(y_test, y_predb,average='micro'))
    print(  "F1 Score",f1_score(y_test, y_predb,average='micro'))
    print(classification_report(y_test, y_predb))
    cm = confusion_matrix(y_test,y_predb.round())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("************************** EVALS OVER *****************************")

