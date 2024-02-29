import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

diccionario = pickle.load(open('data.pickle','rb'))

datos = np.asarray(diccionario['data'])
etiquetas = np.asarray(diccionario['labels'])

x_train, x_val, y_train, y_val = train_test_split(datos,etiquetas,test_size=0.2,shuffle=True,stratify=etiquetas)

modelo = RandomForestClassifier()

modelo.fit(x_train,y_train)

predriccion_y = modelo.predict(x_val)

porcentaje = accuracy_score(predriccion_y,y_val)
print('{}% de clasificacion'.format(porcentaje * 100))

f = open('model.p','wb')
pickle.dump({'model':modelo},f)
f.close()