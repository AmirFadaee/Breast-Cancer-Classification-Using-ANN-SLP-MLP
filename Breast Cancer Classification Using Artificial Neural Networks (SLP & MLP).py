# %% [markdown]
# SLP

# %%
#pip install tensorflow

# %%
from sklearn.datasets import load_breast_cancer

# %%
cancer = load_breast_cancer()
cancer

# %%
x = cancer.data
y = cancer.target
x

# %%
y

# %%
from sklearn.model_selection import train_test_split
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
x_train


# %%
x_test

# %%
y_train

# %%
from keras.models import Sequential # این لایه ها رو پشت سر هم طراحی میکنه
from keras.layers import Dense  # fully connected layer


# %%
x.shape[1],

# %%
modelSLP = Sequential()
modelSLP.add(Dense(units=1,activation='sigmoid' , input_shape=(x.shape[1],))) 

# %%
modelSLP.summary()

# %%
#pip install keras-visualizer 
#pip install graphviz pydot

# %%
modelSLP.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
historySLP = modelSLP.fit(x_train,y_train,epochs=60,validation_data=(x_test,y_test))

# %%
historySLP.history

# %%
import matplotlib.pyplot as plt

# %%
plt.plot(historySLP.history['accuracy'] , label='train')
plt.plot(historySLP.history['val_accuracy'],label ='val')
plt.show()

# %%
from sklearn.preprocessing import StandardScaler

# %%
ss = StandardScaler()

# %%
x_train_ss = ss.fit_transform(x_train)
x_test_ss = ss.transform(x_test)


# %%
modelSLP = Sequential()
modelSLP.add(Dense(units=1,activation='sigmoid',input_shape=(x.shape[1],)))

# %%
modelSLP.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
historySLP_Scaled = modelSLP.fit(x=x_train_ss,y=y_train,epochs=60,validation_data=(x_test_ss,y_test))

# %%
plt.plot(historySLP_Scaled.history['accuracy'],label = 'train')
plt.plot(historySLP_Scaled.history['val_accuracy'],label ='val')
plt.legend()
plt.show()

# %% [markdown]
# MLP

# %%
modelMLP = Sequential()
modelMLP.add(Dense(units=32,activation='relu',input_shape=(x.shape[1],)))
modelMLP.add(Dense(units=16,activation='relu'))
modelMLP.add(Dense(units=8,activation='relu'))
modelMLP.add(Dense(units=1,activation='sigmoid'))

# %%
modelMLP.summary()

# %%
modelMLP.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
historyMLP = modelMLP.fit(x= x_train,y=y_train,epochs=60,validation_data=(x_test,y_test))

# %%
plt.plot(historyMLP.history['accuracy'] , label ='train')
plt.plot(historyMLP.history['val_accuracy'],label = 'val')
plt.legend()
plt.show()

# %%
modelMLP_Scaled = Sequential()
modelMLP_Scaled.add(Dense(units=32,activation='relu',input_shape=(x.shape[1],)))
modelMLP_Scaled.add(Dense(units=16,activation='relu'))
modelMLP_Scaled.add(Dense(units=8,activation='relu'))
modelMLP_Scaled.add(Dense(units=1,activation='sigmoid'))

# %%
modelMLP_Scaled.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %%
historyMLP_Scaled = modelMLP_Scaled.fit(x= x_train_ss,y = y_train,epochs=60,validation_data=(x_test_ss,y_test))

# %%
plt.plot(historyMLP_Scaled.history['accuracy'],label = 'train')
plt.plot(historyMLP_Scaled.history['val_accuracy'],label = 'val')
plt.legend()
plt.show()

# %%



