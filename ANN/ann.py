# %% Veri setinin hazirlanmasi ve preprocessing
from keras.datasets import mnist #load mnist
from keras.utils import to_categorical # kategorik verilere cevirme
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential #sirali model
from keras.layers import Dense # bagli katmanlar

from keras.models import load_model #load model

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# mnist veri setini yukle, egitim ve test veri seti olarak ayri ayri yukle
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# ilk birkac ornegi gorsellestir
plt.figure(figsize=(10,5))

for i in range(6):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(f"index:{i}, Label: {y_train[i]}")
    plt.axis("off")
plt.show()


# veri setini normalize edelim, 0-255 aralığındaki pixel degerlerini 0-1 arasina olceklendiriyoruz
x_train = x_train.reshape((x_train.shape[0],x_train.shape[1]*x_train.shape[2])).astype("float32")/255
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1]*x_test.shape[2])).astype("float32")/255


# etiketleri kategorik hale cevir (0-9 arasindaki rakamlari one-hot encoding yapiyoruz)
y_train = to_categorical(y_train, 10) #10 = sinif sayisi
y_test = to_categorical(y_test, 10)


# %% ANN modelinin oluşturulmasi ve derlenmesi
model = Sequential()

# ilk katman: 512 cell, Relu Activation fuction, input size 28*28 = 784
model.add(Dense(512, activation="relu", input_shape = (28*28,)))

# ikinci katman: 256 celli activation: tanh
model.add(Dense(256, activation="tanh"))

# output layer: 10 tane olmak zorunda, activation softmax (2 tane olsaydı sigmoid kullanırdık)
model.add(Dense(10, activation="softmax"))

model.summary()

# model derlemesi: optimizer (adam: buyuk veri ve kompleks aglar icin idealdir)
# model derlemesi: loss (categorical_crossentropy)
# model derlemesi: metrik (accuracy)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])


# %% Callback'lerin tanimlanmasi ve ANN eğitilmesi

# Erken durdurma: eger val_loss iyilesmiyorsa egitim durduralim
# monitor: dogrulama setindeki (val) kaybi (loss) izler
# potience: 3 -> 3 epoch boyunca val loss degismiyorsa erken durdurma yapalim
# restore_best_weights: en iyi modelin agirlikleri geri yukler

"""
 epoch-1 -> %90 val_loss= 10
 epoch-2 -> %80 val_loss= 8
 epoch-3 -> %80 val_loss= 8
 epoch-4 -> %80 val_loss= 8
"""

early_stopping = EarlyStopping(monitor = "val_loss", patience=5, restore_best_weights=True)

# model checkpoint: en iyi modelin agirliklarını kaydeder
# save_best_only: sadece en iyi performans gosteren modeli kaydeder
checkpoint = ModelCheckpoint("ann_best_model.h5", monitor="val_loss", save_best_only=True)

# model training: 10 epochs, bath size = 64, dogrulama seti orani = %20
# model 60000 veri setini her biri 60 parcadan olusan 1000 kerede train edecek ve biz buna 1 epoch diyecegiz
# ama validasyon ayrimi yaptigimiz icin aslinda 60000 train verisi degil 48000 train verisi var bu nedenle,
# her biri 60 parcadan olusan 800 kerede train yapilacak
history = model.fit(x_train, y_train, #train veri seti
          epochs=10, # model toplamda 10 kere veri setini gorecek yani veri seti 10 kere egitilecek
          batch_size = 60, # veri seti 60'erli parcalar ile egitim yapilacak
          validation_split=0.2, #egitim verisinin %20 si dogrulama verisi olarak kullanilacak
          callbacks=[early_stopping, checkpoint])


# %% Model evaluation, gorsellestirme, model save and load

# test verisi ile model performansi degerlendirme
# evaluate: modelin test verisi uzerindeki loss (test_loss) ve accuracy (test_acc) hesaplar
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test acc: {test_acc}, test loss: {test_loss}")

# training and validation accuracy gorsellestir
plt.figure()
plt.plot(history.history["accuracy"],marker= "o", label = "Training Accuracy")
plt.plot(history.history["val_accuracy"],marker= "o" ,label = "Validation Accuracy")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()


# training and validation loss gorsellestirme
plt.figure()
plt.plot(history.history["loss"], marker= "o", label = "Training Loss")
plt.plot(history.history["val_loss"], marker= "o", label = "Validation Loss")
plt.title("ANN Accuracy on MNIST Data Set")
plt.xlabel("Epochs")
plt.ylabel("Acc")
plt.legend()
plt.grid(True)
plt.show()


# modeli kaydet
model.save("final_mnist_ann_model.h5")

loaded_model = load_model("final_mnist_ann_model.h5")

test_loss, test_acc = loaded_model.evaluate(x_test, y_test)
print(f"Loaded Model Result -> Test acc: {test_acc}, test loss: {test_loss}")

