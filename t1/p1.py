import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# skinuli smo sranja za trainat network, PREK network hihihaha funny
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# svaka ta slika ima neki correspoding(ti labeli) integer(0-9) pa smo napravi class shit
# da moremo poslje to koristim s tim labelima, nama lakse ne kompjuteru
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#tu vidis kolko je zaprvo tih slika, 60k vs onih 10k s kojim gledas kolko je tak tocno
print(len(train_images))

# --plt.figure(figsize=(2,2))
# imshow s tim zaparvo odredujes koju sliku hoces pokazat -> ne radi plt.show() bez toga npr
# --plt.imshow(train_images[1])
# grid ti jednostavno prek slike stavi grid, mores stavit nutra false pa onda nebu, kaj je
# isto ko da jednostavno ne stavis pa neznam y
# --plt.grid()
# sa strane na slici dobis boje, ko neki bar boja, opet neznam zakaj bi mi to useful bilo
# --plt.colorbar()
# a bez show-a nebres nist videt xd, fukncija sluzi da pokazes to kaj god da si posral prije
# --plt.show()


# prije nego idemo feedat taj model, scaleamo slike na 0-1 skalu umjesto 0-255
# kolko je pixela, tho mislim da je to samo da nama bude lakse, again kompjuteru svejendo
train_images = train_images / 255.0
test_images = test_images / 255.0

# displayamo te slike, koje trebas skuzit kak to dela lol, al to je manje vise
# neznanje matplotsranja, a ne tensora il bilo ceg drugog
# mathplotlib - generative art (reminder)

plt.figure(figsize=(10,10))
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # plt.subplot(ako to tu stavis, na onom zadnje ti ostane grid, jer majmun glupi loopa
    # pa ne stigne ovo zadnje applayat na prvo, tho ne kuzim zakaj na zadnje ostane a ne na
    # prvom), moral bi provjerit kak tocno loopat po tom gridu
    # --plt.xlabel(i) - brojeva loopa normalno kak bi trebal, al ovo gore ne
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
    # -dok stavim samo train_label, to je samo array brojeva, pa onda dok to dole
    # puknes dobis zapravo class_names[broj] od tog === name
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Original-code tensorflow website
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    # pretvara 2d array u 1d array tak da zapravo sve te pixele svrstva v jen array, a znas
    # kak izgleda 1d array, boli kompjutera dal su mu pixeli slozeni, a ne ko nama budale smo,
    # vj ovi doli algoritmi ti isto srede kaj pojma nemam kj delaju
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    # layeri koji keepaju score o tome kolko je accurate to jednoj od tih 10 klasa - again
    # nije nekaj kaj prevec kuzim
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
        # https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
        # jao meni majke moje
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# imamo train_images i train_labels i pokusavamo dvoje spojit, trazimo od modela da predvidi slike s 
# labelima, i moramo verifaja da prediciton matcha labele, kaj again ne kuzim u potpunosti kak, ali 
# prema ovoj gore funkciji, optimizer, loss, metrics, ... , loss nam mjeri kolko je model tocan,
# a optimiter updatea sranja based na podacima koje vidi i loss fuknciji
model.fit(train_images, train_labels, epochs=10)

# test_dataset vs train_dataset, desava se overfitting zato jer test dataset nema nist "prije" radi samo s informacijama koje je
# dobil od shit train_dataset-a, a cesto taj train_dataset zapmati sranja(nije istina, zapamti sve kaj mu velimo da zapamti, samo kaj to negativno impacta unknown dataset)
# koja nebi treba, odnosno, zapamti "noise" i detalje koje
# onda negativno utjecu na nevideni SET, jer nezna bolje

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
print(test_loss)

# softmax pretvara 'logits' u postotke da je nama lakse razumijet, again nama, ne kompjuteru
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

# stavimo nutra nase slike da nam "predicta" to
predictions = probability_model.predict(test_images)
# dobimo array s 10 vrijednosti, znaci tih 0-9 mjesta u arrayu i svaka vrijednost predstavlja kolko je zapravo slicno cemu
# dal je nekaj slicnu labelu 0-carapa il kaj vec ili 9-majca ili kaj vec
print(predictions[0])
# funkcija argmax nam daje kojem ja najslicnija, znaci ovo da destom mjestu u arrayju je imalo 9.78 kaj je ekvivalent sranju na labelu broj 9
print(np.argmax(predictions[0]))
