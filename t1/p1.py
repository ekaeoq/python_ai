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

# ova sljedeca funkcija zgleda dosta intimidating, ali stvarno nije dok uzmes dobrih 10 minut da sve prokuzis
# dodatni opis budem napravil s screenshot-om

def plot_image(i, predictions_array, true_label, img):
    # nist novo, sve kaj smo gore vec objasnili
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    
    # kaj ova fukncija radi: stavljamo ovo isto gore argmax da dobimo prediction koji bu LABEL imal najveci postotak slicnosti
    # dok taj predicted_label bude jednak tome s NAJVISE slicnosti stavili budemo ga kao true_label(naravno ne kao vrijednost)
    # jer se vrijednost true_label-a ne mijenja ali mi sami sebi da pokazemo aha to je plavo, jer u tim true_labelima ko 
    # argument stavaljmo unutra test_labels(onih 10 kaj znam vec kroz cijeli projekt)
    # BIG TIP: stavil bum dodatno objasnjenje toga sa slikicom
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else: 
        color = 'red'
    
    # tu samo ispijemo u graf keyword-e lmao, a upravo oni na dopustaju pa da si i najveca budale na svetu da razmes kaj tocno
    # ta fukncija radi u kojem djelu
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


# imamo tu sad jos jedan graf koji je manje vise prvi dio isto sranje, a ovo dole su mathlab bullshit-i koji su 
# pretty strightforward i neda mi se ta sranja objasnjavat
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# sad stavimo parametre nutra u funkcije koje smo slozili i to je pretty much to, unutra stavljamo ove "varijable"
# koje znamo od prije pa nam ja jasnije kaj se gore dogada opet(ak si videl to prije lamo xd)
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# pokazali smo slikice, kak to dela, sve lepo krasno, ali sad idemo stavit sliku nutra i zapravo koristit nas
# trained model
# koristil bum novu varijablu, samo radi claritija, makar je ovo samo u fuknciji ~img

img_test = test_images[1]
img_test = (np.expand_dims(img_test, 0))
# ovo gore znaci da stavljamo sliku u "batch", ko neki collection, makar je sama, tak to radi lamo
# pa prinatmo to da vidimo, (1, 28, 28) jedna slikica 28x28 jedna slikica 28x28
print(img_test)

single_prediction = probability_model.predict(img_test)
print(single_prediction)

# evo kraja, brate mili ako je neko ovo celo procital, svaka ti cast, doslovno napisi mi poruku na instagram(ekaeoo)lmao
# zadnji graf koji nam pokaze kaj model misli da ta slikica je:
plot_value_array(1, single_prediction[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# jedino za kraj kaj bi jos htel napravit je napravit side-by-side graf sa slikom koju imamo i tim zadnji predictionom
# ok peace out
