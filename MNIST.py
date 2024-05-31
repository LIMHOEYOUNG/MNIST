import gzip
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import time

def fetch(url, path):
    fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()
    else:
        with open(fp, "wb") as f:
            data = requests.get(url).content
            f.write(data)
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()

path = './data' #경로
"""
MNIST 데이터셋의 이미지 파일은 각 이미지마다 16바이트의 헤더 정보를 가지고 있습니다. 
MNIST 데이터셋의 라벨 파일은 각 라벨마다 8바이트의 헤더 정보를 가지고 있습니다. 
[0x10:] = 16(16진수)'을 사용하여 메타데이터를 건너뛰고 이미지가 2D 배열(reshape((-1, 28 * 28) ))로 재구성됩니다.
reshape()의 '-1'이 의미하는 바는, 변경된 배열의 '-1' 위치의 차원은 "원래 배열의 길이와 남은 차원으로 부터 추정"
예를 들면 원소가 12개의 원소가 들어있는 배열 x를 재구조화 하면
x.reshape(-1, 1) => shape(12,1) 가 된다.

라벨에 [8:]를 하여  메타데이터를 건너뜀니다.
"""
train_images_sample = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", path)[0x10:].reshape((-1, 28 * 28))
train_labels_sample = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", path)[8:]
test_images = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", path)[0x10:].reshape((-1, 28 * 28))
test_labels = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", path)[8:]


train_sample = len(train_images_sample);   #학습데이터의 크기

# 학습데이터에서 8:2비율로 학습과 검증 데이터로 나누기 위해서 변수를 구해 만들어 놓는다.
split_data = int(train_sample*0.8)

# 8의 비율의 학습 데이터 설정합니다.
train_images = train_images_sample[:split_data]
train_labels = train_labels_sample[:split_data]

# 2의 비율의 검증 데이터 설정합니다
validation_images = train_images_sample[split_data:]
validation_labels = train_labels_sample[split_data:]


"""
X = X / 255` 및 `X_test = X_test / 255`은 이미지의 픽셀 값을 255로 나누어 정규화합니다. 
이 단계는 일반적으로 원래 범위인 0- 255에서 0-1의 범위로 신경망의 train과정을 개선하는 데 도움을 줍니다.
Local Minima에 빠질 위험 감소(학습 속도 향상)
"""
train_images = train_images / 255
test_images = test_images / 255
validation_images = validation_images/255

print("학습 데이터셋 크기: ", train_images.shape, train_labels.shape)
print("검증 데이터셋 크기: ", validation_images.shape, validation_labels.shape)
print("테스트 데이터셋 크기: ", test_images.shape, test_labels.shape)




# 랜덤하게 섞은 학습 데이터 중 맨 앞에있는 데이터를 시각화
image = train_images[0][0:].reshape(28, 28)  # 이미지를 2차원 배열로 변형
label = train_labels[0]                  # 이미지에 대한 라벨은 label에 저장합니다.

plt.imshow(image, cmap='gray')  # plt.imshow()를 사용하여 이미지를 표시하고
plt.title(f'Label: {label}')    # plt.title()을 사용하여 제목을 레이블로 설정하고
plt.axis('off')                 # plt.axis('off')를 사용하여 축을 제거한다.
plt.show()                      # plt.show()를 사용하여 시각화한 이미지를 보여준다.

epochs = 10 #10번 돕니다.
learning_rate = 0.001   #학습률이 0.001일때 가장 안정적인 그래프가 출력됨

class CrossEntropy:
    def forward(self, answer, pred):
        self.pred = pred # 예측값
        self.answer = np.zeros(10)  #0~9
        self.answer[answer] = 1
        self.loss = -np.sum(self.answer * np.log(pred)) #크로스 엔트로피 손실 계산
        return self.loss

    #역전파
    def backward(self):
        dz = -self.answer / self.pred   #크로스엔트로피 미분
        return dz

#소프트 맥스
class Softmax:
    def forward(self, x):
        x = x - np.max(x)
        self.p = np.exp(x) / np.sum(np.exp(x)) # softmax 확률 계산
        return self.p

    #Jacobian matrix (행렬(벡터)의 1차 편미분)를통해 GD를 계산
    #그레디언트와 마찬가지로 일차미분을 나타내는 것은 동일하나 그레디언트는
    #다변수 스칼라 함수에 대한 일차 미분이면 야코비안행렬은 다변수 벡터에 관한 일차 미분입니다.
    def backward(self, dz):
        #대각 행렬의 성분을 추출하여
        jacobian = np.diag(dz)
        for i in range(len(jacobian)):
            for j in range(len(jacobian)):
                if i == j:  #대각선일 경우 입력된 값에 대한 출력의 변화율 계산
                    jacobian[i][j] = self.p[i] * (1 - self.p[j])
                else: # 대각선 이외의 값들
                    jacobian[i][j] = -self.p[i] * self.p[j]

        return dz@jacobian


class ReLu:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dz):
        dz[self.x < 0] = 0
        return dz



class MLP:  #플 커넥트
    def __init__(self, input, output):  #입력크기와 출력크기
        # 가중치 초기화: He 초기화(He initialization)라고도 알려진 초기화 방법 He 초기화는 ReLU 활성화 함수와 함께 사용할 때 좋은 성능을 보이는 초기화 방법7
        self.W = np.random.normal(scale=1, size=(output, input)) * np.sqrt(2 / (input + output))
        #np.sqrt(2 / (in_size + out_size))는 표준편차계산위한 값


        self.b = np.zeros(output) #신경망의 바이어스 벡터 b를 0으로 초기화합니다. b의 크기는 출력 레이어의 단위 수와 일치합니다.

    def forward(self, x):
        self.x = x
        return np.dot(self.W, self.x) + self.b  # y = ax+b

    def backward(self, dz, learning_rate):    #학습률을 기반으로 가중치와 편향을 조절한다.
        self.dW = np.outer(dz, self.x)  # 가중치에 대한 미분 값을 계산하여 self.dW에 저장합니다.

        self.db = dz #편향에 대한 미분 값

        self.dx = np.dot(dz, self.W) # 이전 레이어로 전파될 미분 값을 계산하여 self.dx에 저장

        # 가중치와 편향을 업데이트 한다.
        self.W = self.W - learning_rate * self.dW
        self.b = self.b - learning_rate * self.db

        return self.dx


class assignmentModel:
    def __init__(self):

        # 입력층 레이어가 입력크기가 784(이미지 형태가 28*28)이고 출력이 100인 젓 번째 연결 레이어를 생성합니다.
        # 활성화 함수는 ReLu()함수를 사용합니다.
        self.f1_layer = MLP(784, 100)
        self.a1_layer = ReLu()

        #위 레이어의 출력인 100을 입력으로 사용하고 출력을 50을 생성하는 두 번째 레이어를 생성합니다.
        #이 레이어 다음 레이어
        self.f2_layer = MLP(100, 50)
        self.a2_layer = ReLu()


        #이 레이어 또한 이전 레이어와 마찬가지로 위 레이어의 출력인 50을 입력으로 사용하고 출력은 0~9까지의 숫자를 하나를 판별하기 위해서 10으로 설정합니다.
        # Softmax는 출력 값을 확률로 변환하여 모든 클래스에 대한 확률의 합이 1이 되도록 합니다(0~9까지의 숫자중 무조건 1개는 되어야 하기 떄문).
        # 이를 통해 모델은 각 입력에 대한 클래스에 대한 확률 분포를 제공합니다.
        self.f3_layer = MLP(50, 10)
        self.a3_layer = Softmax()



    #순방향을 진행합니다.
    """
    'forward' 방식은 입력 'x'를 각 층에 순차적으로 전달하고 각각의 활성 함수(ReLu)를 적용하고 마지막 output에서는 소프트맥스를 적용하여 최종 출력(0~9)을 생성하여 신경망을 통한 순방향 전파를 수행합니다.
    """
    def forward(self, x):
        net = self.f1_layer.forward(x)
        net = self.a1_layer.forward(net)

        net = self.f2_layer.forward(net)
        net = self.a2_layer.forward(net)

        net = self.f3_layer.forward(net)
        net = self.a3_layer.forward(net)

        return net


    """
    'backward'는 역전파(체인룰)를 이용하여 가중치와 편향을 업데이트합니다.
    """
    def backward(self, dz, learning_rate):
        dz = self.a3_layer.backward(dz)
        dz = self.f3_layer.backward(dz, learning_rate)

        dz = self.a2_layer.backward(dz)
        dz = self.f2_layer.backward(dz, learning_rate)

        dz = self.a1_layer.backward(dz)
        dz = self.f1_layer.backward(dz, learning_rate)

        return dz

    #연결된 노드와 노드사이에 가중치 출력
    def showWeight(self):
        print("W1 = ",self.f1_layer.W)
        print("W2 = ",self.f2_layer.W)
        print("W3 = ",self.f3_layer.W)

#트레이닝과 테스트 데이터셋을 돌면서 정확도를 측정한다.
def check_acc(images, labels, model):
    acc = 0.0
    dataSize = len(images)  #정확도를 측정할 데이터의 크기
    for i in range(dataSize):
        y_h = model.forward(images[i])    # y_h 는 각 클래스(0~9)에 대한 예측 확률 측정한다.
        y = np.argmax(y_h)  #가장 높은 값
        if y == labels[i]:  #결과와 예측값이 같으면 정확도가 1=100%
            acc += 1.0
    return acc / len(labels)

def modelTrain(model, train_images,validation_images ,test_images, train_labels, validation_labels,test_labels, epochs, learning_rate, loss_func):
    loss_tr_array = []    #train손실 그래프
    loss_val_array = []   #검증 쇤실값
    acc_tr_array = []     #학습 정확도
    acc_val_array = []    #검증 정확도
    acc_test_array = []   #test정확도

    for epoch in range(epochs):


        start = time.time()
        print(f"\nEpoch: {epoch} 진행")

        loss_train = 0     # 학습 손실값
        loss_validation = 0    # 검증 손실값

        train_mix = list(range(len(train_images))) # train_images-1 만큼의 리스트를 생성
        np.random.shuffle(train_mix)    # numpy.random.shuffle() 함수를 사용하여 학습 데이터와 검증 데이터를 무작위로 섞습니다.

        for i in range(len(train_images)):
            x = train_images[train_mix[i]]    #랜덤한 학습 이미지 데이터 입력
            y = train_labels[train_mix[i]]    #랜덤한 학습 라벨 데이터 입력
            pred = model.forward(x)     #순전파를 통한 데이터 예측
            loss = loss_func.forward(y, pred)   #예측한 데이터로 부터 손실값을 구하기 위한 역잔판 실행
            loss_train += loss                     #학습이 진행 될 떄마다 손실값이 누적
            dz = model.backward(loss_func.backward(), learning_rate)    #학습률과 학습을 통해 나온 손실값들을 역전파진행하여 가중치와 편향을 업데이트
        train_end = time.time()

        val_mix = list(range(len(validation_images)))
        np.random.shuffle(val_mix)
        #학습과 손실값 구하는 것은 동일하지만 검증이기 떄문에 가중치 업데이트는 하지 않습니다.
        for i in range(len(validation_images)):
            x = validation_images[val_mix[i]]
            y = validation_labels[val_mix[i]]
            pred = model.forward(x)
            loss = loss_func.forward(y, pred)
            loss_validation += loss

        loss_train /= len(train_images)
        loss_tr_array.append(loss_train)
        acc_train = check_acc(train_images, train_labels, model)
        acc_tr_array.append(acc_train)

        loss_validation /= len(validation_images)
        loss_val_array.append(loss_validation)
        acc_val = check_acc(validation_images, validation_labels, model)
        acc_val_array.append(acc_val)

        acc_test = check_acc(test_images,test_labels,model)
        acc_test_array.append(acc_test)

        model.showWeight()
        print("\ntrain 정확도: ",acc_train," / train Loss: ",loss_train)
        print("Valid 정확도: ",acc_val," / Valid Loss: ",loss_validation)
        print("test 정확도: ",acc_test)
        print("Epoch: ",epoch," 학습시간 : ",train_end-start)


    show_graphs(loss_tr_array, "Loss train","Loss", 1)
    show_graphs(loss_tr_array, "Loss valid","loss", 2)
    show_graphs(acc_tr_array, "Acc train", "Acc", 3)
    show_graphs(acc_val_array, "Acc valid","Acc", 4)
    show_graphs(acc_test_array, "Acc test", "Acc",5)
    plt.show()


def show_graphs(array, name,y_label, result):
    plt.figure(1,figsize=(20,4))
    plt.subplot(1, 5, result)
    plt.plot(array)
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    plt.title(name)


def show_examples(model, test_images,test_labels):
    for test in range(1, 10):    #9개의 데스트 이미지를 출력하기 위한 반복문
        plt.subplot(3, 3, test)  # 5x5 그리드 레이아웃으로 그림 내에 서브플롯을 생성하고 test는 현재 서브플롯에 인덱스를 뜻합니다.
        random_number = int(np.random.uniform(0, test_images.shape[0]))  #0부터 60000중 숫자 하나
        predicted = np.argmax(model.forward(test_images[random_number])) #제공된 모델을 사용하여 무작위로 선택된 예시(x_test[random_number]) 예측합니다. np.argmax는 확률이 가장 높은 클래스(0~9)의 인덱스를 얻기 위해 사용됩니다.
        plt.title("pred: "+str(predicted)+" label: "+str(test_labels[random_number]))
        plt.imshow(test_images[random_number].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.axis('off') #그래프 축과 라벨 제거
    plt.show()


total_start = time.time()  # 시작 시간 저장
trainModel = assignmentModel()
loss_func = CrossEntropy()  #손실값
modelTrain(trainModel, train_images, validation_images,test_images, train_labels, validation_labels, test_labels,epochs, learning_rate, loss_func)
print("총 학습 시간:", time.time() - total_start)  # 현재시각 - 시작시간 = 실행 시간
show_examples(trainModel, test_images,test_labels)    #학습한 모델 test