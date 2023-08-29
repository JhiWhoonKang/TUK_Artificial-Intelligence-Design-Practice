import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##
## 5개의 특징 추출 알고리즘 양식
##

###############################################################################
def feature_1(input_data):
    # 특징 후보 1번 : 가로축 Projection => 확률밀도함수로 변환 => 기댓값
    sum_input = np.sum(input_data, axis=0)
    total = np.sum(sum_input) 
    func = sum_input / total
    output_value = sum(sum_input * func)
    return output_value
###############################################################################
def feature_2(input_data):
    # 특징 후보 2번 : 가로축 Projection => 확률밀도함수로 변환 => 분산
    sum_input = np.sum(input_data, axis=0)
    total = np.sum(sum_input) 
    func = sum_input / total
    _func = np.sum(sum_input * func)
    output_value = sum((sum_input - _func)**2 * func)
    return output_value
###############################################################################
def feature_3(input_data):
    # 특징 후보 3번 : 세로축 Projection => 확률밀도함수로 변환 => 기댓값
    sum_input = np.sum(input_data.T, axis=0)
    total = np.sum(sum_input)
    func = sum_input / total
    output_value = np.sum(sum_input * func)
    return output_value
###############################################################################
def feature_4(input_data):
    # 특징 후보 4번 : 세로축 Projection => 확률밀도함수로 변환 => 분산
    sum_input = np.sum(input_data.T, axis=0)
    total = np.sum(sum_input)
    func = sum_input / total
    _func = np.sum(sum_input * func)
    output_value = sum((sum_input - _func)**2 * func)
    return output_value
###############################################################################
def feature_5(input_data):
    # 특징 후보 5번 : Diagonal 원소배열 추출 => 밀도함수로 변환 => 기댓값
    diag_input_data = np.diag(input_data)
    sum_input = np.sum(diag_input_data)
    func = diag_input_data / sum_input
    output_value = np.sum(diag_input_data * func)
    return output_value
###############################################################################
def feature_6(input_data):
    # 특징 후보 6번 : Diagonal 원소배열 추출 => 밀도함수로 변환 => 분산
    diag_input_data = np.diag(input_data)
    sum_input = np.sum(diag_input_data)
    func = diag_input_data / sum_input
    _func = np.sum(diag_input_data * func)
    output_value = sum((diag_input_data - _func)**2 * func)
    return output_value
###############################################################################
def feature_7(input_data):
    # 특징 후보 7번 : Diagonal 원소배열 추출 => 0의 개수
    diag_input_data = np.diag(input_data)
    output_value = np.sum(diag_input_data == 0)
    return output_value
###############################################################################
def feature_8(input_data):
    # 특징 후보 8번 : Anti-Diagonal 원소배열 추출 => 밀도함수로 변환 => 기댓값
    anti_input_data = np.diag(np.fliplr(input_data))
    sum_anti_input_data = np.sum(anti_input_data)
    func = anti_input_data / sum_anti_input_data
    output_value = np.sum(anti_input_data * func)
    return output_value
###############################################################################
def feature_9(input_data):
    # 특징 후보 9번 : Anti-Diagonal 원소배열 추출 => 밀도함수로 변환 => 분산
    anti_input_data = np.diag(np.fliplr(input_data))
    sum_anti_input_data = np.sum(anti_input_data)
    func = anti_input_data / sum_anti_input_data
    _func = np.sum(anti_input_data * func)
    output_value = np.sum((anti_input_data - _func)**2 * func)
    return output_value
###############################################################################
def feature_10(input_data):
    # 특징 후보 10번 : Anti-Diagonal 원소배열 추출 => 0의 개수
    anti_input_data = np.diag(np.fliplr(input_data))
    output_value = np.sum(anti_input_data == 0)
    return output_value
###############################################################################

##
##
##

def sigmoid(x):  # 시그모이드 함수
    return 1 / (1 + np.exp(-x))

class Two_Layer_Neural_Network:  # Two Layer Neural Network Class
    def __init__(self, Input, Hidden_Node, Output, Learning_rate, Test_set):
        self.Input_Node = Input.shape[1]  # 입력층 사이즈
        self.Hidden_Node = Hidden_Node  # 은닉층 노드 개수
        self.Output_Node = Output.shape[1]  # 출력층 사이즈
        self.x = Input  # 트레이닝 셋 입력
        self.y = Output  # 트레이닝 셋 출력(정답)
        self.Test_set = Test_set
        self.lr = Learning_rate  # Learning rate
        self.w_hidden = np.random.randn(self.Input_Node + 1, self.Hidden_Node) # 초기 w_hidden 랜덤 설정
        self.w_output = np.random.randn(self.Hidden_Node + 1, self.Output_Node) # 초기 w_output 랜덤 설정
        
    def Display_Inform(self):  # Input,Output 체크함수
        print('=============================================================')
        print('Input Layer Node 수: ', self.Input_Node)
        print('Hidden Layer Node 수: ', self.Hidden_Node)
        print('Output Layer Node 수: ', self.Output_Node)
        print('Selected Learning Rate Parameter: ', self.lr)
        print('=============================================================\n')
    
    def Display_Learning_outcome(self): # 학습 결과 Display
        print('\n=============================================================')
        print('Input Layer Node 수: ', self.Input_Node)
        print('Hidden Layer Node 수: ', self.Hidden_Node)
        print('Output Layer Node 수: ', self.Output_Node)
        print('Learning rate: ', self.lr)
        print('Epoch 수 : ', self.epoch)
        print('최종 MSE: ', self._MSE)
        print('최종 Test Data Accuracy: ', self._Test_Accuracy)
        print('Parameter w_hidden: ', self.w_hidden)
        print('Parameter w_output: ', self.w_output)
        print('=============================================================')
    
    def High_Accuracy(self): # 목표 정확도가 나타날 경우 Display
        print('\n-------------------------------------------------------------')
        print('현재 Epoch: ', self.epoch)
        print('현재 MSE: ', self._MSE)
        print('현재 Parameter w_hidden:\n', self.w_hidden)
        print('현재 Parameter w_output:\n', self.w_output)
        print('-------------------------------------------------------------\n')
    
    def y_predict(self, x): # 예측 y 계산
       sigmoid_input = sigmoid(np.dot(np.append(x, 1), self.w_hidden))
       Hidden_output = np.append(sigmoid_input, 1)
       self.Hidden_output = Hidden_output
       return sigmoid(np.dot(Hidden_output, self.w_output))
   
    def Back_propagation(self): # 역전파 알고리즘
        lr = self.lr  # Learning rate load
        for index in range(len(self.x)):
            y_hat = self.y_predict(self.x[index])  # y_predict 함수를 통해 예측값 변수 y_hat에 load
            Input = np.append(self.x[index], 1)  # Dummy Node 추가
            
            # 출력층 업데이트
            for index1 in range(self.Hidden_Node + 1):
                for index2 in range(self.Output_Node):
                    output_gradient = 2 * (y_hat[index2] - self.y[index][index2]) * y_hat[index2] * (1 - y_hat[index2]) * self.Hidden_output[index1] # 출력층 매개변수 기울기
                    self.w_output[index1][index2] = self.w_output[index1][index2] - lr * output_gradient  # w_output 업데이트
            
            # 은닉층 업데이트
            for index3 in range(self.Input_Node + 1):
                for index4 in range(self.Hidden_Node):
                    hidden_gradient = 0
                    for index5 in range(self.Output_Node):
                        hidden_gradient += 2 * (y_hat[index5] - self.y[index][index5]) * y_hat[index5] * (1 - y_hat[index5]) \
                            * self.w_output[index4][index5] * self.Hidden_output[index4] * (1 -self.Hidden_output[index4]) * Input[index3] # 은닉층 매개변수 기울기
                    self.w_hidden[index3][index4] = self.w_hidden[index3][index4] - lr * hidden_gradient # w_hidden 업데이트
    
    def Epoch(self, epoch): # Training
        self.epoch = epoch  # epoch load
        self.MSE = []  # MSE append
        self.Accuaracy = []  # 정확도 append
        self.Test_Accuracy = []  # 테스트 정확도 append
        
        for _epoch in range(epoch):
            data = np.concatenate([self.x, self.y], 1)  # x와 y dataset 결합
            np.random.shuffle(data)  # dataset suffle ## 일반화 성능을 높히고 overfitting을 방지
            self.x, self.y = np.hsplit(data, (self.Input_Node, )) # suffle 후 data 다시 분리
            self.Back_propagation()  # 역전파 과정으로 W업데이트
            
            # if ((_epoch + 1) % 100) == 0 or (_epoch + 1) == epoch: # 100회 마다 배열에 MSE, Accuracy 계산 및 Test ## 최적의 w 찾기 위함
            mse = []  # MSE 저장
            count = 0 # Accuracy count를 위한 변수
            for index in range(len(self.x)):
                y_hat = self.y_predict(self.x[index]) # 예측값 y 저장
                mse.append(np.mean((y_hat - self.y[index]) ** 2)) # 예측값과 실제값을 비교하여 MSE 계산 후 저장
                most_predicted = np.argmax(self.y_predict(self.x[index]))  # 가장 높은 값 확인 후 저장
                _one_hot = np.array([0] * self.Output_Node) # 예측값 배열 생성
                _one_hot[most_predicted] = 1 # 가장 높은 곳 지정하여 1 저장
               
                # 정확도
                if np.array_equal(_one_hot, self.y[index]):  # numpy.array_equal 활용 배열 비교 -> 같으면 count++
                    count += 1
            MSE = np.mean(mse) # 전제 데이터셋에 대한 MSE
            self._MSE = MSE # 마지막 MSE 정보 저장
            self.MSE.append(MSE) # MSE 저장
            
            Accuracy = count / len(self.x) # Accuracy 계산
            self.Accuaracy.append(Accuracy) # Accuracy 저장
                
            # Test(Validation)
            Test_x, Test_y = np.hsplit(self.Test_set, (self.Input_Node, ))
            count = 0
            for index in range(len(Test_x)):
                highperc = np.argmax(self.y_predict(Test_x[index]))
                test_one = np.array([0] * self.Output_Node)
                test_one[highperc] = 1
                if np.array_equal(test_one, Test_y[index]):
                    count += 1
            test_Accuracy = count / len(Test_x) # Test Accuracy 계산
            self._Test_Accuracy = test_Accuracy # 마지막 Test Accuracy 저장
            self.Test_Accuracy.append(test_Accuracy) # Test Accuracy 저장
            
            # 테스트 세트의 정확도가 95% 이상일 때 그때의 매개변수 w 저장
            if test_Accuracy > 0.95:
                self.High_Accuracy()
                # Weight Matrix .csv 파일로 저장
                w_hidden = pd.DataFrame(self.w_hidden)
                w_hidden.to_csv("highly_accuracy_w_hidden.csv", header = 'None') # highly_accuracy_w_hidden.csv
                w_output = pd.DataFrame(self.w_output)
                w_output.to_csv("highly_accuracy_w_output.csv", header = 'None') # highly_accuracy_w_output.csv
            
            print("Epoch %d || MSE : %f , Accuracy : %f , Test_Accuracy : %f" %(_epoch + 1, MSE, Accuracy, test_Accuracy))
        
        # Weight Matrix .csv 파일로 저장
        w_hidden = pd.DataFrame(self.w_hidden.T)
        w_hidden.to_csv("w_hidden.csv", header = 'None') # w_hidden.csv ## L by 6로 변환 위해 Transpose
        w_output = pd.DataFrame(self.w_output.T)
        w_output.to_csv("w_output.csv", header = 'None') # w_output.csv ## 3 by L+1로 변환 위해 Transpose

# 입력속성 저장되어있는 배열
feature = [0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8, feature_9, feature_10]

### 교수님 Hint code
x_train = np.array([],dtype ='float32')
x_train = np.resize(x_train, (0,5))
y_train = np.array([],dtype ='float32')
y_train = np.resize(x_train, (0,3))

## Data Set 생성
for Class in range(3): # Class는 0, 1, 2므로 for문을 통해 0~2 범위 반복
    for Index in range(500): # Class 당 데이터 개수는 500개이므로 for문을 통해 500회 반복
        # MNIST csv 파일 read    
        MNIST = str(Class) + '_' + str(Index + 1) + '.csv' # 첫 번째 데이터가 0이 아닌 1부터 시작하므로 Index + 1
        MNIST = pd.read_csv(MNIST, header = None).to_numpy(dtype ='float32')
        
        ## 입력 속성 선택
        x0 = feature[1](MNIST)
        x1 = feature[2](MNIST)
        x2 = feature[3](MNIST)
        x3 = feature[4](MNIST)
        x4 = feature[10](MNIST)

        ## 5개의 입력속성에 MNIST 입력에 따른 x, y 배열 생성
        x_total = np.array([[x0, x1, x2, x3, x4]], dtype ='float32') # MNIST를 각각의 feature 속성에 입력하여 (1, 5) 크기의 x 배열 생성
        y_total = np.array([[0] * 3]) # (1, 3) 크기의 배열 생성
        # One-Hot 인코딩
        y_total[0][Class] = 1 # Class에 맞게 One-Hot 인코딩 ## 클래스0 : [1, 0, 0], 클래스1 : [0, 1, 0], 클래스2: [0, 0, 1]
        
        # x_train, y_train에 각 클래스를 통해 생성된 x, y 배열 연결하여 하나의 Data set 구성
        x_train = np.concatenate((x_train, x_total), axis = 0) 
        y_train = np.concatenate((y_train, y_total), axis = 0)
### END

##데이터 셔플 및 Train,Test set 구분
DATAset = np.concatenate([x_train, y_train], 1) # 데이터 세트 결합
np.random.shuffle(DATAset) ## data set suffle ## 일반화 성능을 높히고 overfitting을 방지

Traning_set = DATAset[:1300] # Training set
Test_set = DATAset[1300:] # Test set
x_total, y_total = np.hsplit(Traning_set,(5, ))

Termproject = Two_Layer_Neural_Network(Input = x_total, Hidden_Node = 15, Output = y_total, Learning_rate = 0.001, Test_set = Test_set) # 조건 설정
Termproject.Display_Inform() # 초기 입력 조건 정보 Display
Termproject.Epoch(1000) # Epoch 설정 후 Training

##
## 출력
##
Termproject.Display_Learning_outcome() # 학습 결과 정보 Display

plt.figure(figsize=(30,30))

## 행을 3등분 하고 그래프 Display
plt.subplot(3,1,1) # 첫 번째 열 select
plt.xlabel('Epoch')
plt.ylabel("MSE")
plt.grid(True)
plt.plot(range(len(Termproject.MSE)),Termproject.MSE, 'b-')

plt.subplot(3,1,2) # 두 번째 열 select
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.grid(True)
plt.plot(range(len(Termproject.Accuaracy)),Termproject.Accuaracy, 'b-')

plt.subplot(3,1,3) # 세 번째 열 select
plt.xlabel('Epoch')
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.plot(range(len(Termproject.Test_Accuracy)),Termproject.Test_Accuracy, 'b-')

plt.show()