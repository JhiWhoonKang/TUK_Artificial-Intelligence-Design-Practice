import numpy as np

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
def feature_10(input_data):
    # 특징 후보 10번 : Anti-Diagonal 원소배열 추출 => 0의 개수
    anti_input_data = np.diag(np.fliplr(input_data))
    output_value = np.sum(anti_input_data == 0)
    return output_value
###############################################################################

##
##
##